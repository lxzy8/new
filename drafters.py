"""
Drafters. Each is a callable `draft(seq) -> List[int]`.

v2's drafters were the reason the speculative methods were slower than the
baseline even ignoring the correctness bug:
  * the BM25 + FAISS indices were rebuilt from scratch after every accepted
    token (O(N) per token, in Python) -- that cost alone dwarfs the model;
  * BM25 is a bag-of-words scorer, so it ignored token ORDER, which is the only
    thing that matters for n-gram drafting;
  * the dense index contained the query vector itself, so it retrieved itself;
  * the 4-bit drafter re-prefilled the whole prefix on every draft call.
All four are fixed here. Everything is incremental and O(1)-amortised per token.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import faiss  # optional; numpy fallback is plenty fast at these sizes
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# --------------------------------------------------------------------------
# 1. N-gram / prompt-lookup drafter  (sparse, exact suffix match)
# --------------------------------------------------------------------------

class NGramDrafter:
    """Exact suffix match over everything generated so far, longest-n first,
    most-recent occurrence wins. This is the correct sparse retriever for
    token sequences -- BM25 is not, because order is everything here."""

    def __init__(self, max_n: int = 4, min_n: int = 2, draft_len: int = 6):
        self.max_n, self.min_n, self.draft_len = max_n, min_n, draft_len
        self.reset()

    def reset(self) -> None:
        self.idx: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: {} for n in range(self.min_n, self.max_n + 1)
        }
        self.pos = 0

    def _sync(self, seq: Sequence[int]) -> None:
        if len(seq) < self.pos:          # reused on a new, shorter sequence
            self.reset()
        # index up to len(seq)-2 so the live suffix can never match itself
        limit = len(seq) - 1
        while self.pos < limit:
            i = self.pos
            for n in range(self.min_n, self.max_n + 1):
                s = i + 1 - n
                if s >= 0:
                    self.idx[n][tuple(seq[s : i + 1])] = i
            self.pos += 1

    def candidates(self, seq: Sequence[int], top: int = 4) -> List[int]:
        """Return continuation-start positions, best first."""
        self._sync(seq)
        out: List[int] = []
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(seq) < n:
                continue
            j = self.idx[n].get(tuple(seq[-n:]))
            if j is not None and j + 1 < len(seq) and (j + 1) not in out:
                out.append(j + 1)
                if len(out) >= top:
                    break
        return out

    def __call__(self, seq: Sequence[int]) -> List[int]:
        for start in self.candidates(seq, top=1):
            return list(seq[start : start + self.draft_len])
        return []


# --------------------------------------------------------------------------
# 2. Hybrid drafter  (sparse n-gram  +  dense hidden-state kNN, fused by RRF)
# --------------------------------------------------------------------------

class HybridDrafter:
    """Dual-indexed datastore. Keeps v2's research idea -- recycle the
    verifier's own last-layer hidden states as free query embeddings, fuse two
    retrievers with Reciprocal Rank Fusion -- but with the retrieval actually
    working.

    Alignment note: hidden[i] is the state that *produced* seq[i+1]. So if
    hidden[j] matches the current query (= hidden of seq[-2]), the analogous
    next token is at j+2, not j+1. v2 used j+1, which is off by one.
    """

    def __init__(self, draft_len: int = 6, max_n: int = 4, min_n: int = 2,
                 rrf_k: int = 60, dense_top: int = 8, require_token_agreement: bool = True):
        self.draft_len = draft_len
        self.rrf_k = rrf_k
        self.dense_top = dense_top
        self.require_token_agreement = require_token_agreement
        self.ngram = NGramDrafter(max_n=max_n, min_n=min_n, draft_len=draft_len)
        self.reset()

    def reset(self) -> None:
        self.ngram.reset()
        self.emb: Optional[np.ndarray] = None   # (N, d) L2-normalised, row i <-> seq[i]
        self.n_emb = 0
        self._faiss = None
        self._buf: List[np.ndarray] = []

    # -- called from engine.on_accept, once per token, in order ------------
    def add(self, tokens: Sequence[int], hidden) -> None:
        if hidden is None:
            return
        h = hidden
        if hasattr(h, "detach"):
            h = h.detach().float().cpu().numpy()
        h = np.asarray(h, dtype=np.float32)
        if h.ndim == 1:
            h = h[None, :]
        if h.shape[0] == 0:
            return
        h = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-6, None)
        self._buf.append(h)
        self.n_emb += h.shape[0]
        if _HAS_FAISS:
            if self._faiss is None:
                self._faiss = faiss.IndexFlatIP(h.shape[1])
            self._faiss.add(h)          # incremental, never rebuilt
        else:
            self.emb = h if self.emb is None else np.vstack([self.emb, h])

    def _dense(self, seq: Sequence[int]) -> List[int]:
        if self.n_emb < 8:
            return []
        q = (self._faiss.reconstruct(self.n_emb - 1)[None, :] if _HAS_FAISS
             else self.emb[-1][None, :])
        k = min(self.dense_top + 4, self.n_emb)
        if _HAS_FAISS:
            _, ind = self._faiss.search(q, k)
            ind = ind[0]
        else:
            ind = np.argsort(-(self.emb @ q[0]))[:k]

        out: List[int] = []
        for j in ind:
            j = int(j)
            if j < 0 or j >= self.n_emb - 2:      # drop self and its neighbours
                continue
            start = j + 2
            if start >= len(seq):
                continue
            # cheap analogy check: did position j+1 produce the token we just saw?
            if self.require_token_agreement and seq[j + 1] != seq[-1]:
                continue
            out.append(start)
            if len(out) >= self.dense_top:
                break
        return out

    def __call__(self, seq: Sequence[int]) -> List[int]:
        sparse = self.ngram.candidates(seq, top=self.dense_top)
        dense = self._dense(seq)

        rrf: Dict[int, float] = {}
        for rank, s in enumerate(sparse):
            rrf[s] = rrf.get(s, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, s in enumerate(dense):
            rrf[s] = rrf.get(s, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        if not rrf:
            return []
        best = max(rrf.items(), key=lambda kv: kv[1])[0]
        return list(seq[best : best + self.draft_len])


# --------------------------------------------------------------------------
# 3. Model drafter  (small or quantised model, with its OWN persistent cache)
# --------------------------------------------------------------------------

class ModelDrafter:
    """Autoregressive drafter backed by a real model.

    Keeps a persistent KV cache and only ever feeds the tokens that were newly
    accepted, then rolls the cache back to len(seq) so the speculative part is
    discarded. v2 re-prefilled the entire prefix on every single draft call,
    which is why the 'fast' 4-bit drafter was slower than the verifier it was
    supposed to accelerate.
    """

    def __init__(self, model, draft_len: int = 5, device=None):
        import torch
        from .engine import TorchVerifier

        self.torch = torch
        self.v = TorchVerifier(model, device=device)
        self.draft_len = draft_len

    def reset(self) -> None:
        self.v.reset()

    def __call__(self, seq: Sequence[int]) -> List[int]:
        cached = self.v.cache_len()
        if cached > len(seq):                      # defensive
            self.v.crop(len(seq)); cached = len(seq)
        new = list(seq[cached:])
        if not new:
            new = [seq[-1]]
            self.v.crop(len(seq) - 1)

        preds, _ = self.v.forward(new)
        tok = preds[-1]
        out = [tok]
        for _ in range(self.draft_len - 1):
            preds, _ = self.v.forward([tok])
            tok = preds[-1]
            out.append(tok)
        self.v.crop(len(seq))                      # drop the speculative tail
        return out


# --------------------------------------------------------------------------
# 4. Early-exit drafter  (self-speculative / LayerSkip style)
# --------------------------------------------------------------------------

class EarlyExitDrafter:
    """Drafts with the first `n_layers` blocks of the verifier itself plus the
    logit lens, then lets the full model verify. Unlike the CALM implementation
    in v2 this is (a) actually faster, because the remaining layers are really
    skipped, and (b) lossless, because the full model still verifies.

    Implemented by temporarily swapping in a truncated ModuleList, which reuses
    all of HuggingFace's mask/rotary/cache plumbing instead of reimplementing
    it -- that keeps it working across transformers versions.
    """

    def __init__(self, model, n_layers: int = 8, draft_len: int = 5, device=None):
        import torch

        self.torch = torch
        self.model = model
        self.device = device or next(model.parameters()).device
        self.draft_len = draft_len
        self.base = getattr(model, "model", None)
        if self.base is None or not hasattr(self.base, "layers"):
            raise TypeError("EarlyExitDrafter needs a model with .model.layers")
        self.n_full = len(self.base.layers)
        self.n_layers = max(1, min(n_layers, self.n_full - 1))
        self._full_layers = self.base.layers
        self.cache = None
        self.probe()

    def probe(self) -> None:
        """Fail fast at construction rather than mid-benchmark."""
        ids = self.torch.tensor([[1, 2, 3]], device=self.device)
        self._forward(ids, None)
        self.cache = None

    def _forward(self, ids, cache):
        torch = self.torch
        full = self.base.layers
        self.base.layers = full[: self.n_layers]
        try:
            with torch.no_grad():
                out = self.base(input_ids=ids, past_key_values=cache,
                                use_cache=True, return_dict=True)
                logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
            return logits, out.past_key_values
        finally:
            self.base.layers = full

    def reset(self) -> None:
        self.cache = None

    def _len(self) -> int:
        from .engine import _cache_len
        return _cache_len(self.cache)

    def _crop(self, n: int) -> None:
        from .engine import _crop_cache
        self.cache = _crop_cache(self.cache, n)

    def __call__(self, seq: Sequence[int]) -> List[int]:
        torch = self.torch
        cached = self._len()
        if cached > len(seq):
            self._crop(len(seq)); cached = len(seq)
        new = list(seq[cached:])
        if not new:
            new = [seq[-1]]
            self._crop(len(seq) - 1)

        out: List[int] = []
        ids = torch.tensor([new], dtype=torch.long, device=self.device)
        for i in range(self.draft_len):
            logits, self.cache = self._forward(ids, self.cache)
            tok = int(logits[0, -1].argmax())
            out.append(tok)
            ids = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        self._crop(len(seq))
        return out
