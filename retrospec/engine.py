"""
RetroSpec v3 - corrected decoding engine.

The whole point of this file is the state invariant, which v2 got wrong:

    cache holds seq[: len(seq) - len(pending)]
    pending is the list of tokens that are in `seq` but NOT yet in the cache

After every step `pending == [seq[-1]]`, so the loop is uniform and the KV
cache never contains a token that was rejected. That is what makes the
speculative methods bit-exact against greedy decoding.

The driver loop is backend-agnostic (see the Verifier protocol) so it can be
unit-tested against a mock model with no GPU. See tests/test_engine.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Verifier backends
# --------------------------------------------------------------------------

class Verifier:
    """Protocol.

    forward(tokens, need_hidden) -> (preds, hidden)
        `tokens` are appended to the cache. `preds[j]` is the greedy prediction
        for the token that follows tokens[j]. `hidden[j]` is the last-layer
        hidden state of tokens[j] (or None when need_hidden is False).
    crop(n)      -> truncate the cache to exactly n positions
    cache_len()  -> current number of cached positions
    reset()      -> drop the cache
    """

    def forward(self, tokens: Sequence[int], need_hidden: bool = False): ...
    def crop(self, n: int) -> None: ...
    def cache_len(self) -> int: ...
    def reset(self) -> None: ...


class TorchVerifier(Verifier):
    """HuggingFace causal-LM backend."""

    def __init__(self, model, device=None):
        import torch

        self.torch = torch
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cache = None
        self._supports_cache_position = True

    def reset(self) -> None:
        self.cache = None

    def cache_len(self) -> int:
        return _cache_len(self.cache)

    def crop(self, n: int) -> None:
        self.cache = _crop_cache(self.cache, n)

    def forward(self, tokens: Sequence[int], need_hidden: bool = False):
        torch = self.torch
        base = self.cache_len()
        ids = torch.tensor([list(tokens)], dtype=torch.long, device=self.device)
        total = base + ids.shape[1]

        kwargs = dict(
            input_ids=ids,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            output_hidden_states=bool(need_hidden),
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
        )
        if self._supports_cache_position:
            kwargs["cache_position"] = torch.arange(base, total, device=self.device)

        with torch.no_grad():
            try:
                out = self.model(**kwargs)
            except TypeError:
                # older transformers without cache_position
                self._supports_cache_position = False
                kwargs.pop("cache_position", None)
                out = self.model(**kwargs)

        self.cache = out.past_key_values
        preds = out.logits[0].argmax(dim=-1).tolist()
        hidden = out.hidden_states[-1][0] if need_hidden else None
        return preds, hidden


# --------------------------------------------------------------------------
# Cache helpers (robust across transformers versions)
# --------------------------------------------------------------------------

def _cache_len(cache) -> int:
    if cache is None:
        return 0
    get_len = getattr(cache, "get_seq_length", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:
            pass
    layers = getattr(cache, "layers", None)
    if layers:
        keys = getattr(layers[0], "keys", None)
        if keys is not None:
            return int(keys.shape[-2])
    kc = getattr(cache, "key_cache", None)
    if kc:
        return int(kc[0].shape[-2])
    if isinstance(cache, (tuple, list)) and cache:
        return int(cache[0][0].shape[-2])
    return 0


def _crop_cache(cache, n: int):
    if cache is None:
        return None
    if _cache_len(cache) <= n:
        return cache

    crop = getattr(cache, "crop", None)
    if callable(crop):
        try:
            crop(n)
            return cache
        except Exception:
            pass

    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys = layer.keys[:, :, :n, :]
                layer.values = layer.values[:, :, :n, :]
        return cache

    kc = getattr(cache, "key_cache", None)
    if kc is not None:
        for i in range(len(kc)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :n, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :n, :]
        return cache

    if isinstance(cache, (tuple, list)):
        return tuple((k[:, :, :n, :], v[:, :, :n, :]) for k, v in cache)
    return cache


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class GenResult:
    token_ids: List[int]                 # newly generated tokens only
    latency_sec: float = 0.0
    forward_passes: int = 0              # verifier forwards == wall-clock driver
    drafted: int = 0
    accepted: int = 0                    # drafted tokens that survived verification
    corrections: int = 0
    bonus: int = 0
    empty_draft_steps: int = 0
    draft_sec: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Theoretical speedup ceiling. Baseline == 1.0."""
        return self.n_tokens / self.forward_passes if self.forward_passes else 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.n_tokens / self.latency_sec if self.latency_sec > 0 else 0.0

    def to_dict(self) -> dict:
        d = {
            "n_tokens": self.n_tokens,
            "latency_sec": self.latency_sec,
            "tokens_per_sec": self.tokens_per_sec,
            "forward_passes": self.forward_passes,
            "tokens_per_forward": self.tokens_per_forward,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance_rate": self.acceptance_rate,
            "corrections": self.corrections,
            "bonus": self.bonus,
            "empty_draft_steps": self.empty_draft_steps,
            "draft_sec": self.draft_sec,
        }
        d.update(self.extra)
        return d


# --------------------------------------------------------------------------
# The two generators
# --------------------------------------------------------------------------

def greedy_generate(
    verifier: Verifier,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 128,
    eos_ids: Iterable[int] = (),
    sync: Optional[Callable[[], None]] = None,
) -> GenResult:
    """Reference greedy decoding through the *same* backend as the speculative
    methods. Using this instead of model.generate() is what makes the speedup
    numbers honest."""
    return speculative_generate(
        verifier, prompt_ids, draft_fn=None,
        max_new_tokens=max_new_tokens, eos_ids=eos_ids, sync=sync,
    )


def speculative_generate(
    verifier: Verifier,
    prompt_ids: Sequence[int],
    draft_fn: Optional[Callable[[List[int]], Sequence[int]]] = None,
    max_new_tokens: int = 128,
    eos_ids: Iterable[int] = (),
    on_accept: Optional[Callable[[List[int], Any], None]] = None,
    need_hidden: bool = False,
    sync: Optional[Callable[[], None]] = None,
) -> GenResult:
    """Lossless greedy speculative decoding.

    With draft_fn=None this degenerates to plain greedy decoding, which is
    exactly what we want the baseline to be.

    on_accept(tokens, hidden) is called once per token, in order, never twice,
    with `hidden` aligned one-to-one with `tokens`. Datastore methods rely on
    this: datastore index i corresponds to seq[i].
    """
    eos = set(eos_ids or ())
    verifier.reset()

    seq: List[int] = list(prompt_ids)
    n_prompt = len(seq)
    pending: List[int] = list(seq)

    r = GenResult(token_ids=[])
    if sync:
        sync()
    t0 = time.perf_counter()

    while len(seq) - n_prompt < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - n_prompt)

        # ---- 1. draft ------------------------------------------------------
        draft: List[int] = []
        if draft_fn is not None and remaining > 1:
            td = time.perf_counter()
            draft = list(draft_fn(seq))[: remaining - 1]
            r.draft_sec += time.perf_counter() - td
        if not draft:
            r.empty_draft_steps += 1

        # ---- 2. one verifier forward over pending + draft ------------------
        tokens_in = pending + draft
        base = verifier.cache_len()
        preds, hidden = verifier.forward(tokens_in, need_hidden=need_hidden)
        r.forward_passes += 1

        P, K = len(pending), len(draft)
        r.drafted += K

        # ---- 3. longest matching prefix ------------------------------------
        # preds[P - 1 + i] is the verifier's own greedy choice for draft[i]
        n = 0
        while n < K and draft[n] == preds[P - 1 + n]:
            n += 1

        if n == K:
            nxt = preds[P - 1 + K]          # bonus token (or plain next token if K==0)
            accepted = draft + [nxt]
            keep = P + K
            r.accepted += K
            if K:
                r.bonus += 1
        else:
            nxt = preds[P - 1 + n]          # verifier overrides the draft here
            accepted = draft[:n] + [nxt]
            keep = P + n
            r.accepted += n
            r.corrections += 1
            verifier.crop(base + keep)      # <-- rejected tokens leave the cache

        # ---- 4. hidden states, aligned 1:1 with kept tokens -----------------
        if on_accept is not None:
            on_accept(list(tokens_in[:keep]), None if hidden is None else hidden[:keep])

        seq.extend(accepted)
        pending = [seq[-1]]                 # invariant restored

        hit_eos = False
        for i, t in enumerate(accepted):
            if t in eos:
                del seq[len(seq) - len(accepted) + i + 1:]
                hit_eos = True
                break
        if hit_eos:
            break

    if sync:
        sync()
    r.latency_sec = time.perf_counter() - t0
    r.token_ids = seq[n_prompt:]
    return r
