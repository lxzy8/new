"""
CALM-style early exit.

Read this before using it, because v2's version could not possibly have shown a
speedup and the write-up did not say so:

  v2 ran a FULL forward pass, then optionally used an intermediate layer's
  logit-lens prediction instead of the final one. That is strictly more work
  than the baseline (full forward + an extra 128k-row lm_head projection), so
  its speedup is bounded above by ~0.9x *by construction*, while its quality is
  strictly worse. It could only ever lose on both axes.

Real CALM skips the remaining layers, which leaves those layers with no KV
entries for the skipped position. Fixing that needs "state propagation" --
synthesising K/V for the skipped layers from the exit hidden state -- which is
model-surgery and version-fragile.

So this module is deliberately scoped as a QUALITY-LOSS MEASUREMENT: how far
does the output drift, and how often is an early layer already confident. For
actual speed from early exit, use EarlyExitDrafter, which gets the win
losslessly by letting the full model verify.

`mode="truncated"` does genuinely skip layers and IS faster, at the cost of a
cache that only ever holds the first L layers -- i.e. the model becomes a
shallow model for the whole generation. It is a legitimate ablation ("how good
is the first half of the network on its own"), not CALM. Labelled as such.
"""
from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence

from .engine import GenResult, _cache_len, _crop_cache


def calm_generate(
    model,
    prompt_ids: Sequence[int],
    exit_layer: int = 16,
    confidence_threshold: float = 0.8,
    max_new_tokens: int = 128,
    eos_ids: Iterable[int] = (),
    mode: str = "oracle",          # "oracle" (full fwd, measures drift) | "truncated"
    device=None,
    sync=None,
) -> GenResult:
    import torch

    eos = set(eos_ids or ())
    device = device or next(model.parameters()).device
    base = model.model
    n_full = len(base.layers)
    L = max(1, min(exit_layer, n_full))

    seq = list(prompt_ids)
    n_prompt = len(seq)
    cache = None
    early_exits = 0
    steps = 0

    if sync:
        sync()
    t0 = time.perf_counter()

    with torch.no_grad():
        while len(seq) - n_prompt < max_new_tokens:
            cached = _cache_len(cache)
            new = seq[cached:] or [seq[-1]]
            ids = torch.tensor([new], dtype=torch.long, device=device)
            steps += 1

            if mode == "truncated":
                full = base.layers
                base.layers = full[:L]
                try:
                    out = base(input_ids=ids, past_key_values=cache,
                               use_cache=True, return_dict=True)
                finally:
                    base.layers = full
                cache = out.past_key_values
                logits = model.lm_head(out.last_hidden_state[:, -1:, :])[0, -1]
                tok = int(logits.argmax())
                early_exits += 1
            else:
                out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                            output_hidden_states=True, return_dict=True)
                cache = out.past_key_values
                h = base.norm(out.hidden_states[L][:, -1:, :])
                early = model.lm_head(h)[0, -1]
                p = torch.softmax(early.float(), dim=-1)
                conf, cand = torch.max(p, dim=-1)
                if float(conf) >= confidence_threshold:
                    tok = int(cand)
                    early_exits += 1
                else:
                    tok = int(out.logits[0, -1].argmax())

            seq.append(tok)
            if tok in eos:
                seq.pop()
                break

    if sync:
        sync()
    gen = seq[n_prompt:]
    r = GenResult(token_ids=gen, latency_sec=time.perf_counter() - t0,
                  forward_passes=steps)
    r.extra = {
        "early_exits": early_exits,
        "early_exit_ratio": early_exits / steps if steps else 0.0,
        "exit_layer": L,
        "n_layers": n_full,
        "confidence_threshold": confidence_threshold,
        "calm_mode": mode,
        "lossless": False,
    }
    return r
