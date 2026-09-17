"""
Benchmark harness.

Timing bugs in v2 that made the numbers meaningless independently of the
algorithm bugs:

 1. device_map="auto". Kaggle's free tier is 2x T4. "auto" shards a 3B model
    across both GPUs, so every forward crosses PCIe twice. Enormous, noisy
    latency. Fixed: pin to one device.
 2. torch_dtype=bfloat16 hard-coded. T4 (Turing) and P100 (Pascal) have no
    bf16 tensor cores, so bf16 is emulated and slow. Fixed: fp16 below Ampere.
 3. Baseline used model.generate(); the speculative methods used a hand-written
    Python loop that also requested output_hidden_states=True on every pass.
    Two different code paths, one carrying extra work -- the comparison was
    never apples-to-apples. Fixed: one engine for everything, hidden states
    only for the method that needs them.
 4. No warmup, no torch.cuda.synchronize(), one sample per prompt, and each
    method ran in a separate process against a baseline recorded earlier, on a
    GPU in a different thermal/occupancy state. Fixed: one process, warmup,
    repeats, median, interleaved order, paired per-prompt ratios.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
from typing import Callable, Dict, Iterable, List, Optional

from .engine import GenResult, TorchVerifier, speculative_generate


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def describe_gpu() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"gpu": None, "n_gpu": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "n_gpu": torch.cuda.device_count(),
        "capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / 1024**3, 1),
        "bf16": props.major >= 8,
    }


def pick_dtype():
    """bf16 only on Ampere+. On T4/P100 bf16 has no hardware support and is
    materially slower than fp16."""
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16


def sync_fn():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.synchronize
    return lambda: None


def load_model(name: str, quantize_4bit: bool = False, device: str = "cuda:0",
               dtype=None, attn: str = "sdpa"):
    """Single-device load. Never device_map='auto' -- see note 1 above."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or pick_dtype()
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw = dict(attn_implementation=attn, low_cpu_mem_usage=True)
    try:
        kw["dtype"] = dtype                      # transformers >= 4.56
        model = AutoModelForCausalLM.from_pretrained(name, **_qcfg(kw, quantize_4bit, dtype, device))
    except TypeError:
        kw.pop("dtype", None)
        kw["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(name, **_qcfg(kw, quantize_4bit, dtype, device))

    if not quantize_4bit:
        model = model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tok


def _qcfg(kw, quantize_4bit, dtype, device):
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        kw["device_map"] = {"": device}
    return kw


def encode(tok, prompt: str, system_prompt: Optional[str] = None) -> List[int]:
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    msgs.append({"role": "user", "content": prompt})
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return tok(f"User: {prompt}\nAssistant:", add_special_tokens=True)["input_ids"]


def eos_ids_for(tok, model) -> List[int]:
    ids = set()
    for v in (tok.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    for s in ("<|eot_id|>", "<|end_of_text|>", "<|im_end|>"):
        try:
            i = tok.convert_tokens_to_ids(s)
            if isinstance(i, int) and i >= 0:
                ids.add(i)
        except Exception:
            pass
    return sorted(ids)


# --------------------------------------------------------------------------
# Running one method over the prompt suite
# --------------------------------------------------------------------------

def run_method(
    name: str,
    runner: Callable[[List[int]], GenResult],
    prompts: List[Dict],
    tok,
    max_new_tokens: int = 128,
    repeats: int = 3,
    warmup: int = 1,
    reset: Optional[Callable[[], None]] = None,
    lossless: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    rows = []
    if warmup and prompts:
        for _ in range(warmup):
            if reset:
                reset()
            runner(encode(tok, prompts[0]["prompt"], prompts[0].get("system_prompt")))

    for p in prompts:
        ids = encode(tok, p["prompt"], p.get("system_prompt"))
        runs: List[GenResult] = []
        for _ in range(repeats):
            if reset:
                reset()
            runs.append(runner(ids))
        # median run by latency; all runs are byte-identical for greedy methods
        best = sorted(runs, key=lambda r: r.latency_sec)[len(runs) // 2]
        row = {
            "method": name, "prompt_id": p["id"], "category": p.get("category", "general"),
            "grounded": p.get("grounded", False), "n_prompt_tokens": len(ids),
            "lossless": lossless,
            "latency_sec": statistics.median(r.latency_sec for r in runs),
            "latency_min": min(r.latency_sec for r in runs),
            "latency_iqr": (max(r.latency_sec for r in runs) - min(r.latency_sec for r in runs)),
            "text": tok.decode(best.token_ids, skip_special_tokens=True),
            "token_ids": best.token_ids,
        }
        row.update(best.to_dict())
        row["latency_sec"] = statistics.median(r.latency_sec for r in runs)
        row["tokens_per_sec"] = row["n_tokens"] / row["latency_sec"] if row["latency_sec"] else 0.0
        rows.append(row)
        if verbose:
            print(f"  [{name}] p{p['id']:<6} {row['n_tokens']:>3}tok "
                  f"{row['latency_sec']:.3f}s  {row['tokens_per_sec']:6.1f} tok/s  "
                  f"tok/fwd={row.get('tokens_per_forward', 0):.2f}  "
                  f"accept={row.get('acceptance_rate', 0):.2f}")
    return rows


def attach_quality(rows: List[Dict], baseline_rows: List[Dict]) -> List[Dict]:
    """Compare against the baseline on TOKEN IDS, not ROUGE.

    Every lossless method must score exact_match == 1.0. If it does not, there
    is a bug -- that is the point of the metric. v2 reported ROUGE-L, which
    degrades gracefully and therefore hid the cache-corruption bug completely.
    """
    base = {r["prompt_id"]: r for r in baseline_rows}
    for r in rows:
        b = base.get(r["prompt_id"])
        if not b:
            continue
        r["speedup"] = b["latency_sec"] / r["latency_sec"] if r["latency_sec"] else 0.0
        r["fwd_reduction"] = b["forward_passes"] / r["forward_passes"] if r["forward_passes"] else 0.0
        r["exact_match"] = int(r["token_ids"] == b["token_ids"])
        r["prefix_match"] = _prefix_match(r["token_ids"], b["token_ids"])
        try:
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            r["rouge_l"] = sc.score(b["text"], r["text"])["rougeL"].fmeasure
        except Exception:
            r["rouge_l"] = None
    return rows


def _prefix_match(a: List[int], b: List[int]) -> float:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n / max(len(b), 1)


def summarize(rows: List[Dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    agg = df.groupby("method").agg(
        n=("prompt_id", "count"),
        tokens=("n_tokens", "mean"),
        latency_s=("latency_sec", "mean"),
        tok_per_s=("tokens_per_sec", "mean"),
        speedup_mean=("speedup", "mean"),
        speedup_median=("speedup", "median"),
        tok_per_fwd=("tokens_per_forward", "mean"),
        acceptance=("acceptance_rate", "mean"),
        exact_match=("exact_match", "mean"),
        rouge_l=("rouge_l", "mean"),
        lossless=("lossless", "first"),
    ).sort_values("speedup_mean", ascending=False)
    return df, agg.round(3)


def sanity_check(agg) -> None:
    """Loudly flag the failure modes that produced v2's bad results."""
    print("\n--- sanity check ---")
    ok = True
    for m, r in agg.iterrows():
        if r.get("lossless") and m != "baseline" and r["exact_match"] < 1.0:
            print(f"  FAIL  {m}: lossless method but exact_match={r['exact_match']:.2f} "
                  "-> verification or KV-cache bug")
            ok = False
        if r.get("tok_per_fwd", 0) > 1.05 and r.get("speedup_mean", 0) < 1.0:
            print(f"  WARN  {m}: saves forward passes ({r['tok_per_fwd']:.2f} tok/fwd) but is "
                  "still slower -> drafting overhead dominates; shrink gamma or the drafter")
        if r.get("acceptance", 0) < 0.25 and m not in ("baseline",):
            print(f"  WARN  {m}: acceptance {r['acceptance']:.2f} is too low to pay for itself")
    if ok:
        print("  all lossless methods are bit-exact against the baseline")


def save(rows: List[Dict], path: str = "results/results.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {path}")


def free():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
