# What was wrong with RetroSpecV2

Short version: it was not the dataset. There were two fatal bugs in
`core/verification.py` that made every speculative method produce different
text from the baseline, and four separate bugs that made them slower than the
baseline. A larger dataset would have measured the same broken system more
precisely.

Ranked by how much damage each one did.

---

## Tier 1 — the output was wrong

### 1. The KV cache kept rejected draft tokens (`verification.py:190-204`)

```python
keep = min(n_accepted, K)
target_kv_len = prev_kv_len + keep
past_key_values = _trim_cache(out_verify["past_key_values"], target_kv_len)
logits = verify_logits[:, keep - 1 : keep, :]
```

On a mismatch at draft index `n`, the code appends the verifier's correction
token `c` and keeps `n + 1` cache positions. But position `prev + n` holds the
key/value computed for `draft[n]` — the token that was just **rejected** — not
for `c`. From that point on the model attends to a token it never emitted.
`logits` is read from the same wrong position, so the next prediction is
conditioned on the rejected token too.

Correct behaviour: keep only the `n` positions that were actually accepted, and
carry `c` forward to be processed by the next forward pass.

### 2. The bonus token was emitted but never fed back

When the whole draft was accepted, `bonus = argmax(verify_logits[K-1])` was
appended to the sequence, but the loop then set `logits = verify_logits[K-1]` —
the distribution that *produced* the bonus. So the next iteration's `pred` was
the bonus token again, and the bonus token's own KV was never computed.

### Proof

I replayed exactly that bookkeeping against a deterministic mock LM where the
prediction is a pure function of the tokens sitting in the cache:

```
v2 logic: 60/60 runs diverge from greedy decoding
```

Not "sometimes", not "on hard prompts" — **always**. Every ROUGE-L number in
the v2 results table was measuring drift, not decoding quality.

### 3. ROUGE-L hid it

`metrics.py` scored quality with ROUGE-L against the baseline text. ROUGE
degrades gracefully, so a decoder that diverges after ten tokens still scores
~0.6 and looks like a tuning problem. For a *lossless* method the only correct
metric is token-id equality, which is either 1.0 or you have a bug. v3 scores
`exact_match` on token ids and the benchmark asserts on it.

### 4. `max_new_tokens` was silently exceeded

The draft was clipped to `remaining`, but the bonus token was appended *after*
that check. So a run ending in a full acceptance produced `max_new_tokens + 1`
tokens against the baseline's `max_new_tokens`, which alone made
`exact_match == False` even if everything else had been right.

---

## Tier 2 — the speed measurements were meaningless

### 5. `device_map="auto"` on a 2×T4 Kaggle box

`model_utils.py:10` defaults to `device_map="auto"`. Kaggle's free tier is two
T4s, so a 3B model gets **sharded across both GPUs** and every forward pass
crosses PCIe twice. Latency roughly doubles and the variance explodes. Pin to
one device.

### 6. `torch_dtype=bfloat16` hard-coded

T4 (Turing, sm_75) and P100 (Pascal, sm_60) have no bf16 tensor cores. bf16 is
emulated there and is materially slower than fp16. v3 picks bf16 only on
Ampere+ (`bench.pick_dtype`).

### 7. Baseline and speculative methods ran different code

Baseline used `model.generate()`. Everything else used a hand-written Python
loop that additionally passed `output_hidden_states=True` on **every** forward
— allocating 29 hidden-state tensors per pass on a 3B model, including during
prefill, for methods like `ngram` and `quantized_selfspec` that never used
them. Two different code paths, one carrying extra work. v3 runs the baseline
through the same engine with `draft_fn=None`, and requests hidden states only
for the hybrid method.

### 8. No warmup, no `cuda.synchronize()`, n=1, cross-process comparison

Each notebook was a separate process that loaded the model fresh and compared
its latency against a baseline JSON recorded earlier, on a GPU in a different
thermal and occupancy state. First-call CUDA kernel autotuning landed entirely
on prompt 1. One sample per prompt, five prompts total. v3 runs everything in
one process, interleaved, with warmup, three repeats, median latency, and
**paired per-prompt** speedup ratios.

### 9. `max_new_tokens=30`

Far too short for speculative decoding to reach steady state — prefill and
per-prompt setup dominate. 128 minimum.

---

## Tier 3 — the drafters could not have worked

### 10. The datastore rebuilt both indices after every accepted token

`datastore.py:71` calls `_rebuild_indices()` from `add_tokens()`, which
reconstructs the **entire** BM25 index and a **fresh** `faiss.IndexFlatIP` over
all tokens, from scratch, every single step. With a 200-token prompt that is
~200 Python-loop window constructions plus a full FAISS re-add, per token. This
cost alone dwarfs the 3B model's forward pass. `ngram` and `hybrid` could not
have been faster than baseline at any acceptance rate.

v3 indices are incremental: O(1) amortised per token, FAISS `add` only, never
rebuilt.

### 11. BM25 ignores token order

BM25 is a bag-of-words scorer. For n-gram speculative decoding the *only* thing
that matters is the order of the suffix. BM25 over stringified token ids
retrieves windows containing the same ids in any arrangement, and rewards rare
ids. The right sparse retriever here is exact suffix matching (prompt-lookup
decoding), which is what v3 uses — and it is also O(1) instead of O(N).

### 12. The dense index contained its own query

`search_dense` searched an index that already held the query vector, so the
nearest neighbour was almost always the query's own position. v3 excludes self
and its immediate neighbours.

### 13. Off-by-one in the dense continuation

`search_dense` returns `idx + 1`. But `hidden[j]` is the state that *produced*
`seq[j+1]`. If `hidden[j]` matches the current query, the analogous **next**
token is at `j + 2`. v3 also adds a cheap analogy check (`seq[j+1] == seq[-1]`)
which lifts acceptance noticeably.

### 14. Prompt tokens were added to the datastore twice

`notebook_2/3` call `datastore.add_tokens(input_ids)` before generation, and
then the engine's `on_verifier_forward` prefill callback adds the whole prompt
again. Duplicated context, and from then on datastore index `i` no longer
corresponded to sequence position `i`. Bonus tokens were never added at all, so
the two drifted further every step.

### 15. The 4-bit drafter re-prefilled the entire prefix on every draft call

`notebook_4:71` — `draft_model(curr_ids, use_cache=True)` with no persistent
cache. Every draft step re-ran a full prefill over the whole sequence on the
quantised model. v3's `ModelDrafter` keeps its own KV cache and rolls it back
after each draft.

### 16. 4-bit self-speculation cannot be a latency win

A 4-bit nf4 copy of the *same* model is not a cheaper drafter. At batch size 1
inference is memory-bandwidth-bound, and nf4 dequantisation overhead makes it
~2-3× **slower** than the bf16 original. It saves VRAM, not time. The method
that actually wins is a genuinely smaller drafter — **Llama-3.2-1B-Instruct**,
same tokenizer family as the 3B verifier. v3 keeps the 4-bit variant but
applies it to the 1B drafter and reports it as the memory-lean option.

### 17. CALM was guaranteed to lose on both axes

`calm_generator.py:39-46` runs a **full** forward and then optionally uses an
intermediate layer's logit-lens prediction. That is strictly more work than the
baseline (full forward *plus* an extra 128k-row `lm_head` projection), so its
speedup is bounded above by roughly 0.9× by construction, while its quality is
strictly worse. Nothing could have been tuned to fix it.

Real CALM skips the remaining layers, which leaves those layers with no KV
entries for the skipped position; fixing that requires synthesising K/V for the
skipped layers ("state propagation"), which is model surgery. v3 keeps this as
an honest **quality-drift measurement** and adds `EarlyExitDrafter`, which gets
the speed win *losslessly* by using the first half of the network as a drafter
and letting the full model verify.

---

## Was the dataset the problem?

Partly, but not the way you think. Five hard-coded prompts is no statistical
power, so yes — use the 500. But the more important point:

**N-gram and hybrid drafting only win when the output reuses spans from the
input.** Summarisation, RAG, code editing, long grounded contexts. On open Q&A
there is nothing to retrieve, so acceptance stays near zero and you will
measure ~1.0× no matter how good the implementation is.

If `dataset_500.csv` is plain questions, `ngram` and `hybrid` will look flat
and you will have proved nothing about them. That is why `data.py` has
`make_grounded_variants()` and the benchmark reports a grounded-vs-open
breakdown. The gap between those two slices *is* the finding for the
retrieval-based methods.
