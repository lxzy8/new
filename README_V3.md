# RetroSpec v3

Corrected engine + reliable notebooks. Read `DIAGNOSIS.md` first.

## Install into your repo

```
RetroSpecV2/
  retrospec/          <- new package (replaces core/)
  notebooks/
    01_correctness_and_calibration.ipynb
    02_full_benchmark.ipynb
  tests/test_engine.py
  DIAGNOSIS.md
  requirements.txt
```

Keep `core/` around if you want the before/after comparison for the write-up;
nothing in v3 imports it.

## Run order

```bash
python tests/test_engine.py        # no GPU needed, <1s, must pass
```

Then on Kaggle (GPU T4 x2 or P100), in order:

1. `01_correctness_and_calibration.ipynb` — asserts every lossless method is
   bit-exact against greedy decoding, then sweeps gamma per drafter and writes
   `configs/calibration_v3.json`.
2. `02_full_benchmark.ipynb` — point `CSV` at your dataset, run everything.

Notebook 1 is not optional. It is the gate that would have caught the v2 bug in
about ninety seconds.

## Methods

| method | lossless | what it is |
|---|---|---|
| `baseline` | — | greedy, through the same engine as everything else |
| `ngram` | yes | exact suffix lookup over the live sequence |
| `hybrid` | yes | n-gram + dense hidden-state kNN, fused with RRF |
| `spec_1b` | yes | Llama-3.2-1B drafter, 3B verifier |
| `spec_1b_4bit` | yes | same, nf4 drafter — memory-lean, not faster |
| `earlyexit_spec` | yes | first half of the verifier drafts, full model verifies |
| `calm` | **no** | logit-lens early exit; quality-drift measurement only |

## Requirements

`HF_TOKEN` as a Kaggle Secret (Llama-3.2 is gated). If you would rather not
deal with that, swap in `Qwen/Qwen2.5-1.5B-Instruct` + `Qwen/Qwen2.5-0.5B-Instruct`
— same tokenizer family, ungated, and the whole suite runs faster.
