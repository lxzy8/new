"""
Dataset loading.

v2 hard-coded 5 prompts, so every reported number had n=5 and no error bars.
This loads your 500-question CSV and, importantly, lets you tag prompts as
`grounded` or `open` -- retrieval-based drafting (n-gram, hybrid) only wins
when the answer reuses spans from the input, so mixing the two and reporting
one average is what hides the effect.
"""
from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional

QUESTION_COLS = ["question", "prompt", "query", "text", "instruction",
                 "input", "problem", "Question", "Prompt", "Query"]
CATEGORY_COLS = ["category", "type", "topic", "subject", "label", "domain"]
CONTEXT_COLS = ["context", "passage", "document", "article", "background"]

DEFAULT_PATHS = [
    "/kaggle/input/datasets/totaldose/ques-500/dataset_500.csv",
    "/kaggle/input/**/dataset_500.csv",
    "/kaggle/input/**/*.csv",
    "./dataset_500.csv",
]


def find_csv(path: Optional[str] = None) -> str:
    cands = [path] if path else []
    cands += DEFAULT_PATHS
    for p in cands:
        if not p:
            continue
        if os.path.isfile(p):
            return p
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        "No CSV found. Pass the path explicitly, e.g.\n"
        "  load_prompts('/kaggle/input/<your-dataset-slug>/dataset_500.csv')\n"
        "Run  !ls -R /kaggle/input | head -50  to see what is actually mounted."
    )


def _pick(cols: List[str], wanted: List[str]) -> Optional[str]:
    low = {c.lower().strip(): c for c in cols}
    for w in wanted:
        if w.lower() in low:
            return low[w.lower()]
    return None


def load_prompts(
    path: Optional[str] = None,
    n: Optional[int] = 60,
    seed: int = 0,
    question_col: Optional[str] = None,
    category_col: Optional[str] = None,
    context_col: Optional[str] = None,
    min_chars: int = 15,
    system_prompt: str = "You are a helpful assistant. Answer clearly and completely.",
) -> List[Dict]:
    """Return [{id, category, prompt, system_prompt, grounded}, ...]."""
    import pandas as pd

    csv = find_csv(path)
    df = pd.read_csv(csv)
    cols = list(df.columns)

    qcol = question_col or _pick(cols, QUESTION_COLS)
    if qcol is None:
        # fall back to the widest text column
        obj = [c for c in cols if df[c].dtype == object]
        if not obj:
            raise ValueError(f"No text column in {csv}. Columns: {cols}")
        qcol = max(obj, key=lambda c: df[c].astype(str).str.len().mean())

    ccol = category_col or _pick(cols, CATEGORY_COLS)
    xcol = context_col or _pick(cols, CONTEXT_COLS)

    rows: List[Dict] = []
    for i, r in df.iterrows():
        q = str(r[qcol]).strip()
        if len(q) < min_chars or q.lower() == "nan":
            continue
        ctx = str(r[xcol]).strip() if xcol and str(r[xcol]).strip().lower() != "nan" else ""
        prompt = f"{ctx}\n\nQuestion: {q}" if ctx else q
        rows.append({
            "id": int(i),
            "category": str(r[ccol]).strip() if ccol else "general",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "grounded": bool(ctx),
        })

    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
        rows.sort(key=lambda d: d["id"])

    print(f"Loaded {len(rows)} prompts from {csv}")
    print(f"  question column : {qcol!r}")
    print(f"  category column : {ccol!r}")
    print(f"  context column  : {xcol!r}  (grounded={sum(r['grounded'] for r in rows)})")
    return rows


def make_grounded_variants(rows: List[Dict], max_chars: int = 1200) -> List[Dict]:
    """Turn open questions into grounded ones by asking the model to work from
    a supplied passage. Use this to build the slice where retrieval drafting is
    actually expected to win -- without it, n-gram and hybrid will look flat and
    you will have proved nothing."""
    out = []
    for r in rows:
        body = r["prompt"][:max_chars]
        out.append({
            **r,
            "id": r["id"] + 100000,
            "category": r["category"] + "_grounded",
            "grounded": True,
            "prompt": (
                "Read the passage and answer using its exact wording wherever "
                "possible.\n\nPassage:\n" + body +
                "\n\nTask: Summarise the passage, then answer the question it raises."
            ),
        })
    return out
