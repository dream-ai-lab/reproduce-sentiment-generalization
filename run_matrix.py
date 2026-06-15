"""Run every (model, benchmark) cell through the standard eval-lib runner.

Each cell logs a normal MLflow run (same paper_id experiment), with the model
and dataset already captured as params by eval-lib — so the model x benchmark
matrix is just a pivot in the MLflow UI / search.py. We only write model_fn.
"""

import glob
import os
import random

import numpy as np
import torch
from transformers import pipeline

from eval_lib import load_spec, run_paper

SPECS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs", "*.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_id(label: str) -> int:
    """Map any model's label string to 0=negative / 1=positive."""
    u = label.upper().strip()
    if u.startswith("LABEL_"):
        return int(u.split("_")[1])
    if u.isdigit():
        return int(u)
    return {"NEGATIVE": 0, "NEG": 0, "POSITIVE": 1, "POS": 1}[u]


def build_model_fn(spec):
    clf = pipeline(
        "text-classification",
        model=spec.model.hf_id,
        revision=spec.model.revision,
        device=-1,
    )

    def model_fn(texts):
        outs = clf(
            list(texts),
            batch_size=spec.inference.batch_size,
            truncation=True,
            max_length=spec.inference.max_length,
        )
        return [to_id(o["label"]) for o in outs]

    return model_fn


def main():
    rows = []
    for sp in sorted(glob.glob(SPECS)):
        spec = load_spec(sp)
        set_seed(spec.inference.seed)
        model = spec.model.hf_id.split("/")[-1]
        bench = spec.dataset.hf_id.split("/")[-1]
        r = run_paper(sp, build_model_fn(spec), role="reproduce", run_name=f"{model}@{bench}")
        rows.append((model, bench, r["results"]["accuracy"], r["results"]["f1"]))
        print(f"[cell] {model:48} x {bench:16} acc={r['results']['accuracy']:.4f} f1={r['results']['f1']:.4f}")

    benches = sorted({b for _, b, _, _ in rows})
    models = sorted({m for m, _, _, _ in rows})
    print("\n=== accuracy matrix (model x benchmark) ===")
    print("model \\ benchmark".ljust(50) + "".join(b.ljust(18) for b in benches))
    for m in models:
        line = m.ljust(50)
        for b in benches:
            acc = next((a for mm, bb, a, _ in rows if mm == m and bb == b), None)
            line += (f"{acc:.4f}".ljust(18) if acc is not None else "-".ljust(18))
        print(line)


if __name__ == "__main__":
    main()
