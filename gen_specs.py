"""Generate one pinned eval_spec per (model, benchmark) cell.

The eval-lib contract pins exactly one model + one dataset per spec, so a
multi-model x multi-benchmark study is a grid of specs. This keeps every cell
honest: each run's golden record logs the exact model+dataset that produced it.
Run: python gen_specs.py  (re-run after editing MODELS/BENCHMARKS).
"""

import os

PAPER_ID = "sentiment-generalization"

# All binary sentiment classifiers (label 0=negative, 1=positive).
MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", "714eb0fa89d2f80546fda750413ed43d93601a13", "distilbert-sst2"),
    ("textattack/roberta-base-SST-2", "84ee248b91053ef5d0c748bbac4edfba1cf89584", "roberta-sst2"),
    ("textattack/distilbert-base-uncased-SST-2", "6fea14f6264ea28d8405573dac228b3e11137643", "distilbert-textattack"),
]

# Binary sentiment benchmarks (small test/validation splits).
# hf_id, revision, config, split, version, text_field, label_field, short
BENCHMARKS = [
    ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb", "", "validation", "2.0.0", "sentence", "label", "sst2"),
    ("cornell-movie-review-data/rotten_tomatoes", "aa13bc287fa6fcab6daf52f0dfb9994269ffea28", "", "test", "1.0.0", "text", "label", "rotten-tomatoes"),
]

TEMPLATE = """# Auto-generated cell: {mshort} on {bshort}. Edit gen_specs.py, not this file.
spec_version: "1.0.0"
paper_id: "{paper_id}"
task: "binary_sentiment"
metric_lib_version: "0.1.0"
dataset:
  hf_id: "{hf_id}"
  config: {config}
  split: "{split}"
  version: "{version}"
  revision: "{ds_rev}"
  text_field: "{text_field}"
  label_field: "{label_field}"
model:
  hf_id: "{model_id}"
  revision: "{model_rev}"
inference:
  seed: 42
  max_length: 256
  batch_size: 32
metrics:
  primary: "accuracy"
  secondary: ["f1"]
reproduce_target:
  accuracy: {{ min: 0.5, max: 1.0 }}
"""


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")
    os.makedirs(out, exist_ok=True)
    n = 0
    for mid, mrev, mshort in MODELS:
        for hf, drev, cfg, split, ver, tf, lf, bshort in BENCHMARKS:
            content = TEMPLATE.format(
                mshort=mshort, bshort=bshort, paper_id=PAPER_ID,
                hf_id=hf, config=(f'"{cfg}"' if cfg else "null"), split=split,
                version=ver, ds_rev=drev, text_field=tf, label_field=lf,
                model_id=mid, model_rev=mrev,
            )
            path = os.path.join(out, f"{mshort}__{bshort}.yaml")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            n += 1
            print("wrote specs/" + f"{mshort}__{bshort}.yaml")
    print(f"{n} cell specs ({len(MODELS)} models x {len(BENCHMARKS)} benchmarks)")


if __name__ == "__main__":
    main()
