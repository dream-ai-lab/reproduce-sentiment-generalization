# reproduce-sentiment-generalization

A **multi-model × multi-benchmark** study in one paper, built on the eval
standard. It measures how well sentiment classifiers generalise from their
training distribution (SST-2) to an out-of-domain benchmark (Rotten Tomatoes).

Created from [experiment-template](https://github.com/dream-ai-lab/experiment-template);
depends on a pinned [eval-lib](https://github.com/dream-ai-lab/eval-lib).

## The matrix

| | SST-2 (in-domain) | Rotten Tomatoes (out-of-domain) |
|---|---|---|
| distilbert-base-uncased-finetuned-sst-2-english | ✓ | ✓ |
| textattack/roberta-base-SST-2 | ✓ | ✓ |
| textattack/distilbert-base-uncased-SST-2 | ✓ | ✓ |

3 models × 2 benchmarks = **6 cells**. Each cell is its own pinned
`eval_spec.yaml` under `specs/` (one model + one dataset, real HF commit SHAs),
so every run's golden record is exact. All cells share `paper_id:
sentiment-generalization`, so they land in one MLflow experiment and the matrix
is a pivot on `params.model.hf_id` × `params.dataset.hf_id`.

## Run it

```bash
pip install -r requirements.txt
python gen_specs.py          # (re)generate specs/ from the model×benchmark lists
# log to the shared server:
export MLFLOW_TRACKING_URI=https://mlflow.note.transformerlabs.ai
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<pass>
python run_matrix.py         # runs all 6 cells, prints the accuracy matrix
```

## Add a model or benchmark

Edit `MODELS` / `BENCHMARKS` in `gen_specs.py`, re-run it, commit the new
`specs/*.yaml`, and `run_matrix.py` picks them up. CI validates every cell spec
against the pinned standard.

## What you only write

`model_fn` in `run_matrix.py` — a single generic sentiment classifier that maps
any model's label (`POSITIVE`/`NEGATIVE` or `LABEL_1`/`LABEL_0`) to 0/1.
Everything else (data loading, metrics, MLflow golden record) is eval-lib.
