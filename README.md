# RLHF for Automated Clinical Red-Teaming

**Audrey Tjokro · Stephen Dong · Niki Karanikola**  
Cornell University — CS 5788 Generative Models, Spring 2026

Multi-turn adversarial stress-testing for clinical LLMs, comparing **baseline / DPO / PPO** for training red-team agents on Med-Safety-Bench.

---

## Layout

```
configs/   YAML defaults for each method (source of truth for hyperparameters)
notebooks/ Colab driver only — no training logic lives here
src/       Python package (CLI, methods, results store, env capture)
results/   Local mirror of per-run artifacts; canonical store is GCS
```

The Colab notebook is just a launcher. All training, eval, and logging logic
lives in `src/`. To change a hyperparameter, edit a YAML — never the notebook.

## Setup (local)

```bash
git clone <this-repo>
cd rlhf-clinical-redteaming
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # then fill in OPENAI_API_KEY + GCS_BUCKET
```

For Google Cloud auth (results sync): `gcloud auth application-default login`.

## Reproduction commands

Every run takes `--config`, optionally `--run-name`, `--seed`, `--gcs-bucket`,
and any number of `--override path.to.key=value` entries. Anything passed via
`--override` wins over both the YAML and the typed flags.

### Baseline (no training; ASR / TTF on dev split)

```bash
python -m src baseline \
    --config configs/baseline.yaml \
    --run-name baseline-dev-v1 \
    --gcs-bucket gs://results_043026
```

### DPO (iterative; LoRA adapters checkpointed every outer iter)

```bash
python -m src dpo \
    --config configs/dpo.yaml \
    --run-name dpo-beta01-lr5e6 \
    --gcs-bucket gs://results_043026 \
    --beta 0.1 --lr 5e-6 --n-outer 4
```

### PPO (TRL 0.9.6; LoRA adapters checkpointed every 20 steps)

```bash
python -m src ppo \
    --config configs/ppo.yaml \
    --run-name ppo-kl0.3-lr1e5 \
    --gcs-bucket gs://results_043026 \
    --target-kl 0.3 --lr 1e-5 --n-train-steps 100
```

### Final (test-split) evaluation — append `--use-test` to any of the above

```bash
python -m src baseline --config configs/baseline.yaml --use-test \
    --run-name baseline-test --gcs-bucket gs://results_043026
```

### Querying past runs

Each run drops `run_record.json` at `gs://results_043026/<method>/<run-uuid>/`.
To load all runs into a DataFrame:

```python
import json, pandas as pd
from google.cloud import storage

bucket = storage.Client().bucket("results_043026")
records = []
for blob in bucket.list_blobs():
    if blob.name.endswith("run_record.json"):
        records.append(json.loads(blob.download_as_text()))
df = pd.json_normalize(records)
```


## Colab usage

Open `notebooks/run_cli.ipynb`. Edit the `CONFIG` cell to pick method + run
name, then **Runtime → Run all**. Artifacts auto-sync to GCS at the end.

## Storage layout (canonical = GCS)

```
gs://results_043026/
├── baseline/<run-uuid>/
│   ├── run_record.json      # config + env + final metrics
│   ├── trajectories.jsonl   # one record per evaluated trajectory
│   ├── eval_log.jsonl       # streaming per-trial summary
│   └── split_fingerprint.jsonl
├── dpo/<run-uuid>/
│   ├── run_record.json
│   ├── training_log.jsonl   # per-outer-iter metrics
│   └── checkpoints/iter_000/, iter_001/, ...   (LoRA adapters; every iter kept)
├── ppo/<run-uuid>/
│   ├── run_record.json
│   ├── training_log.jsonl   # per-step PPO stats
│   ├── trajectories.jsonl
│   └── checkpoints/step_20/, step_40/, ..., final/
└── data_rlhf/               # cached dataset shards (optional)
```
