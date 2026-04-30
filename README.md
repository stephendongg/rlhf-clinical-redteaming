# RLHF for Automated Clinical Red-Teaming

**Audrey Tjokro · Stephen Dong · Niki Karanikola**
Cornell University — CS 6782 Generative Models, Spring 2026

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

## Switching the judge (OpenAI ↔ local HF)

The judge is selected per-run via two YAML fields (overridable from the CLI):

```yaml
judge_backend: openai            # or hf_local
judge_model:   gpt-4o-mini       # or e.g. meta-llama/Llama-3.1-8B-Instruct
```

The `hf_local` backend loads the model once per run with the same 4-bit nf4 +
bf16 + sdpa setup as the attacker/target, then reuses it across calls. Both
backends return the same dict schema (same `score_trajectory_judgment`
output), so swapping them does not require any code change.

### Recommended HF judge models

| Model | Size (4-bit) | Notes |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | ~5 GB | Default recommendation. **Gated** — accept the license on HF and set `HF_TOKEN`. |
| `Qwen/Qwen2.5-14B-Instruct`        | ~9 GB | Stronger reasoning; ungated. Don't pair with a Qwen attacker (same-family bias). |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~5 GB | Lightweight fallback; weaker JSON adherence. |

### GPU memory budget (40 GB A100)

| Run | Models on GPU | Total (4-bit) | Fits 40 GB? |
|---|---|---|---|
| baseline + openai judge | attacker + target | ~10 GB | ✓ |
| baseline + hf_local judge | attacker + target + judge | ~16 GB | ✓ |
| dpo + openai judge | attacker(+LoRA) + target | ~10 GB | ✓ |
| dpo + hf_local judge | attacker(+LoRA) + target + judge | ~16 GB | ✓ |
| ppo + openai judge | policy + ref + target | ~16 GB | ✓ |
| ppo + hf_local judge | policy + ref + target + judge | ~22 GB + KV cache | ⚠ tight; OOM likely |

For PPO with an HF judge, use an 80 GB A100 or stick with `openai`.

### Switching mid-project: validate first

Switching judge backend changes the evaluation contract. Before publishing
numbers from a new judge, validate agreement with the old one:

```bash
# Re-judge a sample of existing trajectories with the HF judge
python -c "
import json
from src.judge import make_judge, rejudge_traces
hf_judge = make_judge('hf_local', 'meta-llama/Llama-3.1-8B-Instruct')
# load N existing traces, re-score, compute Cohen's kappa on attack_success
# and Pearson r on policy_violation against the GPT-4o-mini judgments
"
```

If Cohen's κ on `attack_success` < 0.6 between the two judges, do not compare
ASR numbers across them — keep `openai` as primary.

## Reproducibility contract

- `--config` YAML is the source of truth. CLI flags only override.
- `setup_seed.seed_everything(seed)` seeds Python / NumPy / torch (CPU+CUDA)
  and sets `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- Each run captures `git_sha`, `git_dirty` flag, full `git diff` (if dirty),
  Python version, `pip freeze`, GPU model, CUDA version, hostname, OS.
- The CLI **refuses to run on a dirty git tree** unless `--allow-dirty`.
- The resolved config is hashed (SHA256, key-sorted JSON) so two runs with
  identical configs share a `config_hash`.

## How to add a new method

1. Add `configs/<name>.yaml` with `method: <name>` and the tunable knobs.
2. Add `src/methods/<name>.py` exposing `run(config: dict, logger: ResultsLogger) -> dict`.
   Inside, load models / data, run training/eval, call `logger.log_jsonl(...)`
   for streaming logs, and return a final-metrics dict.
3. Wire it into `src/cli.py`:
   - Add a `sub.add_parser("<name>")` block in `build_parser`.
   - Add a dispatch arm in `main()`.
   - (Optional) extend `_typed_flag_overrides` for any method-specific typed flags.
4. Document the reproduction command in this README.

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
