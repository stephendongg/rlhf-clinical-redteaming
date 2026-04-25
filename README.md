# RLHF for Automated Clinical Red-Teaming

**Audrey Tjokro · Stephen Dong · Niki Karanikola**  
Cornell University — CS 6782 Generative Models, Spring 2026

Multi-turn adversarial stress-testing for clinical LLMs, comparing PPO and DPO for training red-team agents on Med-Safety-Bench.

---

## Setup

**Requires a Colab A100 GPU.**

1. Clone the repo into Google Drive (run once):
```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/stephendongg/rlhf-clinical-redteaming /content/drive/MyDrive/rlhf-clinical-redteaming
```

2. Add your OpenAI key to Colab Secrets (🔑 icon, left sidebar): `OPENAI_API_KEY`

3. Open `redteam.ipynb` from Drive in Colab and run all cells.

Each subsequent session: open the notebook from Drive and run `!git pull` to get the latest code.

---

## Dataset

**Med-Safety-Bench** (Han et al., NeurIPS 2024) — 1,800 harmful medical requests paired with safe responses.

- Loaded via HuggingFace: `israel-adewuyi/med-safety-bench-reproduced`
- Train split (900): used for PPO/DPO fine-tuning
- Test split (900): held out for final evaluation

---

## Models

| Role | Model |
|---|---|
| Attacker | `Qwen/Qwen2.5-7B-Instruct` |
| Target | `BioMistral/BioMistral-7B` |
| Judge | `GPT-4o` via OpenAI API |
