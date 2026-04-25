# RLHF for Automated Clinical Red-Teaming

**Audrey Tjokro · Stephen Dong · Niki Karanikola**  
Cornell University — CS 6782 Generative Models, Spring 2026

Multi-turn adversarial stress-testing for clinical LLMs, comparing PPO and DPO for training red-team agents on Med-Safety-Bench.

---

## Local Setup (VS Code)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# then open .env and paste your OPENAI_API_KEY
```

Then open `redteam.ipynb` in VS Code. Click the **kernel picker** (top-right of the notebook) → **Python Environments** → select `.venv`.

---

## Google Colab Setup

Clone the repo into Google Drive once, then open `redteam.ipynb` from Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/stephendongg/rlhf-clinical-redteam /content/drive/MyDrive/rlhf-clinical-redteam
```

Each subsequent session just run `!git pull` to get the latest code.

---

## Dataset

**Med-Safety-Bench** (Han et al., NeurIPS 2024)  
1,800 harmful medical requests paired with safe responses.

- Loaded via HuggingFace: `israel-adewuyi/med-safety-bench-reproduced`
- Train split (900): used for PPO/DPO fine-tuning
- Test split (900): held out for final evaluation

---

## Models

| Role | Model |
|---|---|
| Target (defender) | `BioMistral-7B` |
| Attacker (adversary) | `Qwen2.5-7B` |
| Judge | `GPT-4o` via OpenAI API |
