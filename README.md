# Multilingual Toxic Comment Detection (CIS5300)
**Translated vs Original-Language Training with Detoxify (XLM-R)**

We reproduce a strong multilingual baseline for toxic comment detection (**Detoxify**) and run two extensions:

1) **Translated regime** — train/fine-tune on machine-translated (EN→XX) variants of the English corpus.  
2) **Original-language regime** — train/fine-tune directly on non-English labeled data from **Jigsaw 2020 (multilingual)**.

We evaluate **per language** (ROC-AUC primary, F1 secondary), provide a **simple baseline** (TF-IDF + Logistic Regression), a **unified scoring script**, and an **error analysis**.

> ⚠️ You must accept Kaggle terms to download Jigsaw datasets. This repo does **not** redistribute any Kaggle data.

---

## TL;DR

- **Data**: Jigsaw 2018 (EN), 2019 (EN + identities), 2020 (multilingual)  
- **Strong baseline**: [`unitaryai/detoxify`](https://github.com/unitaryai/detoxify) (XLM-R)  
- **Compare**: Translated vs Original-language training regimes  
- **Outputs**: `score.py`, per-language predictions, tables/figures for report

---

## Repository Structure

```text
toxicity-xlingual-detoxify/
├── code/
│   ├── simple_baseline.py          # TF-IDF + Logistic Regression (per language)
│   ├── score.py                    # Unified evaluation (ROC-AUC, F1)
│   ├── scripts/
│   │   ├── make_multilingual_splits.py   # 80/10/10 splits from Jigsaw 2020 validation per language
│   │   └── prepare_translated_index.py   # Manifest for translated training files
├── configs/
│   ├── detoxify_multilingual.yaml  # Base config copied from Detoxify (edit paths)
│   ├── translated.yaml             # Overrides for translated-regime runs
│   └── original.yaml               # Overrides for original-language runs
├── data/
│   ├── raw/                        # Kaggle CSVs (ignored by git)
│   └── processed/                  # Language-specific train/dev/test splits
├── external/
│   └── detoxify/                   # Detoxify as a submodule/clone
├── output/
│   ├── runs/                       # Checkpoints, logs
│   └── predictions/                # Per-language prediction files
├── docs/
│   ├── data.md                     # Data schema, sources, sizes, examples
│   └── scoring.md                  # Metric definitions + CLI examples
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 0) Clone and (optionally) add Detoxify as a submodule
```bash
git clone https://github.com/<YOUR-ORG>/toxicity-xlingual-detoxify
cd toxicity-xlingual-detoxify

# Optional but recommended: keep Detoxify pinned as a submodule
git submodule add https://github.com/unitaryai/detoxify external/detoxify
```

### 1) Environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create `requirements.txt` with:
```txt
torch>=2.1
transformers>=4.42
datasets>=2.19
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
pytorch-lightning>=2.3
tqdm>=4.66
textattack>=0.3.8   # optional (robustness later)
```

### 2) Download datasets from Kaggle
Create `~/.kaggle/kaggle.json` (Kaggle API token). Then:
```bash
# Jigsaw 2018 (English)
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
# Jigsaw 2019 (English + identities)
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
# Jigsaw 2020 (Multilingual)
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification

unzip -o '*.zip' -d data/raw/
```

---

## Data Preparation

### A) Original-language splits (from Jigsaw 2020 validation)
Jigsaw 2020 provides labeled non-English validation data. Create 80/10/10 splits **per language** (e.g., `es`, `it`, `tr`):
```bash
python code/scripts/make_multilingual_splits.py   --in data/raw/validation.csv   --langs es it tr   --outdir data/processed/
```

This emits:
```text
data/processed/es/train.csv  dev.csv  test.csv
data/processed/it/train.csv  dev.csv  test.csv
data/processed/tr/train.csv  dev.csv  test.csv
```

### B) Translated-regime manifest (if you maintain EN→XX corpora)
```bash
python code/scripts/prepare_translated_index.py   --translated_root /path/to/translated_corpora/   --out data/processed/translated_manifest.json
```
This JSON maps:
```json
{"es": "/path/to/es_train.csv", "it": "/path/to/it_train.csv", "tr": "/path/to/tr_train.csv"}
```

---

## Simple Baseline (TF-IDF + Logistic Regression)

Train/eval per language:
```bash
python code/simple_baseline.py   --train data/processed/es/train.csv   --dev   data/processed/es/dev.csv   --test  data/processed/es/test.csv   --text_col comment_text --label_col toxicity   --out   output/predictions/es_simple.csv
```

Score:
```bash
python code/score.py   --pred output/predictions/es_simple.csv   --gold data/processed/es/test.csv   --text_col comment_text --label_col toxicity   --average macro
```

---

## Strong Baseline (Reproduction): Detoxify XLM-R

### Option 1: Use Detoxify directly
```bash
cd external/detoxify
pip install -r requirements.txt

# Train (example config; edit to point to your CSVs)
python scripts/train.py --config configs/multilingual.yaml

# Predict on a language-specific test split
python scripts/predict.py   --model multilingual   --test_csv ../../data/processed/es/test.csv   --text_col comment_text   --out ../../output/predictions/es_detoxify.csv
```

### Option 2: Run Detoxify from this repo (using your configs)
```bash
# Translated-regime
python external/detoxify/scripts/train.py --config configs/translated.yaml

# Original-language regime
python external/detoxify/scripts/train.py --config configs/original.yaml
```

Then score with our unified script:
```bash
python code/score.py   --pred output/predictions/es_detoxify.csv   --gold data/processed/es/test.csv   --text_col comment_text --label_col toxicity
```

---

## Extensions We Compare

### 1) Translated regime
- Train/fine-tune with EN→(ES/IT/TR/…) machine-translated corpora.  
- Evaluate per language on **original** (non-translated) held-out test sets.

### 2) Original-language regime
- Train/fine-tune directly on in-language labeled splits from Jigsaw 2020 validation.  
- Same evaluation protocol.

We report per-language **ROC-AUC** and **F1**, plus macro averages; include a brief error analysis (e.g., profanity-free toxicity, identity mentions, obfuscation).

---

## Scoring (Unified)
```bash
python code/score.py   --pred output/predictions/LANG_MODEL.csv   --gold data/processed/LANG/test.csv   --text_col comment_text --label_col toxicity   --average macro
```
**Primary:** ROC-AUC (macro)  
**Secondary:** F1 (macro)  
See `docs/scoring.md` for formulas and CLI examples.

---

## Results Template

| Language | Regime         | ROC-AUC |   F1 |
|---------:|----------------|--------:|-----:|
| ES       | Translated     |  0.____ | .___ |
| ES       | Original       |  0.____ | .___ |
| IT       | Translated     |  0.____ | .___ |
| IT       | Original       |  0.____ | .___ |
| TR       | Translated     |  0.____ | .___ |
| TR       | Original       |  0.____ | .___ |
| **Macro**| **—**          |   **—** | **—** |

---

## Reproducibility

- Package versions pinned in `requirements.txt`  
- Random seeds set in training/eval (where applicable)  
- All commands used to produce reported scores will be mirrored in `docs/README-commands.md`  
- Predictions + gold files stored in `output/predictions/` for grading

---

## Ethics & Safety
Toxicity datasets contain offensive content. Handle with care, follow Kaggle’s ToS, and avoid redistributing raw text.

---

## Team
- Zixuan Bian, Aria Shi, Siyuan Shen, Alex Yang

---

## References / Useful Links
- Detoxify (models + scripts): https://github.com/unitaryai/detoxify  
- Jigsaw 2018 (EN): https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge  
- Jigsaw 2019 (bias): https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification  
- Jigsaw 2020 (multilingual): https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification
