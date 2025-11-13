# Simple Baseline (`simple-baseline.md`)

This document describes the **simple baseline model** used in our multilingual toxicity detection project.

The simple baseline is a **per-language TF–IDF + Logistic Regression classifier**.  
It is deliberately lightweight and uses only bag-of-words features, serving as a strong traditional baseline against which we compare modern transformer-based models.

---

## 1. Model Overview

For each language \(\ell \in \{\text{en}, \text{es}, \text{it}, \text{tr}\}\), we train an independent binary classifier:

1. **Text preprocessing**
   - Lowercase conversion
   - Removal of URLs (`http://...`, `https://...`, `www...`)
   - Removal of `@mentions`
   - Optional stopword removal using NLTK (controlled by `--use_stopwords`)

2. **TF–IDF vectorization**
   - Character/word-level: word n-grams (1–2 grams)
   - `max_features = 200000`
   - `min_df = 3`, `max_df = 0.95`
   - Fitted **separately** for each language

3. **Classifier: Logistic Regression**
   - `class_weight = "balanced"` to handle class imbalance
   - `max_iter = 2000`
   - One binary classifier per language

4. **Threshold selection**
   - For each language, we sweep over thresholds on the validation set and pick the one that maximizes F1:
     \[
     \tau_\ell^\star = \arg\max_{\tau} \mathrm{F1}_\ell(\tau).
     \]
   - The chosen threshold is saved in the prediction files as `best_threshold`.

---

## 2. Code Location

- Main script: `code/simple_baseline.py`
- Data splits: `data/processed/{lang}/train.csv`, `val.csv`, `test.csv`
- Outputs:
  - Per-language predictions:  
    `output/predictions/baseline_{lang}.csv`
  - Summary metrics:  
    `output/predictions/baseline_metrics.csv`
  - Saved models and preprocessing artifacts:  
    `output/runs/baseline/{lang}/model.pkl`, `vectorizer.pkl`, `preprocessor.pkl`, `threshold.pkl`

---

## 3. Command-line Usage

### 3.1 Training and Evaluation

To train and evaluate the simple TF–IDF + LR baseline on all four languages:

```bash
python code/simple_baseline.py \
  --config configs/detoxify_multilingual.yaml \
  --languages en es it tr \
  --run_tag baseline
````

This will:

1. Load training, validation, and test splits for each language from `data/processed/{lang}`.
2. Fit a TF–IDF vectorizer and Logistic Regression model per language.
3. Tune a decision threshold on the validation set.
4. Generate prediction files in `output/predictions/baseline_{lang}.csv`.
5. Compute metrics and write `output/predictions/baseline_metrics.csv`.

### 3.2 Optional Arguments

Key optional flags (see `simple_baseline.py` for full details):

* `--use_stopwords`
  Remove language-specific stopwords during preprocessing (requires NLTK stopwords).
* `--max_features`
  Maximum number of TF–IDF features (default: 200000).
* `--max_iter`
  Maximum number of optimization iterations for Logistic Regression.

Example with stopwords:

```bash
python code/simple_baseline.py \
  --config configs/detoxify_multilingual.yaml \
  --languages en es it tr \
  --run_tag baseline \
  --use_stopwords
```

---

## 4. Output Format

Each per-language prediction file has the form:

```csv
id,y_true,y_prob,y_pred,best_threshold
0,0,0.1234,0,0.4560
1,1,0.7890,1,0.4560
2,0,0.3456,0,0.4560
...
```

* `id`: unique row identifier
* `y_true`: gold binary label (0 = non-toxic, 1 = toxic)
* `y_prob`: predicted toxicity probability from the Logistic Regression model
* `y_pred`: binarized prediction using `best_threshold`
* `best_threshold`: per-language threshold tuned on the validation set

The scoring script (`score.py`) reads these files to compute evaluation metrics.

---

## 5. Performance Summary

The table below summarizes the simple baseline’s performance on the **test set** for all languages:

| language | ROC-AUC |     F1 | Precision | Recall | Accuracy |
| -------- | ------: | -----: | --------: | -----: | -------: |
| en       |  0.9686 | 0.7371 |    0.7339 | 0.7404 |   0.9495 |
| es       |  0.8604 | 0.6067 |    0.5745 | 0.6429 |   0.8606 |
| it       |  0.8430 | 0.5192 |    0.4909 | 0.5510 |   0.8008 |
| tr       |  0.9569 | 0.7368 |    0.8400 | 0.6562 |   0.9500 |
| macro    |  0.9072 | 0.6500 |    0.6598 | 0.6476 |   0.8902 |

Key observations:

* The baseline achieves **strong ROC-AUC** on English and Turkish (≈0.96–0.97), with solid F1 and accuracy.
* Performance on Spanish and Italian is lower but still reasonable, reflecting smaller dataset sizes and domain shift.
* Macro-averaged ROC-AUC is **0.91**, and macro F1 is **0.65**, leaving room for improvement with stronger models such as Detoxify.