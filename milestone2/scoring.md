# Scoring Documentation (`scoring.md`)

This document describes the evaluation metrics and scoring pipeline for the **multilingual toxicity detection** project.

We evaluate **binary toxicity classification** (toxic vs. non-toxic) for multiple languages.  
Our **primary metric** is ROC-AUC and our **secondary metric** is F1.  
All metrics are reported **per language**, plus **macro** and **micro** averages across languages.

---

## 1. Notation

For a given language and test set, let:

- \(y_i \in \{0,1\}\) be the gold label for example \(i\)  
- \(\hat{p}_i \in [0,1]\) be the model’s predicted toxicity probability  
- \(\hat{y}_i \in \{0,1\}\) be the binarized prediction after applying a threshold \(\tau\)

We define the usual confusion–matrix counts:

- \(TP = \#\{i : y_i = 1, \hat{y}_i = 1\}\)  
- \(FP = \#\{i : y_i = 0, \hat{y}_i = 1\}\)  
- \(TN = \#\{i : y_i = 0, \hat{y}_i = 0\}\)  
- \(FN = \#\{i : y_i = 1, \hat{y}_i = 0\}\)

The total number of samples is \(N = TP + FP + TN + FN\).

---

## 2. Core Metrics

### 2.1 ROC-AUC (primary metric)

For each language, we compute the **Receiver Operating Characteristic – Area Under Curve (ROC-AUC)**.

- The ROC curve plots the **True Positive Rate** against the **False Positive Rate** as the threshold \(\tau\) varies:
  \[
  \mathrm{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}, \quad
  \mathrm{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}.
  \]
- ROC-AUC is the area under this curve.  
  Intuitively, ROC-AUC equals the probability that a randomly chosen toxic comment receives a higher score than a randomly chosen non-toxic comment.

We use the standard implementation in `sklearn.metrics.roc_auc_score`.

---

### 2.2 Precision, Recall, and F1 (secondary metrics)

Given a fixed decision threshold \(\tau\) and binarized predictions \(\hat{y}_i = \mathbb{I}[\hat{p}_i \ge \tau]\):

- **Precision**
  \[
  \mathrm{Precision} = \frac{TP}{TP + FP}
  \]
- **Recall**
  \[
  \mathrm{Recall} = \frac{TP}{TP + FN}
  \]
- **F1 score** (harmonic mean of precision and recall)
  \[
  \mathrm{F1} = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
  \]

F1 is our **secondary metric** for comparing baselines.

---

### 2.3 Accuracy and Additional Diagnostics

For completeness we also report:

- **Accuracy**
  \[
  \mathrm{Accuracy} = \frac{TP + TN}{N}
  \]
- **Specificity** (True Negative Rate)
  \[
  \mathrm{Specificity} = \frac{TN}{TN + FP}
  \]
- **False Positive Rate**
  \[
  \mathrm{FPR} = \frac{FP}{FP + TN}
  \]
- **False Negative Rate**
  \[
  \mathrm{FNR} = \frac{FN}{FN + TP}
  \]

These are computed using `sklearn.metrics.confusion_matrix` and are mainly used for error analysis.

---

## 3. Macro vs. Micro Averaging Across Languages

Let \(\mathcal{L}\) be the set of languages (e.g., `en`, `es`, `it`, `tr`).

### Macro Average

For any metric \(m\) (e.g., ROC-AUC, F1), we compute the **macro average** as the simple mean of per-language values:
\[
m_{\text{macro}} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} m_\ell.
\]

This treats all languages equally, regardless of their dataset size.

### Micro Average

For **micro averaging**, we concatenate all predictions across languages and then recompute the metrics:

- Concatenate \(\{y_i^{(\ell)}\}\) and \(\{\hat{p}_i^{(\ell)}\}\) over all languages \(\ell\).
- Compute ROC-AUC and F1 **once** on this combined set.

This gives a size-weighted view of performance.

---

## 4. Threshold Selection

Model training scripts (e.g., `simple_baseline.py` and `strong_baseline_detoxify.py`) select a per-language threshold \(\tau_\ell\) on the **validation set**, by maximizing F1:

\[
\tau_\ell^\star = \arg\max_{\tau \in [0, 1]} \mathrm{F1}_\ell(\tau).
\]

The chosen threshold is stored in the prediction files in a column called `best_threshold`.  
The scoring script (`score.py`) will:

1. Load `y_true` and `y_prob` from the prediction CSV.
2. If a binary prediction column is not present, it will:
   - Read `best_threshold` (if available) and use it as \(\tau\); otherwise default to \(\tau = 0.5\).
   - Compute \(\hat{y}_i = \mathbb{I}[\hat{p}_i \ge \tau]\).

---

## 5. Using the Evaluation Script (`score.py`)

### 5.1 Command-line Arguments

The main arguments are:

- `--config`: YAML config file (e.g., `configs/detoxify_multilingual.yaml`)
- `--pred_dir`: directory containing prediction CSVs (e.g., `output/predictions`)
- `--run_tag`: prefix of prediction files (e.g., `baseline` or `detoxify`)
  - The script will look for files like `{run_tag}_en.csv`, `{run_tag}_es.csv`, etc.
- `--languages`: list of languages to evaluate
- `--output`: path to the metrics CSV to write
- `--dump_errors` (optional): whether to dump FP/FN examples per language
- `--errors_dir` (optional): directory for error CSVs (defaults to `pred_dir`)

---

### 5.2 Example: Evaluating the TF–IDF + Logistic Regression Baseline

```bash
python code/score.py \
  --config configs/detoxify_multilingual.yaml \
  --pred_dir output/predictions \
  --run_tag baseline \
  --languages en es it tr \
  --output output/predictions/baseline_metrics.csv
````

This command expects prediction files:

* `output/predictions/baseline_en.csv`
* `output/predictions/baseline_es.csv`
* `output/predictions/baseline_it.csv`
* `output/predictions/baseline_tr.csv`

and will write a summary metrics file:

* `output/predictions/baseline_metrics.csv`

The output CSV contains per-language rows plus macro/micro rows.
Example (values shown for the current baseline):

```text
language,n_samples,positive_rate,roc_auc,f1,precision,recall,accuracy
en,22355,0.0956,0.9686,0.7371,0.7339,0.7404,0.9495
es,251,0.1673,0.8604,0.6067,0.5745,0.6429,0.8606
it,251,0.1952,0.8430,0.5192,0.4909,0.5510,0.8008
tr,300,0.1067,0.9569,0.7368,0.8400,0.6562,0.9500
macro,23157,0.1412,0.9072,0.6500,0.6598,0.6476,0.8902
micro,23157,0.0976,0.9661,0.7296,0.7259,0.7333,0.9469
```

---

### 5.3 Example: Evaluating the Detoxify Strong Baseline

```bash
python code/score.py \
  --config configs/detoxify_multilingual.yaml \
  --pred_dir output/predictions \
  --run_tag detoxify \
  --languages en es it tr \
  --output output/predictions/detoxify_metrics.csv
```

This will read:

* `output/predictions/detoxify_en.csv`
* `output/predictions/detoxify_es.csv`
* `output/predictions/detoxify_it.csv`
* `output/predictions/detoxify_tr.csv`

and produce `output/predictions/detoxify_metrics.csv`, which for the current strong baseline looks like:

```text
language,n_samples,positive_rate,roc_auc,f1,precision,recall,accuracy
en,22355,0.0956,0.9889,0.8238,0.8244,0.8232,0.9663
...
macro,23157,0.1412,0.9413,0.7049,0.7088,0.7786,0.8881
micro,23157,0.0976,0.9873,0.8118,0.8043,0.8195,0.9629
```

---

## 6. References

* Standard definitions of ROC-AUC, precision, recall, and F1 follow the descriptions in common machine learning textbooks and online resources (e.g., the Wikipedia pages on ROC curves and F-scores).
* Implementation uses `sklearn.metrics` from scikit-learn.