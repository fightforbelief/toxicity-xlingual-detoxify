# Strong Baseline (`strong-baseline.md`)

This document describes the **strong baseline model** used in our multilingual toxicity detection project.

The strong baseline is the **Detoxify multilingual model**, which uses an XLM-RoBERTa transformer to produce toxicity scores for comments in multiple languages. We treat this model as a strong, literature-based baseline and integrate it into our evaluation pipeline.

---

## 1. Model Overview

### 1.1 Detoxify Multilingual XLM-R

- We use the **`Detoxify("multilingual")`** model from the open-source Detoxify library.
- Internally, this model is based on **XLM-RoBERTa**, a multilingual transformer pretrained on many languages.
- The model outputs a set of toxicity-related scores. We use the **`toxicity`** score as our prediction for the binary toxicity label.

Formally, for an input comment \(x\), Detoxify produces a real-valued score:

\[
\hat{p} = f_{\text{Detoxify}}(x) \in [0, 1],
\]

which we interpret as the probability that \(x\) is toxic.

### 1.2 Inference-only Strong Baseline

For Milestone 2, we use Detoxify **in inference mode** on our processed splits:

- We do **not** fine-tune the model; rather, we evaluate how well the pretrained multilingual model transfers to our data.
- We apply Detoxify independently to each language’s validation and test split.

---

## 2. Code Location

- Main script: `code/strong_baseline_detoxify.py`
- Data splits: `data/processed/{lang}/val.csv`, `test.csv`
- Outputs:
  - Per-language predictions:  
    `output/predictions/detoxify_{lang}.csv`
  - Summary metrics:  
    `output/predictions/detoxify_metrics.csv`

The script assumes that `detoxify` and its dependencies are installed:

```bash
pip install detoxify torch transformers
````

---

## 3. Command-line Usage

To run the Detoxify strong baseline on all four languages:

```bash
python code/strong_baseline_detoxify.py \
  --config configs/detoxify_multilingual.yaml \
  --languages en es it tr \
  --run_tag detoxify \
  --batch_size 32
```

This will:

1. Build the Detoxify multilingual model.
2. For each language (\ell):

   * Load `val.csv` and `test.csv` from `data/processed/{lang}`.
   * Run Detoxify on the validation texts to obtain toxicity probabilities.
   * Tune a threshold (\tau_\ell^\star) on the validation set by maximizing F1.
   * Run Detoxify on the test texts.
   * Save a prediction file `output/predictions/detoxify_{lang}.csv`.

Important arguments:

* `--text_col` (default: `comment_text`): the name of the text column.
* `--label_col` (default: `toxic`): the name of the binary label column.
* `--device`: optional device string (`cuda` or `cpu`). If omitted, Detoxify chooses automatically.

Example with explicit device:

```bash
python code/strong_baseline_detoxify.py \
  --config configs/detoxify_multilingual.yaml \
  --languages en es it tr \
  --run_tag detoxify \
  --batch_size 32 \
  --device cpu
```

---

## 4. Output Format

Each per-language prediction file has the form:

```csv
id,y_true,y_prob,y_pred,best_threshold
0,0,0.0213,0,0.7300
1,1,0.9458,1,0.7300
2,0,0.1120,0,0.7300
...
```

* `id`: unique row identifier
* `y_true`: gold binary label (0 = non-toxic, 1 = toxic)
* `y_prob`: Detoxify `toxicity` probability
* `y_pred`: binarized prediction using `best_threshold`
* `best_threshold`: threshold tuned on the validation set for this language

The metrics file `detoxify_metrics.csv` is produced by the scoring script:

```bash
python code/score.py \
  --config configs/detoxify_multilingual.yaml \
  --pred_dir output/predictions \
  --run_tag detoxify \
  --languages en es it tr \
  --output output/predictions/detoxify_metrics.csv
```

---

## 5. Performance Summary

The table below summarizes Detoxify’s performance on the **test set** for all languages:

| language | ROC-AUC |     F1 | Precision | Recall | Accuracy |
| -------- | ------: | -----: | --------: | -----: | -------: |
| en       |  0.9889 | 0.8238 |    0.8244 | 0.8232 |   0.9663 |
| es       |  0.9184 | 0.6154 |    0.8696 | 0.4762 |   0.9004 |
| it       |  0.8756 | 0.5584 |    0.4095 | 0.8776 |   0.7291 |
| tr       |  0.9824 | 0.8219 |    0.7317 | 0.9375 |   0.9567 |
| macro    |  0.9413 | 0.7049 |    0.7088 | 0.7786 |   0.8881 |

Key observations:

* Compared to the TF–IDF baseline, Detoxify significantly improves **ROC-AUC** and **F1** on English and Turkish.
* On Spanish, Detoxify achieves a higher ROC-AUC and much higher precision, but with lower recall due to a conservative threshold.
* On Italian, Detoxify trades off precision for very high recall, indicating that it is more “aggressive” in flagging toxicity for this language.
* Overall, macro ROC-AUC increases from **0.907** to **0.941**, and macro F1 increases from **0.650** to **0.705**, showing clear gains from the transformer-based model.