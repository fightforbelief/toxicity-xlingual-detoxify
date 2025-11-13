# Simple Baseline Model - Usage Guide

## Overview

The `simple_baseline.py` script trains TF-IDF + Logistic Regression baseline models for multilingual toxicity detection. Each language gets its own independent model.

## Features

✅ **Text Preprocessing**
- Lowercase conversion
- URL removal (`http://...`, `www...`)
- Mention removal (`@username`)
- Optional stopwords removal (requires NLTK)

✅ **TF-IDF Vectorization**
- Max 200,000 features
- Unigrams + bigrams (1-2 ngrams)
- Min document frequency: 3
- Max document frequency: 95%
- Sublinear TF scaling

✅ **Logistic Regression**
- Max 2000 iterations
- Balanced class weights (handles imbalanced data)
- L2 regularization
- Multi-threaded training

✅ **Threshold Optimization**
- Tests 81 thresholds from 0.1 to 0.9
- Maximizes F1 score on validation set
- Saves optimal threshold per language

✅ **Comprehensive Output**
- Test predictions with probabilities
- Performance metrics (ROC-AUC, F1, Precision, Recall, Accuracy)
- Saved model artifacts (preprocessor, vectorizer, model, threshold)

## Quick Start

### Basic Usage

```bash
# Train baseline models for all languages
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml

# Train for specific languages only
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es fr

# Use stopwords removal (requires NLTK)
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --use_stopwords
```

### Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn pyyaml joblib

# Optional: for stopwords removal
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | *required* | Path to config YAML file |
| `--languages` | list | `[en, es, fr, de, it, pt, ru, tr]` | Languages to train |
| `--use_stopwords` | flag | False | Enable stopwords removal |

## Input Requirements

### Data Structure

The script expects processed data in this structure:

```
data/processed/
├── en/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── es/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
...
```

### CSV Format

Each CSV file must have these columns:

```csv
comment_text,toxic,lang
"This is a comment",0,en
"Another example",1,en
```

- `comment_text`: Text content (string)
- `toxic`: Binary label (0 = non-toxic, 1 = toxic)
- `lang`: Language code (en, es, etc.)

## Output Files

### 1. Model Artifacts

Saved to `output/runs/baseline/{language}/`:

```
output/runs/baseline/
├── en/
│   ├── preprocessor.pkl    # Text preprocessor
│   ├── vectorizer.pkl      # TF-IDF vectorizer
│   ├── model.pkl           # Logistic Regression model
│   └── threshold.pkl       # Optimal threshold
├── es/
│   ├── preprocessor.pkl
│   ├── vectorizer.pkl
│   ├── model.pkl
│   └── threshold.pkl
...
└── baseline_results_summary.csv
```

### 2. Test Predictions

Saved to `output/predictions/baseline_{language}.csv`:

```csv
id,y_true,y_prob,y_pred,best_threshold
0,0,0.123,0,0.456
1,1,0.789,1,0.456
2,0,0.234,0,0.456
...
```

Columns:
- `id`: Sample index
- `y_true`: True label (0 or 1)
- `y_prob`: Predicted probability of toxicity (0.0 to 1.0)
- `y_pred`: Binary prediction using best threshold (0 or 1)
- `best_threshold`: Optimal threshold found on validation set

### 3. Results Summary

Saved to `output/runs/baseline/baseline_results_summary.csv`:

```csv
language,val_roc_auc,val_f1,val_precision,val_recall,test_roc_auc,test_f1,test_precision,test_recall,best_threshold
en,0.9234,0.7856,0.7234,0.8567,0.9145,0.7723,0.7089,0.8456,0.456
es,0.8876,0.8012,0.7945,0.8123,0.8723,0.7934,0.7812,0.8089,0.523
...
```

## Example Workflow

### Step 1: Prepare Data

```bash
# Create data splits (if not done already)
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml
```

### Step 2: Train Baseline Models

```bash
# Train all languages
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml
```

Expected output:
```
Loading config from configs/detoxify_multilingual.yaml

Configuration:
  Data directory: data/processed
  Model output: output/runs/baseline
  Predictions output: output/predictions
  Languages: en, es, fr, de, it, pt, ru, tr
  Use stopwords: False

======================================================================
PROCESSING LANGUAGE: EN
======================================================================
Loaded en data:
  Train: 157184 samples (9.1% toxic)
  Val:   22455 samples (9.1% toxic)
  Test:  22455 samples (9.1% toxic)

Training model for en...
  Preprocessing texts...
  Vectorizing with TF-IDF...
  Feature matrix shape: (157184, 200000)
  Training logistic regression...
  Training complete!

Evaluating on validation set...
  Best threshold: 0.456 (F1: 0.7856)

EN Validation Results:
  ROC-AUC:    0.9234
  F1-Score:   0.7856
  Precision:  0.7234
  Recall:     0.8567
  Accuracy:   0.9567
  Threshold:  0.456

Generating predictions on test set...

EN Test Results:
  ROC-AUC:    0.9145
  F1-Score:   0.7723
  Precision:  0.7089
  Recall:     0.8456
  Accuracy:   0.9534
  Predictions saved to: output/predictions/baseline_en.csv

Model artifacts saved to: output/runs/baseline/en
...
```

### Step 3: Analyze Results

```bash
# View summary
cat output/runs/baseline/baseline_results_summary.csv

# View predictions for a specific language
head output/predictions/baseline_en.csv
```

## Using Saved Models for Inference

```python
import joblib
import pandas as pd

# Load model artifacts for English
model_dir = "output/runs/baseline/en"
preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
model = joblib.load(f"{model_dir}/model.pkl")
threshold_data = joblib.load(f"{model_dir}/threshold.pkl")
best_threshold = threshold_data['best_threshold']

# Predict on new text
new_text = "This is a test comment"
processed_text = preprocessor.preprocess(new_text)
text_tfidf = vectorizer.transform([processed_text])
prob = model.predict_proba(text_tfidf)[0, 1]
pred = 1 if prob >= best_threshold else 0

print(f"Text: {new_text}")
print(f"Toxic probability: {prob:.4f}")
print(f"Prediction: {'TOXIC' if pred else 'NOT TOXIC'}")
print(f"Threshold: {best_threshold:.3f}")
```

## Performance Tips

### 1. Memory Usage

For large datasets (especially English with ~157k samples):

```bash
# Monitor memory during training
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en  # Train one language at a time
```

### 2. Speed Optimization

- **Use fewer features**: Edit `max_features=200000` to `max_features=50000`
- **Disable bigrams**: Change `ngram_range=(1,2)` to `ngram_range=(1,1)`
- **Reduce max_iter**: Change `max_iter=2000` to `max_iter=1000`

### 3. Handling Imbalanced Data

The script already uses `class_weight='balanced'`, but you can also:

1. Use different thresholds for precision vs. recall
2. Apply SMOTE or oversampling (requires modification)
3. Use focal loss (requires custom implementation)

## Troubleshooting

### Issue: "FileNotFoundError: Training data not found"

**Solution**: Run the data splitting script first:
```bash
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml
```

### Issue: "Memory Error" during training

**Solutions**:
1. Reduce `max_features` in the script
2. Process languages one at a time
3. Use a machine with more RAM
4. Enable swap space (Linux)

### Issue: Low F1 scores for some languages

**Possible causes**:
1. Small dataset size (non-English languages have ~1000 samples)
2. Class imbalance (English has 9% toxic vs. 50% for others)
3. Language-specific challenges (idioms, cultural context)

**Solutions**:
1. Use more training data (translation approach)
2. Try multilingual models (BERT, XLM-RoBERTa)
3. Ensemble with other models

### Issue: "KeyError: 'path'" in config

**Solution**: Update your config file to use `raw_dir` and `processed_dir` instead of `path`:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
```

## Limitations

1. **No Cross-lingual Transfer**: Each language model is trained independently
2. **Bag-of-Words**: TF-IDF ignores word order and context
3. **Limited Context**: Bigrams only capture 2-word phrases
4. **No Semantic Understanding**: Cannot understand synonyms or paraphrases
5. **Cold Start**: Poor performance on unseen vocabulary

For better performance, consider:
- BERT-based models (better context understanding)
- XLM-RoBERTa (cross-lingual transfer)
- Ensemble methods (combine multiple models)

## Next Steps

After training baseline models:

1. **Evaluate Performance**: Use `code/score.py` for comprehensive evaluation
2. **Compare Approaches**: Train translation-based models and compare
3. **Error Analysis**: Examine misclassified examples
4. **Feature Engineering**: Add custom features (length, caps ratio, etc.)
5. **Advanced Models**: Train transformer-based models (BERT, RoBERTa)

## References

- **TF-IDF**: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **Logistic Regression**: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **Imbalanced Learning**: [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
