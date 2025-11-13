# Simple Baseline Model - Complete Package

## 📦 Deliverables

### 1. Main Script: `simple_baseline.py`

**Location**: `code/simple_baseline.py`

Complete implementation of TF-IDF + Logistic Regression baseline with:
- ✅ Text preprocessing (lowercase, URL/mention removal, optional stopwords)
- ✅ TF-IDF vectorization (200k features, bigrams, min_df=3)
- ✅ Logistic Regression (max_iter=2000, balanced class weights)
- ✅ Threshold optimization (maximize F1 on validation set)
- ✅ Test predictions with full metrics
- ✅ Model artifact saving (preprocessor, vectorizer, model, threshold)

**Key Features**:
```python
class TextPreprocessor:
    # Handles lowercase, URL removal, mention removal, stopwords
    
TfidfVectorizer(
    max_features=200000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

def find_best_threshold():
    # Tests 81 thresholds (0.1 to 0.9)
    # Maximizes F1 score
```

### 2. Usage Documentation: `simple_baseline_usage.md`

Comprehensive guide including:
- Quick start commands
- Input/output formats
- Performance tips
- Troubleshooting
- Example workflows
- Inference examples

### 3. Verification Script: `verify_predictions.py`

**Location**: `code/verify_predictions.py`

Quick validation script that checks:
- File existence
- Column presence (id, y_true, y_prob, y_pred, best_threshold)
- Data validity (binary labels, probabilities in [0,1])
- Basic statistics (accuracy, toxic rates)

**Usage**:
```bash
python code/verify_predictions.py
```

## 🚀 Quick Start Guide

### Step 1: Prepare Environment

```bash
# Install dependencies
pip install pandas numpy scikit-learn pyyaml joblib

# Optional: for stopwords
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

### Step 2: Prepare Data

```bash
# Run data splitting (if not done)
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml
```

Expected output structure:
```
data/processed/
├── en/
│   ├── train.csv (70%)
│   ├── val.csv (10%)
│   ├── test.csv (10%)
│   └── heldout.csv (10%)
├── es/
...
```

### Step 3: Train Baseline Models

```bash
# Train all languages
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml

# Or train specific languages
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es fr
```

### Step 4: Verify Results

```bash
# Quick verification
python code/verify_predictions.py

# View predictions
head output/predictions/baseline_en.csv

# View summary
cat output/runs/baseline/baseline_results_summary.csv
```

## 📊 Expected Output Structure

```
output/
├── runs/
│   └── baseline/
│       ├── en/
│       │   ├── preprocessor.pkl
│       │   ├── vectorizer.pkl
│       │   ├── model.pkl
│       │   └── threshold.pkl
│       ├── es/
│       │   └── ...
│       ...
│       └── baseline_results_summary.csv
└── predictions/
    ├── baseline_en.csv
    ├── baseline_es.csv
    ...
```

## 📈 Expected Performance

Based on dataset characteristics:

### English (9% toxic, ~157k train samples)
- **ROC-AUC**: 0.90 - 0.93
- **F1-Score**: 0.75 - 0.80
- **Precision**: 0.70 - 0.75
- **Recall**: 0.80 - 0.85

### Other Languages (50% toxic, ~700 train samples)
- **ROC-AUC**: 0.85 - 0.90
- **F1-Score**: 0.80 - 0.85
- **Precision**: 0.78 - 0.83
- **Recall**: 0.82 - 0.87

**Note**: These are baseline models. Transformer-based models (BERT, XLM-RoBERTa) typically achieve 2-5% higher ROC-AUC.

## 🔍 Output File Formats

### 1. Predictions CSV (`baseline_{lang}.csv`)

```csv
id,y_true,y_prob,y_pred,best_threshold
0,0,0.1234,0,0.456
1,1,0.7890,1,0.456
2,0,0.3456,0,0.456
```

### 2. Results Summary CSV

```csv
language,val_roc_auc,val_f1,val_precision,val_recall,test_roc_auc,test_f1,test_precision,test_recall,best_threshold
en,0.9234,0.7856,0.7234,0.8567,0.9145,0.7723,0.7089,0.8456,0.456
es,0.8876,0.8012,0.7945,0.8123,0.8723,0.7934,0.7812,0.8089,0.523
```

## 🔧 Customization Options

### Adjust TF-IDF Parameters

Edit in `simple_baseline.py`:
```python
vectorizer = TfidfVectorizer(
    max_features=200000,     # Reduce for speed (e.g., 50000)
    ngram_range=(1, 2),      # Change to (1, 1) for unigrams only
    min_df=3,                # Increase to reduce vocabulary
    max_df=0.95              # Adjust for common word filtering
)
```

### Adjust Model Parameters

```python
model = LogisticRegression(
    max_iter=2000,           # Reduce for speed (e.g., 1000)
    class_weight='balanced', # Or None, or dict for custom weights
    C=1.0                    # Add for regularization strength
)
```

### Use Stopwords

```bash
python code/simple_baseline.py \
    --config configs/detoxify_multilingual.yaml \
    --use_stopwords
```

## 🐛 Common Issues & Solutions

### Issue 1: "FileNotFoundError: Training data not found"

**Cause**: Data splits haven't been created

**Solution**:
```bash
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml
```

### Issue 2: Memory error during training

**Cause**: Too many features or large dataset

**Solutions**:
1. Reduce `max_features` to 50000 or 100000
2. Train one language at a time
3. Use a machine with more RAM

### Issue 3: Low performance on non-English languages

**Cause**: Small training set (~700 samples per language)

**Solutions**:
1. Use translation approach (translate English data)
2. Use multilingual BERT models for transfer learning
3. Combine with external datasets

### Issue 4: Threshold optimization taking too long

**Cause**: Large validation set or many samples

**Solutions**:
1. Reduce number of thresholds tested (edit `np.linspace(0.1, 0.9, 81)`)
2. Use a subset of validation data for threshold search
3. Use default threshold (0.5) without optimization

## 📝 Prediction File Usage

### Load and Analyze Predictions

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# Load predictions
pred_df = pd.read_csv('output/predictions/baseline_en.csv')

# Calculate metrics
roc_auc = roc_auc_score(pred_df['y_true'], pred_df['y_prob'])
f1 = f1_score(pred_df['y_true'], pred_df['y_pred'])

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"F1-Score: {f1:.4f}")

# Analyze predictions by probability
pred_df['prob_bucket'] = pd.cut(pred_df['y_prob'], bins=10)
bucket_stats = pred_df.groupby('prob_bucket')['y_true'].agg(['mean', 'count'])
print(bucket_stats)

# Find misclassified examples
false_positives = pred_df[(pred_df['y_true'] == 0) & (pred_df['y_pred'] == 1)]
false_negatives = pred_df[(pred_df['y_true'] == 1) & (pred_df['y_pred'] == 0)]

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")
```

### Use Saved Model for Inference

```python
import joblib

# Load model artifacts
model_dir = "output/runs/baseline/en"
preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
vectorizer = joblib.load(f"{model_dir}/vectorizer.pkl")
model = joblib.load(f"{model_dir}/model.pkl")
threshold_data = joblib.load(f"{model_dir}/threshold.pkl")

# Predict on new text
def predict_toxicity(text):
    # Preprocess
    processed = preprocessor.preprocess(text)
    
    # Vectorize
    features = vectorizer.transform([processed])
    
    # Predict probability
    prob = model.predict_proba(features)[0, 1]
    
    # Apply threshold
    pred = 1 if prob >= threshold_data['best_threshold'] else 0
    
    return {
        'text': text,
        'probability': prob,
        'prediction': 'TOXIC' if pred else 'NOT TOXIC',
        'threshold': threshold_data['best_threshold']
    }

# Example usage
result = predict_toxicity("This is a test comment")
print(result)
```

## 🎯 Next Steps

After training baseline models:

### 1. Evaluate Performance
```bash
python code/score.py \
    --config configs/detoxify_multilingual.yaml \
    --pred_dir output/predictions
```

### 2. Error Analysis
- Analyze false positives and false negatives
- Identify common patterns in misclassifications
- Check if certain topics or writing styles are problematic

### 3. Feature Engineering
- Add custom features (text length, caps ratio, punctuation)
- Try different n-gram ranges
- Experiment with character-level features

### 4. Advanced Models
- Train BERT-based models for better context understanding
- Try XLM-RoBERTa for multilingual transfer learning
- Ensemble baseline with neural models

### 5. Translation Approach
- Translate English data to other languages
- Train on larger datasets
- Compare with original language approach

## 📚 References

- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Handling Imbalanced Data](https://imbalanced-learn.org/)
- [Text Classification Guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## ✅ Checklist

Before running the baseline:
- [ ] Config file exists and is correctly formatted
- [ ] Data splits have been created (`data/processed/{lang}/*.csv`)
- [ ] Required packages are installed
- [ ] Output directories have write permissions

After running the baseline:
- [ ] Check log output for errors
- [ ] Verify all languages were processed
- [ ] Verify predictions files exist
- [ ] Run verification script
- [ ] Review results summary
- [ ] Save model artifacts for future use

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section in `simple_baseline_usage.md`
2. Verify your data format and structure
3. Review error messages carefully
4. Check that all dependencies are installed
5. Ensure sufficient disk space and memory

---

**Package Version**: 1.0  
**Last Updated**: 2024-01-XX  
**Status**: Production Ready ✅
