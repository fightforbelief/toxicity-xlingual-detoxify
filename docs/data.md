# Data Documentation (Updated)

## Overview

This project uses the **Jigsaw 2020 Multilingual Toxic Comment Classification** dataset. **Important**: The original dataset contains only **4 languages** (English, Spanish, Italian, Turkish), not the 8 languages initially planned.

## Data Source

### Jigsaw 2020 Multilingual Toxic Comment Classification

- **Competition URL**: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification
- **License**: CC0: Public Domain
- **Actual Languages Available**: 4 (en, es, it, tr)

## Dataset Structure (Actual)

### What's Actually in the Dataset

```
Jigsaw 2020 Download:
├── jigsaw-toxic-comment-train.csv
│   └── ~224,000 English samples with toxicity labels
│
├── validation.csv
│   ├── Spanish (es):  2,500 samples
│   ├── Italian (it):  2,500 samples
│   └── Turkish (tr):  3,000 samples
│   └── Total: 8,000 multilingual validation samples
│
└── test.csv
    └── Unlabeled test set (not used in this project)
```

### Languages NOT in the Dataset

The following languages are **NOT available** in Jigsaw 2020:
- ❌ French (fr)
- ❌ German (de)
- ❌ Portuguese (pt)
- ❌ Russian (ru)

**To train models for these languages**, you need to:
1. Use the **Translation Approach** (translate English data)
2. Find external datasets for these languages
3. Use cross-lingual transfer learning

## Actual Dataset Statistics

### Sample Counts by Language (After Processing)

| Language | Source | Train | Val | Test | Held-out | Total | Toxic % |
|----------|--------|-------|-----|------|----------|-------|---------|
| **English (en)** | train.csv + val subset | 156,484 | 22,355 | 22,355 | 22,355 | 223,549 | 9.6% |
| **Spanish (es)** | validation.csv | 1,749 | 250 | 251 | 250 | 2,500 | 16.9% |
| **Italian (it)** | validation.csv | 1,749 | 250 | 251 | 250 | 2,500 | 19.5% |
| **Turkish (tr)** | validation.csv | 2,099 | 301 | 300 | 300 | 3,000 | 10.7% |
| **TOTAL** | | **162,081** | **23,156** | **23,157** | **23,155** | **231,549** | ~10% |

### Data Distribution Insights

1. **English Dominance**:
   - 223,549 samples (96.5% of total)
   - Highly imbalanced: only 9.6% toxic
   - Rich training data enables strong models

2. **Non-English Languages**:
   - 8,000 samples total (3.5% of total)
   - More balanced: 10-20% toxic
   - Small datasets limit performance potential

3. **Class Distribution Difference**:
   - English: 9.6% toxic (natural distribution)
   - Spanish: 16.9% toxic
   - Italian: 19.5% toxic
   - Turkish: 10.7% toxic

## Data Processing

### Split Strategy (70/10/10/10)

For each language:
1. **Train (70%)**: Model training
2. **Val (10%)**: Hyperparameter tuning, threshold optimization
3. **Test (10%)**: Final evaluation
4. **Held-out (10%)**: Future analysis, cross-validation

### English Data Processing

**Special handling** for English:
1. Load `jigsaw-toxic-comment-train.csv` (~224k samples)
2. Extract English subset from `validation.csv` (~1k samples)
3. Combine both sources
4. Apply 70/10/10/10 split
5. Result: 223,549 total samples

### Non-English Data Processing

For Spanish, Italian, Turkish:
1. Extract language subset from `validation.csv`
2. Apply 70/10/10/10 split
3. Result: ~2,500-3,000 samples per language

## Baseline Model Performance

### Achieved Results (TF-IDF + LogReg)

#### Without Stopwords

| Language | Test ROC-AUC | Test F1 | Test Precision | Test Recall | Threshold |
|----------|--------------|---------|----------------|-------------|-----------|
| en       | **0.9709**   | 0.7448  | 0.7293        | 0.7610     | 0.650     |
| es       | 0.8700       | 0.6364  | 0.6087        | 0.6667     | 0.470     |
| it       | 0.8534       | 0.5500  | 0.4648        | 0.6735     | 0.440     |
| tr       | **0.9628**   | 0.6667  | 0.5814        | 0.7812     | 0.450     |

#### With Stopwords

| Language | Test ROC-AUC | Test F1 | Test Precision | Test Recall | Threshold |
|----------|--------------|---------|----------------|-------------|-----------|
| en       | **0.9686**   | 0.7371  | 0.7339        | 0.7404     | 0.680     |
| es       | 0.8604       | 0.6067  | 0.5745        | 0.6429     | 0.470     |
| it       | 0.8430       | 0.5192  | 0.4909        | 0.5510     | 0.470     |
| tr       | **0.9569**   | 0.7368  | 0.8400        | 0.6562     | 0.570     |

### Performance Analysis

**English**:
- Excellent ROC-AUC (0.97+)
- Large dataset enables strong discrimination
- Stopwords: minimal impact (-0.002 ROC-AUC)

**Turkish**:
- Surprisingly excellent ROC-AUC (0.96+)
- Despite small dataset (3k samples)
- Better with stopwords for precision (0.84)

**Spanish & Italian**:
- Good ROC-AUC (0.84-0.87)
- Limited by small dataset size
- Room for improvement with more data

## Expanding to 8 Languages

### Missing Languages Strategy

To train models for French, German, Portuguese, and Russian:

#### Option 1: Translation Approach ✅
1. Translate English training data to target languages
2. Use Google Translate API, DeepL, or MBART
3. Train models on translated data
4. **Pros**: Large training sets, consistent labels
5. **Cons**: Translation quality, "translationese" effect

#### Option 2: External Datasets
1. Find toxicity datasets in target languages
2. Examples:
   - German: GermEval 2018/2019
   - French: French toxic comment datasets
   - Portuguese: Brazilian Portuguese datasets
3. **Pros**: Natural language, cultural relevance
4. **Cons**: Different label definitions, smaller size

#### Option 3: Cross-lingual Transfer ✅
1. Train multilingual BERT/XLM-RoBERTa on English
2. Fine-tune on small target language data
3. Leverage cross-lingual embeddings
4. **Pros**: State-of-the-art performance
5. **Cons**: Requires GPU, more complex

### Recommended Approach

**For this project**: Use Option 1 (Translation Approach)
- Translate English training data (224k samples) to each language
- Provides consistent training data across all languages
- Enables fair comparison between original and translated approaches

## Data Format

### CSV Structure

```csv
comment_text,toxic,lang
"This is a sample comment",0,en
"Another example text",1,es
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `comment_text` | string | Comment text content |
| `toxic` | integer | Binary label: 1 = toxic, 0 = non-toxic |
| `lang` | string | ISO 639-1 code (en, es, it, tr) |

### Language Codes

| Code | Language | Available | Samples |
|------|----------|-----------|---------|
| en   | English    | ✅ | 223,549 |
| es   | Spanish    | ✅ | 2,500 |
| it   | Italian    | ✅ | 2,500 |
| tr   | Turkish    | ✅ | 3,000 |
| fr   | French     | ❌ | Need translation |
| de   | German     | ❌ | Need translation |
| pt   | Portuguese | ❌ | Need translation |
| ru   | Russian    | ❌ | Need translation |

## Reproducing Data Splits

### Prerequisites

```bash
# Download Jigsaw 2020 data
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
unzip jigsaw-multilingual-toxic-comment-classification.zip -d data/raw/jigsaw-2020/
```

### Create Splits

```bash
# For available languages only
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es it tr
```

### Verify Data

```bash
# Check what's available
python code/scripts/check_data.py
```

Expected output:
```
✅ Found validation file: data/raw/jigsaw-2020/validation.csv
Total samples: 8000

LANGUAGE DISTRIBUTION IN VALIDATION SET
es  :  2500 samples ( 422 toxic,  16.9%)
it  :  2500 samples ( 488 toxic,  19.5%)
tr  :  3000 samples ( 320 toxic,  10.7%)

⚠️  MISSING LANGUAGES:
  - de
  - en (in separate train file)
  - fr
  - pt
  - ru

PROCESSED DATA AVAILABILITY
✅ en: Total: 223549 samples
✅ es: Total: 2500 samples
✅ it: Total: 2500 samples
✅ tr: Total: 3000 samples
```

## Data Quality Considerations

### Known Limitations

1. **Limited Language Coverage**:
   - Only 4 languages natively available
   - Missing major languages (fr, de, pt, ru)

2. **Dataset Size Imbalance**:
   - English: 223k samples (97%)
   - Other languages: 2-3k samples each (3%)
   - Huge disparity affects cross-lingual comparison

3. **Class Imbalance**:
   - English: 9.6% toxic (realistic)
   - Others: 10-20% toxic (artificially balanced)
   - Different distributions require different strategies

4. **Translation Origin**:
   - Non-English validation samples were machine-translated from English
   - May not reflect natural toxic language in those languages
   - Cultural context may be lost

### Implications for Modeling

1. **English Models**:
   - Large dataset → strong performance expected
   - Class imbalance → need balanced class weights
   - Threshold optimization crucial

2. **Non-English Models**:
   - Small datasets → overfitting risk
   - More balanced → standard training works
   - Transfer learning recommended

3. **Cross-lingual Transfer**:
   - English model can help bootstrap other languages
   - Multilingual BERT essential for missing languages
   - Zero-shot transfer possible but less accurate

## Future Data Expansion

### Translation Pipeline (Planned)

```bash
# Translate English data to target languages
python code/scripts/translate_data.py \
    --source en \
    --targets fr de pt ru \
    --method google  # or deepl, mbart
```

### Expected Results

With translation approach:
- **French**: ~224k translated samples
- **German**: ~224k translated samples
- **Portuguese**: ~224k translated samples
- **Russian**: ~224k translated samples

This will enable:
- Fair comparison between original and translated data
- Consistent training set sizes across all languages
- Better models for missing languages

## Data Privacy and Ethics

### Privacy
- All comments are from public online discussions
- Personal information anonymized by Jigsaw
- No PII included

### Ethical Considerations
- Content contains offensive language
- Handle with care for mental health
- Consider bias in training data
- Toxicity definitions vary across cultures

### Content Warning

⚠️ **Warning**: This dataset contains:
- Profanity and hate speech
- Threats and harassment
- Other harmful content

## References

1. Jigsaw/Conversation AI. (2020). Jigsaw Multilingual Toxic Comment Classification. Kaggle.
   https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

2. Ranasinghe, T., & Zampieri, M. (2020). Multilingual Offensive Language Identification with Cross-lingual Embeddings. EMNLP 2020.

3. Caselli, T., et al. (2021). HateBERT: Retraining BERT for Abusive Language Detection in English. ACL 2021.

## Summary

### What We Have
✅ **4 languages** with native data (en, es, it, tr)  
✅ **231,549 total samples**  
✅ **Strong baseline models** (0.84-0.97 ROC-AUC)  
✅ **Complete data pipeline**  

### What We Need
⚠️ **4 languages** require translation (fr, de, pt, ru)  
⚠️ **Translation approach** implementation  
⚠️ **Cross-lingual evaluation** framework  

### Next Steps
1. ✅ Complete baseline models for available languages
2. 🔄 Implement translation pipeline
3. 🔄 Train models on translated data
4. 🔄 Compare original vs. translated approaches
5. 🔄 Implement multilingual BERT models

---

**Last Updated**: 2025-11-13
**Status**: 4/8 languages available with native data  
**Recommendation**: Proceed with translation approach for missing languages