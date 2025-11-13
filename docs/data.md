# Data Documentation

## Overview

This project uses the **Jigsaw Multilingual Toxic Comment Classification** dataset from Kaggle, which provides labeled toxic comments in 8 languages: English, Spanish, French, German, Italian, Portuguese, Russian, and Turkish.

## Data Source

### Primary Dataset: Jigsaw 2020 Multilingual Toxic Comment Classification

- **Competition URL**: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification
- **Download Command**:
  ```bash
  kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
  ```
- **License**: The dataset is provided under the [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) license
- **Citation**:
  ```
  Jigsaw/Conversation AI. (2020). Jigsaw Multilingual Toxic Comment Classification.
  Kaggle. https://kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification
  ```

### Dataset Components

The original dataset consists of:

1. **English Training Data** (`jigsaw-toxic-comment-train.csv`)
   - ~224,000 English comments with toxicity labels
   - Binary classification: toxic (1) or non-toxic (0)

2. **Multilingual Validation Data** (`validation.csv`)
   - ~8,000 comments across 7 languages (excluding English from training)
   - ~1,000 samples per language
   - Languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Russian (ru), Turkish (tr)

3. **Test Data** (`test.csv`)
   - Unlabeled test set for competition submission
   - Not used in this project

## Data Processing

### Split Strategy

We apply a **70/10/10/10** split strategy for each language:

- **Train Set (70%)**: For model training
- **Validation Set (10%)**: For hyperparameter tuning and model selection
- **Test Set (10%)**: For final model evaluation
- **Held-out Set (10%)**: Reserved for future analysis and cross-validation

### Processing Pipeline

1. **Load Raw Data**:
   - English: Combine training set + English validation subset
   - Other Languages: Extract from multilingual validation set

2. **Stratified Splitting**:
   - Maintain class distribution (toxic vs. non-toxic) across all splits
   - Random seed: 42 (for reproducibility)

3. **Output Structure**:
   ```
   data/processed/
   ├── en/
   │   ├── train.csv
   │   ├── val.csv
   │   ├── test.csv
   │   └── heldout.csv
   ├── es/
   │   ├── train.csv
   │   ├── val.csv
   │   ├── test.csv
   │   └── heldout.csv
   ...
   └── dataset_summary.csv
   ```

## Dataset Statistics

### Sample Counts by Language and Split

> **Note**: The exact numbers will be generated after running the data splitting script. Below are expected ranges based on the original dataset distribution.

| Language | Split    | Expected Samples | Toxic Count | Toxic Ratio (%) |
|----------|----------|------------------|-------------|-----------------|
| English  | Train    | ~157,000         | ~14,300     | ~9.1%          |
| English  | Val      | ~22,500          | ~2,000      | ~9.1%          |
| English  | Test     | ~22,500          | ~2,000      | ~9.1%          |
| English  | Held-out | ~22,500          | ~2,000      | ~9.1%          |
| Spanish  | Train    | ~700             | ~350        | ~50%           |
| Spanish  | Val      | ~100             | ~50         | ~50%           |
| Spanish  | Test     | ~100             | ~50         | ~50%           |
| Spanish  | Held-out | ~100             | ~50         | ~50%           |
| French   | Train    | ~700             | ~350        | ~50%           |
| French   | Val      | ~100             | ~50         | ~50%           |
| French   | Test     | ~100             | ~50         | ~50%           |
| French   | Held-out | ~100             | ~50         | ~50%           |
| German   | Train    | ~700             | ~350        | ~50%           |
| German   | Val      | ~100             | ~50         | ~50%           |
| German   | Test     | ~100             | ~50         | ~50%           |
| German   | Held-out | ~100             | ~50         | ~50%           |
| Italian  | Train    | ~700             | ~350        | ~50%           |
| Italian  | Val      | ~100             | ~50         | ~50%           |
| Italian  | Test     | ~100             | ~50         | ~50%           |
| Italian  | Held-out | ~100             | ~50         | ~50%           |
| Portuguese| Train   | ~700             | ~350        | ~50%           |
| Portuguese| Val     | ~100             | ~50         | ~50%           |
| Portuguese| Test    | ~100             | ~50         | ~50%           |
| Portuguese| Held-out| ~100             | ~50         | ~50%           |
| Russian  | Train    | ~700             | ~350        | ~50%           |
| Russian  | Val      | ~100             | ~50         | ~50%           |
| Russian  | Test     | ~100             | ~50         | ~50%           |
| Russian  | Held-out | ~100             | ~50         | ~50%           |
| Turkish  | Train    | ~700             | ~350        | ~50%           |
| Turkish  | Val      | ~100             | ~50         | ~50%           |
| Turkish  | Test     | ~100             | ~50         | ~50%           |
| Turkish  | Held-out | ~100             | ~50         | ~50%           |

### Total Statistics Summary

| Metric                    | Value                |
|---------------------------|----------------------|
| **Total Languages**       | 8                    |
| **Total Samples**         | ~231,000             |
| **English Samples**       | ~224,500 (97%)       |
| **Non-English Samples**   | ~7,000 (3%)          |
| **Overall Toxic Ratio**   | ~10.5%               |
| **English Toxic Ratio**   | ~9.1%                |
| **Non-English Toxic Ratio** | ~50%               |

### Class Distribution Notes

1. **English Data**:
   - **Highly imbalanced**: ~9% toxic, ~91% non-toxic
   - Reflects real-world distribution of toxic content
   - Large dataset size allows for effective minority class learning

2. **Non-English Data**:
   - **Balanced**: ~50% toxic, ~50% non-toxic
   - Validation set was deliberately balanced for fair evaluation
   - Smaller dataset size (~1,000 samples per language)

3. **Implications**:
   - English models need strategies for imbalanced learning (e.g., class weights, focal loss)
   - Non-English models can use standard training approaches
   - Cross-lingual transfer learning is crucial for low-resource languages

## Data Format

### CSV Structure

All split files follow this format:

```csv
comment_text,toxic,lang
"This is a sample comment",0,en
"Another example text",1,es
```

### Column Descriptions

| Column         | Type    | Description                                    |
|----------------|---------|------------------------------------------------|
| `comment_text` | string  | The text content of the comment               |
| `toxic`        | integer | Binary label: 1 = toxic, 0 = non-toxic       |
| `lang`         | string  | ISO 639-1 language code (en, es, fr, etc.)   |

### Language Codes

| Code | Language   | Native Name  |
|------|------------|--------------|
| en   | English    | English      |
| es   | Spanish    | Español      |
| fr   | French     | Français     |
| de   | German     | Deutsch      |
| it   | Italian    | Italiano     |
| pt   | Portuguese | Português    |
| ru   | Russian    | Русский      |
| tr   | Turkish    | Türkçe       |

## Data Quality

### Quality Assurance

1. **Label Quality**:
   - Labels were provided by Jigsaw/Conversation AI
   - Based on human annotations
   - Some label noise is expected (inherent in human judgment)

2. **Text Quality**:
   - Comments are from real online discussions
   - May contain typos, slang, and informal language
   - Unicode characters are preserved

3. **Language Detection**:
   - Language labels are provided in the dataset
   - Our pipeline includes optional language detection verification

### Known Issues

1. **Data Imbalance**:
   - English: heavily imbalanced (9% toxic)
   - Other languages: balanced (50% toxic) but small sample size

2. **Translation Quality**:
   - Non-English validation samples were translated from English
   - May not reflect natural native language patterns
   - "Translationese" effect possible

3. **Cultural Differences**:
   - Toxicity definitions may vary across cultures
   - Some expressions toxic in one language may be acceptable in another

## Reproducing the Data Splits

### Prerequisites

```bash
# Install required packages
pip install pandas numpy scikit-learn pyyaml

# Download Jigsaw 2020 data
kaggle competitions download -c jigsaw-multilingual-toxic-comment-classification
unzip jigsaw-multilingual-toxic-comment-classification.zip -d data/raw/jigsaw-2020/
```

### Run the Splitting Script

```bash
# Default split (70/10/10/10)
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml

# Custom split ratios
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.05 \
    --heldout_ratio 0.05

# Specific languages only
python code/scripts/make_multilingual_splits.py \
    --config configs/detoxify_multilingual.yaml \
    --languages en es fr
```

### Verification

After running the script, verify the splits:

```bash
# Check output structure
ls data/processed/

# View summary statistics
cat data/processed/dataset_summary.csv

# Quick sample count check
wc -l data/processed/en/*.csv
```

## Additional Data Sources (Optional)

### Historical Jigsaw Datasets

These are **not required** but can be used for additional training data or comparison:

1. **Jigsaw 2018**: Toxic Comment Classification Challenge
   - URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
   - ~160,000 English comments with multi-label toxicity categories

2. **Jigsaw 2019**: Unintended Bias in Toxicity Classification
   - URL: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
   - ~1,800,000 English comments with bias-related metadata

### External Translation Resources

For the **translation approach**, you may need:

- **Google Translate API**: For automatic translation
- **DeepL API**: Higher quality translations (paid)
- **MBART**: Facebook's multilingual translation model

## Data Privacy and Ethics

### Privacy Considerations

- All comments are **publicly available** online discussions
- Personal information has been **anonymized** by Jigsaw
- No user IDs or personally identifiable information included

### Ethical Use

1. **Purpose**: This dataset should be used for research and development of toxicity detection systems
2. **Limitations**: Models trained on this data may not generalize to all contexts
3. **Bias**: Be aware of potential biases in the training data
4. **Impact**: Consider the social impact of deploying toxicity detection systems

### Content Warning

⚠️ **Warning**: This dataset contains offensive, toxic, and potentially disturbing language. The content includes profanity, hate speech, threats, and other harmful content. Handle with care and consider the mental health of team members working with this data.

## References

1. Jigsaw/Conversation AI. (2020). Jigsaw Multilingual Toxic Comment Classification. Kaggle. https://kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification

2. Borkan, D., Dixon, L., Sorensen, J., Thain, N., & Vasserman, L. (2019). Nuanced metrics for measuring unintended bias with real data for text classification. In Companion Proceedings of The 2019 World Wide Web Conference (pp. 491-500).

3. Dixon, L., Li, J., Sorensen, J., Thain, N., & Vasserman, L. (2018). Measuring and mitigating unintended bias in text classification. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society (pp. 67-73).

## Updates and Changelog

### Version 1.0 (2024-01-XX)

- Initial data documentation
- Processed Jigsaw 2020 multilingual dataset
- Created 70/10/10/10 splits for 8 languages
- Generated dataset summary statistics

---

**Last Updated**: 2025-11-13
**Maintainer**: Alex Yang
**Contact**: hanqing.yang189@gmail.com
