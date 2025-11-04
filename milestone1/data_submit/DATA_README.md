# data_submit (sampled)

This folder contains a compact subset of the multilingual toxicity data for grading:

- **train_sample.csv** — stratified sample from `jigsaw-toxic-comment-train.csv` (stratified by `toxic`), capped at **11,178** rows.
- **dev.csv** — copied from `validation.csv` (Jigsaw 2020).
- **test.csv** — copied from `test.csv` (Jigsaw 2020).
- **test_labels.csv** — copied from `test_labels.csv` (if provided by the release).

**Source directory:** `/Users/alex/Desktop/data/jigsaw-multilingual-toxic-comment-classification`

**Provenance:**
- Jigsaw Multilingual Toxic Comment Classification (2020): https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification
- (Historical English baseline) Jigsaw 2018: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- Civil Comments archive: https://storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments.zip

**How this sample was created:**
- Loaded full train from `jigsaw-toxic-comment-train.csv`.
- Built stratification buckets by toxicity (`toxic` or `target>=0.5`), then sampled up to 20,000 rows or 5% of the full train, whichever is smaller.
- Preserved full dev/test for reliable evaluation.

Generated on: 2025-11-04T14:53:28
