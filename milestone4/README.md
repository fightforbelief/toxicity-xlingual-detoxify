# Milestone 4: Hard Sample Mining Extension

This extension improves multilingual toxicity detection by identifying and focusing training on hard samples (misclassified, uncertain, or high-confidence errors) from a baseline model.

## Quick Start

### Step 1: Generate Baseline Predictions on Training Set

```bash
python generate_train_baseline_predictions.py --languages es it tr
```

This generates baseline predictions needed for hard sample identification:
- `data/baseline_predictions/detoxify_es_train.csv`
- `data/baseline_predictions/detoxify_it_train.csv`
- `data/baseline_predictions/detoxify_tr_train.csv`

### Step 2: Identify Hard Samples

```bash
python identify_hard_samples.py --config configs/hard_sample_config.yaml
```

This creates weighted training data with sample weights based on difficulty.

### Step 3: Fine-tune with Hard Sample Weighting

```bash
python hard_sample_finetune.py --config configs/hard_sample_config.yaml --languages es it tr
```

Trains LoRA-enhanced XLM-R models focusing on hard samples using:
- Weighted BCE Loss
- Focal Loss (optional)
- Sample weighting strategy

### Step 4: Generate Predictions

```bash
python hard_sample_inference.py --config configs/hard_sample_config.yaml --languages es it tr
```

Generates predictions on test set:
- `output/predictions/hard_sample_es.csv`
- `output/predictions/hard_sample_it.csv`
- `output/predictions/hard_sample_tr.csv`

### Step 5: Evaluate Results

```bash
python score.py --pred_dir output/predictions --run_tag hard_sample --languages es it tr --output output/predictions/hard_sample_metrics.csv
```

Computes metrics: ROC-AUC, F1, Precision, Recall, Accuracy, FPR, FNR.


## Configuration

Edit `configs/hard_sample_config.yaml` to customize:
- **Hard sample mining strategies**: uncertainty, misclassified, high_confidence_errors
- **Loss functions**: BCE, Weighted BCE, Focal Loss
- **Training hyperparameters**: learning rate, batch size, epochs
- **LoRA parameters**: rank, alpha, dropout

## Directory Structure

```
milestone4/
  ├── data/
  │   ├── processed/              # Train/val/test splits
  │   │   ├── es/, it/, tr/
  │   └── baseline_predictions/   # Baseline model predictions
  ├── configs/
  │   └── hard_sample_config.yaml # Main configuration
  ├── output/
  │   ├── predictions/            # Model predictions
  │   └── runs/                   # Trained models
  └── *.py                        # All scripts
```

## Key Scripts

- `generate_train_baseline_predictions.py`: Generate baseline predictions on training data
- `identify_hard_samples.py`: Identify hard samples and assign weights
- `hard_sample_finetune.py`: Train LoRA models with hard sample focus
- `hard_sample_inference.py`: Generate predictions on test set
- `score.py`: Evaluate model performance

## Environment Setup

```bash
# Create conda environment
conda create -n toxicity python=3.9 -y
conda activate toxicity

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements_hard_sample.txt
```


