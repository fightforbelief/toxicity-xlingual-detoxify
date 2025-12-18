"""
LoRA Fine-tuning with Hard Sample Mining / Active Learning

This script fine-tunes XLM-R with LoRA adapters, using sample weights to focus on hard examples.

Key Features:
- Uses pre-computed sample weights from identify_hard_samples.py
- Custom WeightedSampler for oversampling hard examples
- Weighted loss functions (BCE, Focal Loss)
- Supports both Detoxify checkpoint and vanilla XLM-R

Usage:
    python hard_sample_finetune.py \
        --config configs/hard_sample_config.yaml \
        --languages es it tr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import yaml

warnings.filterwarnings("ignore")


# ============================================================================
# Config and Helper Functions
# ============================================================================

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Select best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ============================================================================
# Custom Weighted Trainer
# ============================================================================

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that uses sample weights in loss computation.
    
    Supports:
    - Weighted BCE Loss
    - Weighted Focal Loss
    """
    
    def __init__(
        self,
        loss_type: str = "weighted_bce",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        print(f"  Using {loss_type.upper()} loss")
        if loss_type == "focal":
            print(f"    Focal alpha: {focal_alpha}, gamma: {focal_gamma}")
    
    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Compute weighted loss."""
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights", None)
        
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        
        if self.loss_type == "weighted_bce":
            loss = self._weighted_bce_loss(logits, labels, sample_weights)
        elif self.loss_type == "focal":
            loss = self._focal_loss(logits, labels, sample_weights)
        else:
            # Default BCE
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _weighted_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Weighted Binary Cross-Entropy Loss."""
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_fct(logits, labels.float())
        
        if sample_weights is not None:
            loss = loss * sample_weights
        
        return loss.mean()
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Focal Loss for hard example mining.
        FL(p) = -alpha * (1 - p)^gamma * log(p)
        """
        probs = torch.sigmoid(logits)
        targets = labels.float()
        
        # Compute p_t and alpha_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # BCE
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Combine
        loss = alpha_t * focal_weight * bce
        
        # Apply sample weights if provided
        if sample_weights is not None:
            loss = loss * sample_weights
        
        return loss.mean()


# ============================================================================
# Data Pipeline with Sample Weights
# ============================================================================

def load_weighted_data(
    data_path: Path,
    text_col: str,
    label_col: str,
    weight_col: str = 'sample_weight'
) -> Dict[str, Any]:
    """Load data with sample weights."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check required columns
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in {data_path}")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {data_path}")
    
    # Add default weights if not present
    if weight_col not in df.columns:
        print(f"  ⚠️  No '{weight_col}' column found, using uniform weights (1.0)")
        df[weight_col] = 1.0
    
    # Ensure proper types
    df[label_col] = df[label_col].astype(int)
    df[weight_col] = df[weight_col].astype(float)
    
    print(f"  Loaded {len(df)} samples")
    print(f"    Toxic: {(df[label_col]==1).sum()} ({(df[label_col]==1).mean()*100:.1f}%)")
    print(f"    Weight range: [{df[weight_col].min():.2f}, {df[weight_col].max():.2f}]")
    print(f"    Weight mean: {df[weight_col].mean():.2f}")
    
    return {
        'texts': df[text_col].tolist(),
        'labels': df[label_col].values,
        'weights': df[weight_col].values,
        'df': df
    }


def prepare_datasets_with_weights(
    train_path: Path,
    val_path: Path,
    text_col: str,
    label_col: str,
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Prepare HuggingFace datasets with sample weights."""
    
    print(f"  Loading training data...")
    train_data = load_weighted_data(train_path, text_col, label_col)
    
    print(f"  Loading validation data...")
    val_data = load_weighted_data(val_path, text_col, label_col)
    
    # Create HF datasets
    train_ds = Dataset.from_dict({
        'text': train_data['texts'],
        'label': train_data['labels'],
        'sample_weight': train_data['weights']
    })
    
    val_ds = Dataset.from_dict({
        'text': val_data['texts'],
        'label': val_data['labels'],
        'sample_weight': val_data['weights']
    })
    
    # Tokenization
    def tokenize_fn(batch):
        enc = tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )
        enc['labels'] = batch['label']
        enc['sample_weights'] = batch['sample_weight']
        return enc
    
    print(f"  Tokenizing...")
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    
    # Set format
    cols = ['input_ids', 'attention_mask', 'labels', 'sample_weights']
    if 'token_type_ids' in train_ds.column_names:
        cols.append('token_type_ids')
    
    train_ds.set_format(type='torch', columns=cols)
    val_ds.set_format(type='torch', columns=cols)
    
    return {
        'train': train_ds,
        'val': val_ds,
        'train_weights': train_data['weights'],
        'train_labels': train_data['labels']
    }


# ============================================================================
# Model Building
# ============================================================================

def build_lora_model(
    base_model_name: str,
    lora_config: Dict[str, Any],
    use_detoxify: bool = False
) -> AutoModelForSequenceClassification:
    """
    Build LoRA model from base checkpoint.
    
    Args:
        base_model_name: HuggingFace model name
        lora_config: LoRA configuration dict
        use_detoxify: Whether to use Detoxify's pretrained checkpoint
    """
    if use_detoxify:
        model_name = "unitary/multilingual-toxic-xlm-roberta"
        print(f"  Loading Detoxify checkpoint: {model_name}")
    else:
        model_name = base_model_name
        print(f"  Loading base model: {model_name}")
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    
    # Apply LoRA
    lora_cfg = LoraConfig(
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 16),
        target_modules=lora_config.get('target_modules', ['query', 'key', 'value']),
        lora_dropout=lora_config.get('dropout', 0.1),
        bias=lora_config.get('bias', 'none'),
        task_type="SEQ_CLS",
    )
    
    print(f"  Applying LoRA (r={lora_cfg.r}, alpha={lora_cfg.lora_alpha})...")
    lora_model = get_peft_model(base_model, lora_cfg)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return lora_model


# ============================================================================
# Main Training Function
# ============================================================================

def train_language(
    lang: str,
    config: Dict[str, Any],
    device: torch.device
):
    """Train LoRA model for a single language with hard sample weighting."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {lang.upper()}")
    print(f"{'='*70}")
    
    # Extract config
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    hard_sample_cfg = config.get('hard_sample_mining', {})
    
    processed_dir = Path(data_cfg['processed_dir'])
    text_col = data_cfg.get('text_col', 'comment_text')
    label_col = data_cfg.get('label_col', 'toxic')
    
    base_model_name = model_cfg.get('base_model_name', 'xlm-roberta-base')
    max_length = model_cfg.get('max_length', 256)
    use_detoxify = model_cfg.get('use_detoxify_checkpoint', False)
    
    output_root = Path(train_cfg['output_dir'])
    batch_size = train_cfg.get('batch_size', 8)
    # lr = train_cfg.get('learning_rate', 2e-5)
    lr = float(train_cfg.get('learning_rate', 2e-5))
    num_epochs = train_cfg.get('num_epochs', 3)
    weight_decay = train_cfg.get('weight_decay', 0.01)
    
    loss_type = hard_sample_cfg.get('loss_type', 'weighted_bce')
    focal_alpha = hard_sample_cfg.get('focal_alpha', 0.25)
    focal_gamma = hard_sample_cfg.get('focal_gamma', 2.0)
    
    # Load tokenizer
    tokenizer_name = "unitary/multilingual-toxic-xlm-roberta" if use_detoxify else base_model_name
    print(f"\n📚 Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Prepare datasets
    print(f"\n📊 Preparing datasets for {lang}...")
    lang_dir = processed_dir / lang
    
    # Check if weighted data exists
    train_weighted_path = lang_dir / "train_weighted.csv"
    if not train_weighted_path.exists():
        print(f"  ⚠️  Weighted training data not found: {train_weighted_path}")
        print(f"  Using original training data without sample weights")
        train_path = lang_dir / "train.csv"
    else:
        train_path = train_weighted_path
        print(f"  ✓ Using weighted training data")
    
    val_path = lang_dir / "val.csv"
    
    datasets = prepare_datasets_with_weights(
        train_path, val_path,
        text_col, label_col,
        tokenizer, max_length
    )
    
    # Build model
    print(f"\n🏗️  Building LoRA model...")
    model = build_lora_model(
        base_model_name,
        model_cfg.get('lora', {}),
        use_detoxify=use_detoxify
    )
    model.to(device)
    
    # Prepare output directory
    lang_out_dir = output_root / lang
    lang_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(lang_out_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    
    # Create trainer
    print(f"\n🎯 Creating trainer with {loss_type.upper()} loss...")
    trainer = WeightedLossTrainer(
        loss_type=loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    trainer.train()
    
    # Save LoRA adapter
    save_path = lang_out_dir / "lora_adapter"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 Saving LoRA adapter to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save training info
    with open(save_path / "training_info.txt", "w") as f:
        f.write(f"Language: {lang}\n")
        f.write(f"Base model: {base_model_name}\n")
        f.write(f"Use Detoxify checkpoint: {use_detoxify}\n")
        f.write(f"Loss type: {loss_type}\n")
        f.write(f"Training samples: {len(datasets['train'])}\n")
        f.write(f"Validation samples: {len(datasets['val'])}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch_size}\n")
    
    print(f"✅ Training complete for {lang}!")
    
    # Clean up
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-R with LoRA using hard sample mining"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to train (e.g., es it tr). If omitted, use config."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/mps/cpu). If None, auto-detect."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config_path = Path(args.config)
    print(f"📄 Loading config from {config_path}...")
    config = load_config(config_path)
    
    # Get languages
    if args.languages:
        languages = args.languages
    else:
        languages = config['data'].get('languages', ['es', 'it', 'tr'])
    
    print(f"🌍 Languages: {languages}")
    
    # Get device
    device = torch.device(args.device) if args.device else get_device()
    print(f"🖥️  Device: {device}")
    
    # Train each language
    for lang in languages:
        try:
            train_language(lang, config, device)
        except Exception as e:
            print(f"\n❌ Error training {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ ALL TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

