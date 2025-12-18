"""
Inference script for Hard Sample Mining LoRA models

Loads trained LoRA adapters and generates predictions on test data.

Usage:
    python hard_sample_inference.py \
        --config configs/hard_sample_config.yaml \
        --languages es it tr \
        --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import yaml

warnings.filterwarnings("ignore")


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


def load_lora_model(
    adapter_path: Path,
    base_model_name: str,
    use_detoxify: bool,
    device: torch.device
):
    """Load base model + LoRA adapter."""
    if use_detoxify:
        model_name = "unitary/multilingual-toxic-xlm-roberta"
    else:
        model_name = base_model_name
    
    print(f"  Loading base model: {model_name}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )
    
    print(f"  Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.to(device)
    model.eval()
    
    return model


def predict_on_data(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device
) -> np.ndarray:
    """Run inference on texts and return probabilities."""
    model.eval()
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)


def run_inference_for_language(
    lang: str,
    config: Dict[str, Any],
    split: str,
    device: torch.device
):
    """Run inference for a single language."""
    
    print(f"\n{'='*70}")
    print(f"INFERENCE: {lang.upper()} ({split} set)")
    print(f"{'='*70}")
    
    # Extract config
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    eval_cfg = config['evaluation']
    
    processed_dir = Path(data_cfg['processed_dir'])
    text_col = data_cfg.get('text_col', 'comment_text')
    label_col = data_cfg.get('label_col', 'toxic')
    
    base_model_name = model_cfg.get('base_model_name', 'xlm-roberta-base')
    max_length = model_cfg.get('max_length', 256)
    use_detoxify = model_cfg.get('use_detoxify_checkpoint', False)
    
    output_root = Path(train_cfg['output_dir'])
    batch_size = train_cfg.get('batch_size', 16)
    
    pred_dir = Path(eval_cfg['output_dir'])
    run_tag = eval_cfg.get('run_tag', 'hard_sample')
    threshold = eval_cfg.get('threshold', 0.5)
    
    # Load test data
    data_path = processed_dir / lang / f"{split}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    print(f"\n📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Samples: {len(df)}")
    
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found. Available: {list(df.columns)}")
    
    texts = df[text_col].tolist()
    
    # Load model
    adapter_path = output_root / lang / "lora_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    
    tokenizer_name = "unitary/multilingual-toxic-xlm-roberta" if use_detoxify else base_model_name
    print(f"\n📚 Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    print(f"\n🏗️  Loading model...")
    model = load_lora_model(adapter_path, base_model_name, use_detoxify, device)
    
    # Run inference
    print(f"\n🔮 Running inference...")
    probs = predict_on_data(model, tokenizer, texts, max_length, batch_size, device)
    
    # Create predictions DataFrame
    pred_df = df.copy()
    pred_df['y_prob'] = probs
    pred_df['y_pred'] = (probs >= threshold).astype(int)
    
    # Add ground truth if available
    if label_col in df.columns:
        pred_df['y_true'] = df[label_col].astype(int)
        
        # Compute quick metrics
        y_true = pred_df['y_true'].values
        y_pred = pred_df['y_pred'].values
        y_prob = pred_df['y_prob'].values
        
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            print(f"\n📊 Quick Metrics:")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            print(f"  F1:        {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
        except Exception as e:
            print(f"  ⚠️  Could not compute metrics: {e}")
    
    # Save predictions
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / f"{run_tag}_{lang}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"\n💾 Saved predictions to: {pred_file}")
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"✅ Inference complete for {lang}!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with Hard Sample Mining LoRA models"
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
        help="Languages to run inference on (e.g., es it tr)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "heldout"],
        help="Which split to run inference on"
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
    print(f"📊 Split: {args.split}")
    
    # Get device
    device = torch.device(args.device) if args.device else get_device()
    print(f"🖥️  Device: {device}")
    
    # Run inference for each language
    for lang in languages:
        try:
            run_inference_for_language(lang, config, args.split, device)
        except Exception as e:
            print(f"\n❌ Error with {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ ALL INFERENCE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

