#Load the Detoxify base model’s underlying HF checkpoint
#Apply LoRA adapter via peft
# Use class-weighted BCE loss
# Train on train.csv
# Validate on val.csv
# Save LoRA adapter weights

"""
LoRA-based fine-tuning of an XLM-R classifier for toxicity detection.

- Loads per-language train/val splits from data/processed/{lang}
- Builds an XLM-R sequence classification model
- Wraps it with LoRA adapters (via `peft`)
- Uses class-weighted BCEWithLogits loss to handle class imbalance
- Trains per language and saves LoRA adapters to output/runs/lora/{lang}

Example:

    python code/lora_finetune.py \
        --config configs/lora_finetune.yaml

Requirements:
    pip install transformers datasets peft torch pyyaml pandas
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_processed_dir(cfg: Dict[str, Any]) -> Path:
    try:
        return Path(cfg["data"]["processed_dir"])
    except KeyError:
        raise KeyError("Config missing data.processed_dir")


def get_languages(cfg: Dict[str, Any], arg_langs: Optional[List[str]]) -> List[str]:
    if arg_langs:
        return arg_langs
    try:
        langs = cfg["data"]["languages"]
        if isinstance(langs, list) and langs:
            return langs
    except KeyError:
        pass
    raise ValueError(
        "No languages specified. Provide --languages or data.languages in YAML."
    )


def compute_class_weights(labels: np.ndarray) -> float:
    """
    Compute positive-class weight for BCEWithLogitsLoss.

    pos_weight = N_neg / N_pos

    Returns a scalar float.
    """
    labels = labels.astype(int)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    if n_pos == 0:
        # avoid divide-by-zero: fallback to weight 1.0
        return 1.0

    return float(n_neg / max(1, n_pos))


# ---------------------------------------------------------------------
# Custom Trainer with class-weighted BCE
# ---------------------------------------------------------------------


class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store as a 1D tensor; move to device in compute_loss
        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # Do NOT mutate original dict in case HF reuses it
        labels = inputs["labels"]  # (batch,)

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits.view(-1)  # (batch,)

        # move weight to correct device
        pos_weight = self.pos_weight.to(logits.device)

        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(logits, labels.float())

        if return_outputs:
            return loss, outputs
        return loss


# ---------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------


def prepare_datasets_for_language(
    processed_dir: Path,
    lang: str,
    text_col: str,
    label_col: str,
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """
    For a given language load train/val CSVs and tokenize them into HF Datasets.
    """
    lang_dir = processed_dir / lang
    train_path = lang_dir / "train.csv"
    val_path = lang_dir / "val.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val split: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    if text_col not in train_df.columns or label_col not in train_df.columns:
        raise KeyError(
            f"Expected columns '{text_col}' and '{label_col}' in {train_path}.\n"
            f"Found columns: {list(train_df.columns)}"
        )

    # Ensure label is 0/1
    train_df[label_col] = train_df[label_col].astype(int)
    val_df[label_col] = val_df[label_col].astype(int)

    # HuggingFace Dataset
    train_ds = Dataset.from_pandas(
        train_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    )
    val_ds = Dataset.from_pandas(
        val_df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    )

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # Set format for PyTorch
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        cols.append("token_type_ids")
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    return {
        "train": train_ds,
        "val": val_ds,
        "train_labels": train_df[label_col].values,
    }


# ---------------------------------------------------------------------
# Model + LoRA
# ---------------------------------------------------------------------


def build_lora_model(base_model_name: str) -> AutoModelForSequenceClassification:
    """
    Load a base XLM-R-like model and wrap it with LoRA adapters.

    num_labels=1 for binary toxicity.
    """
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )

    # LoRA config for XLM-R / RoBERTa-style attention
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    lora_model = get_peft_model(base_model, lora_config)
    return lora_model


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA-based fine-tuning for multilingual toxicity detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/lora_finetune.yaml)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional language override (e.g., es it). "
             "If omitted, use data.languages from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g., cuda, mps, cpu). If None, choose automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    processed_dir = get_processed_dir(cfg)
    languages = get_languages(cfg, args.languages)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    text_col = data_cfg.get("text_col", "comment_text")
    label_col = data_cfg.get("label_col", "toxic")
    base_model_name = model_cfg.get("base_model_name", "xlm-roberta-base")
    max_length = int(model_cfg.get("max_length", 256))

    output_root = Path(train_cfg.get("output_dir", "output/runs/lora"))
    batch_size = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("lr", 2e-5))
    num_epochs = int(train_cfg.get("epochs", 2))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    # ---------------- Device selection ----------------
    if args.device is not None:
        device_str = args.device
    else:
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
    device = torch.device(device_str)

    print(f"Using device: {device}")
    print(f"Using config: {config_path}")
    print(f"Processed data dir: {processed_dir}")
    print(f"Languages: {languages}")
    print(f"Base model: {base_model_name}")
    print(f"Output root: {output_root}")
    print(f"Text column: {text_col}, Label column: {label_col}")
    print(f"Max length: {max_length}, Batch size: {batch_size}, Epochs: {num_epochs}")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    for lang in languages:
        print("\n" + "=" * 70)
        print(f"[{lang}] Preparing data...")
        datasets = prepare_datasets_for_language(
            processed_dir=processed_dir,
            lang=lang,
            text_col=text_col,
            label_col=label_col,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        train_ds = datasets["train"]
        val_ds = datasets["val"]
        train_labels = datasets["train_labels"]

        pos_weight = compute_class_weights(train_labels)
        print(f"[{lang}] Positive class weight (for toxic=1): {pos_weight:.4f}")

        print(f"[{lang}] Building LoRA model...")
        model = build_lora_model(base_model_name)
        model.to(device)
        print(f"[{lang}] Model first param device: {next(model.parameters()).device}")

        # Print how many parameters are trainable (should be mostly LoRA)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{lang}] Trainable params: {trainable_params:,} / {total_params:,}")

        # Per-language output dir
        lang_out_dir = output_root / lang
        lang_out_dir.mkdir(parents=True, exist_ok=True)

        # Minimal TrainingArguments for broad version compatibility
        training_args = TrainingArguments(
            output_dir=str(lang_out_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=50,
            save_steps=500,
            do_eval=True,
        )

        # Create Trainer
        trainer = WeightedBCETrainer(
            pos_weight=pos_weight,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
        )

        print(f"[{lang}] Starting training...")
        trainer.train()
        print(f"[{lang}] Training complete.")

        # Save only the LoRA adapter weights
        save_path = lang_out_dir / "lora_adapter"
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[{lang}] Saving LoRA adapter to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("\nAll languages finished. LoRA adapters saved.")


if __name__ == "__main__":
    main()