# Loads the base model + LoRA adapter
# Runs inference on the **test** split
# Generates prediction CSVs
# code/lora_inference.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_lora_model(adapter_dir: Path, base_model_name: str, device: str):
    # Load base model
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )
    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device)
    model.eval()
    return model


def predict_texts(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    device: str,
) -> torch.Tensor:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits.view(-1)  # shape: (batch,)

    probs = torch.sigmoid(logits)  # toxicity probability
    return probs.cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a LoRA-finetuned XLM-R toxicity model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to lora_finetune.yaml",
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code to load (e.g., es, it, tr).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to classify.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to CSV file containing texts to classify.",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default=None,
        help="Column name in CSV with text (optional; defaults to data.text_col in config).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output/predictions/lora_predictions.csv",
        help="Where to save predictions if using --input_csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for toxicity label.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda or cpu. If None, picks automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(Path(args.config))
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    base_model_name = model_cfg.get("base_model_name", "xlm-roberta-base")
    max_length = int(model_cfg.get("max_length", 512))

    text_col_default = data_cfg.get("text_col", "comment_text")
    text_col = args.text_col or text_col_default

    output_root = Path(train_cfg.get("output_dir", "output/runs/lora"))
    adapter_dir = output_root / args.lang / "lora_adapter"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found at: {adapter_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading tokenizer and model from {base_model_name} + {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = load_lora_model(adapter_dir, base_model_name, device)

    # -------------------------
    # Mode 1: single text
    # -------------------------
    if args.text is not None:
        probs = predict_texts(model, tokenizer, [args.text], max_length, device)
        prob = float(probs[0].item())
        label = int(prob >= args.threshold)
        print(f"Text: {args.text}")
        print(f"Toxicity prob: {prob:.4f}, label@{args.threshold}: {label}")
        return

    # -------------------------
    # Mode 2: CSV file
    # -------------------------
    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)
        if text_col not in df.columns:
            raise KeyError(
                f"Column '{text_col}' not found in {args.input_csv}. "
                f"Available columns: {list(df.columns)}"
            )

        texts = df[text_col].astype(str).tolist()
        print(f"Loaded {len(texts)} rows from {args.input_csv}")

        probs = predict_texts(model, tokenizer, texts, max_length, device)
        df["toxicity_prob"] = probs.numpy()
        df["toxicity_label"] = (df["toxicity_prob"] >= args.threshold).astype(int)

        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
        return

    # If neither text nor CSV specified
    print("Nothing to do: provide either --text or --input_csv.")


if __name__ == "__main__":
    main()
