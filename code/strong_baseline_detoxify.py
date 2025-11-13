"""
Strong baseline using Detoxify's multilingual XLM-R model.

- Loads per-language train/val/test splits from data/processed/{lang}
- Uses Detoxify('multilingual') to obtain toxicity scores
- Tunes a decision threshold on val (maximize F1)
- Writes per-language prediction CSVs to output/predictions/{run_tag}_{lang}.csv
  with columns: id,y_true,y_prob,y_pred,best_threshold

Example:

    python code/strong_baseline_detoxify.py \
        --config configs/detoxify_multilingual.yaml \
        --languages en es it tr \
        --run_tag detoxify \
        --batch_size 32

Requirements:
    pip install detoxify transformers torch
"""

from pathlib import Path
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

try:
    from detoxify import Detoxify
except ImportError as e:
    raise ImportError(
        "Could not import 'detoxify'. Please install it with:\n"
        "    pip install detoxify\n"
        "and ensure 'torch' is installed with GPU support if available."
    ) from e


# ----------------------------------------------------------------------
# I/O utilities
# ----------------------------------------------------------------------

def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_languages(cfg: Dict, arg_langs: Optional[List[str]]) -> List[str]:
    if arg_langs:
        return arg_langs
    try:
        langs = cfg["data"]["languages"]
        if isinstance(langs, list) and langs:
            return langs
    except KeyError:
        pass
    raise ValueError("No languages specified. "
                     "Provide --languages or config.data.languages in YAML.")


def get_processed_dir(cfg: Dict) -> Path:
    try:
        processed = cfg["data"]["processed_dir"]
    except KeyError:
        raise KeyError("Config is missing data.processed_dir.")
    return Path(processed)


def load_split(processed_dir: Path,
               lang: str,
               split: str,
               text_col: str,
               label_col: str) -> pd.DataFrame:
    path = processed_dir / lang / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise KeyError(f"Expected text column '{text_col}' in {path}, "
                       f"found columns: {list(df.columns)}")

    if label_col not in df.columns:
        # For some test sets you may not have labels; we allow this and fill -1
        df[label_col] = -1

    # Ensure an id column
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    return df


# ----------------------------------------------------------------------
# Detoxify inference
# ----------------------------------------------------------------------

def build_detoxify_model(device: Optional[str] = None) -> Detoxify:
    """
    Build the multilingual Detoxify model.

    model_name: 'multilingual' is the XLM-R variant supporting many languages.
    """
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    # 'multilingual' uses XLM-R under the hood
    model = Detoxify("multilingual", **kwargs)
    return model


def predict_toxicity(model: Detoxify,
                     texts: List[str],
                     batch_size: int = 32) -> np.ndarray:
    """
    Run Detoxify on a list of texts and return an array of toxicity probabilities.
    """
    probs: List[float] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]

        # Detoxify returns a dict of arrays/lists
        outputs = model.predict(batch)
        # We care about the 'toxicity' score
        batch_probs = outputs["toxicity"]

        # Make sure we always have a list-like
        if np.isscalar(batch_probs):
            probs.append(float(batch_probs))
        else:
            probs.extend([float(p) for p in batch_probs])

    return np.asarray(probs, dtype=float)


# ----------------------------------------------------------------------
# Threshold tuning and evaluation
# ----------------------------------------------------------------------

def tune_threshold_f1(y_true: np.ndarray,
                      y_prob: np.ndarray,
                      n_steps: int = 81,
                      min_thr: float = 0.1,
                      max_thr: float = 0.9) -> float:
    """
    Search over thresholds in [min_thr, max_thr] to maximize F1 on validation.
    """
    if y_true.ndim != 1:
        y_true = y_true.ravel()

    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(min_thr, max_thr, n_steps):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    # Some test splits may have dummy labels of -1; skip metrics in that case
    if np.all((y_true == -1) | (np.isnan(y_true))):
        return metrics

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    return metrics


# ----------------------------------------------------------------------
# Main per-language pipeline
# ----------------------------------------------------------------------

def run_for_language(
    model: Detoxify,
    processed_dir: Path,
    lang: str,
    text_col: str,
    label_col: str,
    batch_size: int,
    run_tag: str,
    output_dir: Path,
    global_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Run Detoxify strong baseline for a single language:
    - Load val & test splits
    - Tune threshold on val (if global_threshold is None)
    - Predict on test
    - Save prediction CSV
    - Return test metrics (if labels available)
    """
    print(f"\n{'=' * 70}")
    print(f"[{lang}] Loading data...")
    val_df = load_split(processed_dir, lang, "val", text_col, label_col)
    test_df = load_split(processed_dir, lang, "test", text_col, label_col)

    # Validation predictions for threshold tuning
    print(f"[{lang}] Predicting on validation set (n={len(val_df)})...")
    val_probs = predict_toxicity(model, val_df[text_col].tolist(), batch_size=batch_size)
    val_true = val_df[label_col].astype(int).values

    if global_threshold is None:
        print(f"[{lang}] Tuning threshold on validation (maximize F1)...")
        best_thr = tune_threshold_f1(val_true, val_probs)
    else:
        best_thr = float(global_threshold)

    print(f"[{lang}] Using threshold = {best_thr:.4f}")

    # Test predictions
    print(f"[{lang}] Predicting on test set (n={len(test_df)})...")
    test_probs = predict_toxicity(model, test_df[text_col].tolist(), batch_size=batch_size)
    test_true = test_df[label_col].astype(int).values
    test_pred = (test_probs >= best_thr).astype(int)

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{run_tag}_{lang}.csv"

    out_df = pd.DataFrame({
        "id": test_df["id"],
        "y_true": test_true,
        "y_prob": test_probs,
        "y_pred": test_pred,
        "best_threshold": best_thr,
    })
    out_df.to_csv(out_path, index=False)
    print(f"[{lang}] Saved predictions to: {out_path}")

    # Compute metrics on test (if labels are meaningful)
    metrics = compute_metrics(test_true, test_probs, test_pred)
    if metrics:
        print(f"[{lang}] Test metrics:")
        print(
            f"    ROC-AUC = {metrics['roc_auc']:.4f}, "
            f"F1 = {metrics['f1']:.4f}, "
            f"Precision = {metrics['precision']:.4f}, "
            f"Recall = {metrics['recall']:.4f}, "
            f"Accuracy = {metrics['accuracy']:.4f}"
        )
    else:
        print(f"[{lang}] Test labels not available or invalid; skipping metric computation.")

    return metrics


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strong baseline using Detoxify multilingual XLM-R for toxicity detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/detoxify_multilingual.yaml)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to evaluate (e.g., en es it tr). "
             "If omitted, uses config.data.languages.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="detoxify",
        help="Tag used in output filenames (default: detoxify).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/predictions",
        help="Directory to save prediction CSVs (default: output/predictions).",
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="comment_text",
        help="Name of the text column in processed CSVs (default: comment_text).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="toxic",
        help="Name of the binary label column (default: toxic).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for Detoxify inference (default: 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Detoxify model (e.g., cuda, cpu). "
             "If None, Detoxify will pick automatically.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional global threshold. If not set, tune per-language on val.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    processed_dir = get_processed_dir(cfg)
    languages = get_languages(cfg, args.languages)
    output_dir = Path(args.output_dir)

    print(f"Using config: {config_path}")
    print(f"Processed data dir: {processed_dir}")
    print(f"Languages: {languages}")
    print(f"Output dir: {output_dir}")
    print(f"Text column: {args.text_col}, Label column: {args.label_col}")
    print(f"Batch size: {args.batch_size}, Device: {args.device}")
    if args.threshold is not None:
        print(f"Using global threshold: {args.threshold}")
    else:
        print("Threshold will be tuned on validation per language.")

    print("\nBuilding Detoxify multilingual model...")
    model = build_detoxify_model(device=args.device)
    print("Model ready.")

    all_metrics: Dict[str, Dict[str, float]] = {}
    for lang in languages:
        metrics = run_for_language(
            model=model,
            processed_dir=processed_dir,
            lang=lang,
            text_col=args.text_col,
            label_col=args.label_col,
            batch_size=args.batch_size,
            run_tag=args.run_tag,
            output_dir=output_dir,
            global_threshold=args.threshold,
        )
        all_metrics[lang] = metrics

    # Optionally, aggregate macro averages here (print only)
    valid_langs = [l for l, m in all_metrics.items() if m]
    if valid_langs:
        print("\n" + "=" * 70)
        print("Macro-averaged metrics over languages with valid labels:")
        macro = {}
        for key in ["roc_auc", "f1", "precision", "recall", "accuracy"]:
            vals = [all_metrics[l][key] for l in valid_langs if key in all_metrics[l]]
            if vals:
                macro[key] = float(np.mean(vals))
        for k, v in macro.items():
            print(f"  {k}: {v:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
