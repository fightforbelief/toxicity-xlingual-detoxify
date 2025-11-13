"""
Multilingual evaluation script for baseline models.

- Reads per-language prediction CSVs, e.g. output/predictions/baseline_en.csv
- Computes per-language metrics: ROC-AUC, F1, Precision, Recall, Accuracy, confusion-matrix stats
- Computes macro- and micro-averaged metrics across languages
- (Optional) Dumps FP/FN examples for each language to *{run_tag}_errors_{lang}.csv*
"""

from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

import warnings

warnings.filterwarnings("ignore")


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def infer_languages_from_config(cfg: Dict) -> List[str]:
    # Try to use config.data.languages if present
    try:
        langs = cfg["data"]["languages"]
        if isinstance(langs, list) and len(langs) > 0:
            return langs
    except Exception:
        pass
    return []


def infer_languages_from_files(pred_dir: Path, run_tag: str) -> List[str]:
    langs = []
    pattern = f"{run_tag}_"
    for f in pred_dir.glob(f"{run_tag}_*.csv"):
        name = f.stem  # e.g. baseline_en
        if name.startswith(pattern):
            lang = name[len(pattern):]
            if lang:
                langs.append(lang)
    return sorted(set(langs))


def select_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"None of the candidate columns {candidates} found in prediction file; "
            f"available columns: {list(df.columns)}"
        )
    return ""


def load_predictions(pred_file: Path, language: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load predictions from CSV file.

    Returns:
        y_true: shape (n_samples,)
        y_prob: shape (n_samples,)
        y_pred: shape (n_samples,)
        df:     original dataframe (for error dumping)
    """
    if not pred_file.exists():
        raise FileNotFoundError(f"[{language}] Prediction file not found: {pred_file}")

    df = pd.read_csv(pred_file)

    # Choose columns robustly
    label_col = select_column(df, ["y_true", "label", "target", "toxic"])
    prob_col = select_column(df, ["y_prob", "prob", "prediction", "pred_score", "pred"])
    pred_col = select_column(df, ["y_pred", "pred_label", "prediction_binary"], required=False)

    y_true = df[label_col].to_numpy()
    y_prob = df[prob_col].to_numpy()

    # If binary predictions are not present, threshold probabilities
    if pred_col:
        y_pred = df[pred_col].to_numpy()
    else:
        threshold = 0.5
        if "best_threshold" in df.columns:
            try:
                threshold = float(df["best_threshold"].iloc[0])
            except Exception:
                pass
        y_pred = (y_prob >= threshold).astype(int)

    # Ensure numeric
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.asarray(y_pred).astype(int)

    return y_true, y_prob, y_pred, df


def compute_language_metrics(
    language: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
) -> Dict:
    metrics: Dict[str, float] = {}
    n_samples = int(len(y_true))
    n_positive = int((y_true == 1).sum())
    n_negative = n_samples - n_positive

    metrics["language"] = language
    metrics["n_samples"] = n_samples
    metrics["positive_rate"] = n_positive / n_samples if n_samples > 0 else 0.0

    # ROC-AUC
    try:
        if n_positive > 0 and n_negative > 0:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["roc_auc"] = np.nan
    except Exception as e:
        print(f"Warning: Could not calculate ROC-AUC for {language}: {e}")
        metrics["roc_auc"] = np.nan

    # Binary classification metrics (thresholded)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Confusion matrix and derived stats
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    except Exception as e:
        print(f"Warning: Could not calculate confusion matrix for {language}: {e}")

    return metrics


def dump_error_cases(
    df: pd.DataFrame,
    language: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    errors_dir: Path,
    run_tag: str,
    max_per_type: int = 50,
) -> Path:
    """
    Save FP/FN examples for a given language.

    Output filename: {errors_dir}/{run_tag}_errors_{language}.csv
    """
    errors_dir.mkdir(parents=True, exist_ok=True)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    df_fp = df.loc[fp_mask].copy()
    df_fn = df.loc[fn_mask].copy()

    if len(df_fp) == 0 and len(df_fn) == 0:
        print(f"[{language}] No FP/FN samples; skip error dump.")
        return errors_dir / f"{run_tag}_errors_{language}.csv"

    # Tag error type
    df_fp["error_type"] = "FP"
    df_fn["error_type"] = "FN"

    # Subsample if too many
    if len(df_fp) > max_per_type:
        df_fp = df_fp.sample(max_per_type, random_state=0)
    if len(df_fn) > max_per_type:
        df_fn = df_fn.sample(max_per_type, random_state=0)

    df_errors = pd.concat([df_fp, df_fn], axis=0).reset_index(drop=True)

    out_path = errors_dir / f"{run_tag}_errors_{language}.csv"
    df_errors.to_csv(out_path, index=False)
    print(
        f"[{language}] Dumped error cases to {out_path} "
        f"(FP: {len(df_fp)}, FN: {len(df_fn)})"
    )
    return out_path


def add_aggregate_rows(
    all_metrics: List[Dict],
    all_true: Dict[str, np.ndarray],
    all_prob: Dict[str, np.ndarray],
    all_pred: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame(all_metrics)

    metric_cols = [
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "positive_rate",
        "specificity",
        "fpr",
        "fnr",
    ]

    # Macro-average: simple mean over languages
    macro_row: Dict[str, float] = {"language": "macro"}
    for col in metric_cols:
        if col in df.columns:
            macro_row[col] = df[col].mean()
    macro_row["n_samples"] = df["n_samples"].sum()
    df_macro = pd.DataFrame([macro_row])

    # Micro-average: recompute metrics on concatenated predictions
    all_y_true = np.concatenate([all_true[lang] for lang in all_true])
    all_y_prob = np.concatenate([all_prob[lang] for lang in all_prob])
    all_y_pred = np.concatenate([all_pred[lang] for lang in all_pred])
    n_samples = int(len(all_y_true))
    n_positive = int((all_y_true == 1).sum())
    n_negative = n_samples - n_positive

    micro_row: Dict[str, float] = {"language": "micro", "n_samples": n_samples}
    micro_row["positive_rate"] = n_positive / n_samples if n_samples > 0 else 0.0

    try:
        if n_positive > 0 and n_negative > 0:
            micro_row["roc_auc"] = roc_auc_score(all_y_true, all_y_prob)
        else:
            micro_row["roc_auc"] = np.nan
    except Exception:
        micro_row["roc_auc"] = np.nan

    micro_row["f1"] = f1_score(all_y_true, all_y_pred, zero_division=0)
    micro_row["precision"] = precision_score(all_y_true, all_y_pred, zero_division=0)
    micro_row["recall"] = recall_score(all_y_true, all_y_pred, zero_division=0)
    micro_row["accuracy"] = accuracy_score(all_y_true, all_y_pred)

    try:
        tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1]).ravel()
        micro_row["true_positives"] = int(tp)
        micro_row["true_negatives"] = int(tn)
        micro_row["false_positives"] = int(fp)
        micro_row["false_negatives"] = int(fn)
        micro_row["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        micro_row["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        micro_row["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    except Exception:
        pass

    df_micro = pd.DataFrame([micro_row])

    df_out = pd.concat([df, df_macro, df_micro], axis=0, ignore_index=True)
    return df_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multilingual baseline predictions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detoxify_multilingual.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="output/predictions",
        help="Directory containing prediction CSVs.",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="baseline",
        help="Prefix of prediction files, e.g. 'baseline' for baseline_en.csv.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Languages to evaluate, e.g. --languages en es it tr. "
            "If omitted, will try config.data.languages, then infer from files."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/predictions/baseline_metrics.csv",
        help="Path to output CSV for metrics.",
    )
    parser.add_argument(
        "--dump_errors",
        action="store_true",
        help="If set, dump FP/FN examples per language.",
    )
    parser.add_argument(
        "--errors_dir",
        type=str,
        default=None,
        help="Directory to write error CSVs. Default: same as pred_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    pred_dir = Path(args.pred_dir)
    output_path = Path(args.output)
    errors_dir = Path(args.errors_dir) if args.errors_dir is not None else pred_dir

    cfg = load_config(config_path)

    if args.languages is not None and len(args.languages) > 0:
        languages = args.languages
    else:
        languages = infer_languages_from_config(cfg)
        if not languages:
            languages = infer_languages_from_files(pred_dir, args.run_tag)

    if not languages:
        raise ValueError(
            "No languages specified and could not infer from config or files. "
            "Please pass --languages explicitly."
        )

    print(f"Evaluating run_tag='{args.run_tag}' for languages: {languages}")
    all_metrics: List[Dict] = []
    all_true: Dict[str, np.ndarray] = {}
    all_prob: Dict[str, np.ndarray] = {}
    all_pred: Dict[str, np.ndarray] = {}

    for lang in languages:
        pred_file = pred_dir / f"{args.run_tag}_{lang}.csv"
        print(f"[{lang}] Loading predictions from {pred_file}")
        y_true, y_prob, y_pred, df = load_predictions(pred_file, lang)

        metrics = compute_language_metrics(lang, y_true, y_prob, y_pred)
        all_metrics.append(metrics)
        all_true[lang] = y_true
        all_prob[lang] = y_prob
        all_pred[lang] = y_pred

        if args.dump_errors:
            dump_error_cases(df, lang, y_true, y_pred, errors_dir, args.run_tag)

    # Aggregate macro/micro
    df_out = add_aggregate_rows(all_metrics, all_true, all_prob, all_pred)

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nSaved metrics to {output_path}")
    print(df_out)


if __name__ == "__main__":
    main()
