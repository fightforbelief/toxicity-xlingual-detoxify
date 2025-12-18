"""
Identify Hard Samples for Active Learning / Hard Sample Mining

This script analyzes baseline model predictions and identifies:
1. Uncertain samples: predictions close to decision boundary (prob ~0.5)
2. Misclassified samples: false positives and false negatives
3. High-confidence errors: model is very confident but wrong

These hard samples will be weighted more heavily during training.

Usage:
    python identify_hard_samples.py \
        --baseline_pred output/predictions/detoxify_es.csv \
        --train_data data/processed/es/train.csv \
        --val_data data/processed/es/val.csv \
        --output data/processed/es/hard_samples_weights.csv \
        --uncertainty_threshold 0.15 \
        --confidence_threshold 0.8
"""

from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_baseline_predictions(pred_path: Path) -> pd.DataFrame:
    """Load baseline predictions."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    
    # Standardize column names
    if 'y_true' in df.columns:
        df['true_label'] = df['y_true']
    elif 'toxic' in df.columns:
        df['true_label'] = df['toxic']
    
    if 'y_prob' in df.columns:
        df['pred_prob'] = df['y_prob']
    elif 'toxicity' in df.columns:
        df['pred_prob'] = df['toxicity']
    
    if 'y_pred' in df.columns:
        df['pred_label'] = df['y_pred']
    else:
        # Threshold at 0.5 if not provided
        df['pred_label'] = (df['pred_prob'] >= 0.5).astype(int)
    
    return df


def identify_uncertain_samples(df: pd.DataFrame, threshold: float = 0.15) -> pd.Series:
    """
    Identify samples where model is uncertain (prob close to 0.5).
    
    Args:
        df: DataFrame with pred_prob column
        threshold: Distance from 0.5 to be considered uncertain
                  e.g., threshold=0.15 means 0.35 < prob < 0.65
    
    Returns:
        Boolean mask of uncertain samples
    """
    uncertainty = np.abs(df['pred_prob'] - 0.5)
    is_uncertain = uncertainty <= threshold
    
    print(f"\n📊 Uncertain Samples (|prob - 0.5| <= {threshold}):")
    print(f"  Count: {is_uncertain.sum()} / {len(df)} ({is_uncertain.sum()/len(df)*100:.2f}%)")
    
    return is_uncertain


def identify_misclassified_samples(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Identify false positives and false negatives.
    
    Returns:
        Dictionary with 'fp', 'fn', 'tp', 'tn' boolean masks
    """
    fp = (df['true_label'] == 0) & (df['pred_label'] == 1)
    fn = (df['true_label'] == 1) & (df['pred_label'] == 0)
    tp = (df['true_label'] == 1) & (df['pred_label'] == 1)
    tn = (df['true_label'] == 0) & (df['pred_label'] == 0)
    
    print(f"\n❌ Misclassified Samples:")
    print(f"  False Positives: {fp.sum()} ({fp.sum()/len(df)*100:.2f}%)")
    print(f"  False Negatives: {fn.sum()} ({fn.sum()/len(df)*100:.2f}%)")
    print(f"  True Positives:  {tp.sum()} ({tp.sum()/len(df)*100:.2f}%)")
    print(f"  True Negatives:  {tn.sum()} ({tn.sum()/len(df)*100:.2f}%)")
    
    return {'fp': fp, 'fn': fn, 'tp': tp, 'tn': tn}


def identify_high_confidence_errors(df: pd.DataFrame, threshold: float = 0.8) -> pd.Series:
    """
    Identify samples where model is confident but wrong.
    These are particularly valuable for learning.
    
    Args:
        df: DataFrame with predictions
        threshold: Confidence threshold (prob > threshold or prob < 1-threshold)
    
    Returns:
        Boolean mask of high-confidence errors
    """
    is_confident = (df['pred_prob'] >= threshold) | (df['pred_prob'] <= (1 - threshold))
    is_wrong = df['true_label'] != df['pred_label']
    high_conf_error = is_confident & is_wrong
    
    print(f"\n🎯 High-Confidence Errors (confidence >= {threshold}):")
    print(f"  Count: {high_conf_error.sum()} / {len(df)} ({high_conf_error.sum()/len(df)*100:.2f}%)")
    
    return high_conf_error


def compute_sample_weights(
    df: pd.DataFrame,
    is_uncertain: pd.Series,
    classification: Dict[str, pd.Series],
    is_high_conf_error: pd.Series,
    config: Dict
) -> pd.Series:
    """
    Compute training weights for each sample based on difficulty.
    
    Weight strategy:
    - Base weight: 1.0
    - Uncertain samples: * uncertain_weight
    - Misclassified samples: * misclassified_weight
    - High-confidence errors: * high_conf_error_weight
    - Toxic minority class: * minority_weight (optional)
    """
    weights = pd.Series(1.0, index=df.index)
    
    # Apply weights
    weights[is_uncertain] *= config['uncertain_weight']
    weights[classification['fp'] | classification['fn']] *= config['misclassified_weight']
    weights[is_high_conf_error] *= config['high_conf_error_weight']
    
    # Optional: boost minority class (toxic)
    if config.get('boost_minority', False):
        is_toxic = df['true_label'] == 1
        weights[is_toxic] *= config['minority_weight']
    
    # Normalize weights to have mean = 1.0 (optional)
    if config.get('normalize_weights', True):
        weights = weights / weights.mean()
    
    print(f"\n⚖️  Sample Weights Statistics:")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std:  {weights.std():.3f}")
    print(f"  Min:  {weights.min():.3f}")
    print(f"  Max:  {weights.max():.3f}")
    print(f"  Median: {weights.median():.3f}")
    
    return weights


def analyze_hard_samples(df: pd.DataFrame, weights: pd.Series, text_col: str = 'comment_text'):
    """Print analysis of hard samples."""
    print(f"\n" + "="*70)
    print("HARD SAMPLE ANALYSIS")
    print("="*70)
    
    # Top 10 hardest samples
    hardest_indices = weights.nlargest(10).index
    print(f"\n🔥 Top 10 Hardest Samples (highest weights):")
    for i, idx in enumerate(hardest_indices, 1):
        row = df.loc[idx]
        text = row.get(text_col, '')[:100] + '...' if len(row.get(text_col, '')) > 100 else row.get(text_col, '')
        print(f"\n{i}. Weight: {weights[idx]:.2f} | True: {int(row['true_label'])} | "
              f"Pred: {row['pred_prob']:.3f}")
        print(f"   Text: {text}")
    
    # Weight distribution by true label
    print(f"\n📊 Average Weight by Class:")
    for label in [0, 1]:
        mask = df['true_label'] == label
        avg_weight = weights[mask].mean()
        label_name = "Toxic" if label == 1 else "Non-toxic"
        print(f"  {label_name}: {avg_weight:.3f} (n={mask.sum()})")


def match_weights_to_train_data(
    baseline_pred_df: pd.DataFrame,
    train_df: pd.DataFrame,
    weights: pd.Series,
    text_col: str = 'comment_text'
) -> pd.DataFrame:
    """
    Match computed weights back to training data.
    
    Note: Baseline predictions might be on val/test set, so we need to match by text.
    If predictions are on train set, we can match by index.
    """
    # Try to match by text content
    train_df = train_df.copy()
    train_df['sample_weight'] = 1.0  # Default weight
    
    # Create a mapping from text to weight
    weight_map = dict(zip(baseline_pred_df[text_col], weights))
    
    # Match weights
    matched = 0
    for idx, row in train_df.iterrows():
        text = row[text_col]
        if text in weight_map:
            train_df.loc[idx, 'sample_weight'] = weight_map[text]
            matched += 1
    
    print(f"\n🔗 Matched {matched} / {len(train_df)} training samples to baseline predictions")
    
    if matched < len(train_df) * 0.5:
        print(f"⚠️  Warning: Less than 50% of training samples matched!")
        print(f"   This might happen if baseline predictions are on val/test set.")
        print(f"   Consider running baseline on training set for better matching.")
    
    return train_df


def save_weighted_data(
    df: pd.DataFrame,
    output_path: Path,
    text_col: str = 'comment_text',
    label_col: str = 'toxic'
):
    """Save training data with sample weights."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save essential columns
    cols_to_save = [text_col, label_col, 'sample_weight']
    if 'lang' in df.columns:
        cols_to_save.append('lang')
    
    df_save = df[cols_to_save].copy()
    df_save.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved weighted training data to: {output_path}")
    print(f"   Columns: {list(df_save.columns)}")
    print(f"   Rows: {len(df_save)}")


def plot_weight_distribution(weights: pd.Series, output_dir: Path):
    """Plot distribution of sample weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(weights, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.2f}')
    axes[0].axvline(weights.median(), color='green', linestyle='--', label=f'Median: {weights.median():.2f}')
    axes[0].set_xlabel('Sample Weight')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sample Weights')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    axes[1].boxplot(weights, vert=True)
    axes[1].set_ylabel('Sample Weight')
    axes[1].set_title('Sample Weight Box Plot')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'sample_weight_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Saved weight distribution plot to: {plot_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify hard samples for active learning"
    )
    parser.add_argument(
        "--baseline_pred",
        type=str,
        required=True,
        help="Path to baseline prediction CSV (from strong baseline or detoxify)"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="Path to validation data CSV (optional, for additional analysis)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for weighted training data"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="comment_text",
        help="Name of text column"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="toxic",
        help="Name of label column"
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.15,
        help="Threshold for uncertain samples (default: 0.15)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="Threshold for high-confidence errors (default: 0.8)"
    )
    parser.add_argument(
        "--uncertain_weight",
        type=float,
        default=2.0,
        help="Weight multiplier for uncertain samples (default: 2.0)"
    )
    parser.add_argument(
        "--misclassified_weight",
        type=float,
        default=3.0,
        help="Weight multiplier for misclassified samples (default: 3.0)"
    )
    parser.add_argument(
        "--high_conf_error_weight",
        type=float,
        default=5.0,
        help="Weight multiplier for high-confidence errors (default: 5.0)"
    )
    parser.add_argument(
        "--minority_weight",
        type=float,
        default=1.5,
        help="Weight multiplier for minority class (toxic) (default: 1.5)"
    )
    parser.add_argument(
        "--boost_minority",
        action="store_true",
        help="Whether to boost minority class weight"
    )
    parser.add_argument(
        "--normalize_weights",
        action="store_true",
        default=True,
        help="Normalize weights to mean=1.0"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("HARD SAMPLE IDENTIFICATION")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading data...")
    baseline_pred_df = load_baseline_predictions(Path(args.baseline_pred))
    train_df = pd.read_csv(args.train_data)
    
    print(f"  Baseline predictions: {len(baseline_pred_df)} samples")
    print(f"  Training data: {len(train_df)} samples")
    
    # Identify hard samples
    is_uncertain = identify_uncertain_samples(
        baseline_pred_df, 
        threshold=args.uncertainty_threshold
    )
    
    classification = identify_misclassified_samples(baseline_pred_df)
    
    is_high_conf_error = identify_high_confidence_errors(
        baseline_pred_df,
        threshold=args.confidence_threshold
    )
    
    # Compute weights
    config = {
        'uncertain_weight': args.uncertain_weight,
        'misclassified_weight': args.misclassified_weight,
        'high_conf_error_weight': args.high_conf_error_weight,
        'minority_weight': args.minority_weight,
        'boost_minority': args.boost_minority,
        'normalize_weights': args.normalize_weights,
    }
    
    weights = compute_sample_weights(
        baseline_pred_df,
        is_uncertain,
        classification,
        is_high_conf_error,
        config
    )
    
    # Analyze hard samples
    analyze_hard_samples(baseline_pred_df, weights, text_col=args.text_col)
    
    # Match weights to training data
    train_df_weighted = match_weights_to_train_data(
        baseline_pred_df,
        train_df,
        weights,
        text_col=args.text_col
    )
    
    # Save
    save_weighted_data(
        train_df_weighted,
        Path(args.output),
        text_col=args.text_col,
        label_col=args.label_col
    )
    
    # Plot
    if args.plot:
        output_dir = Path(args.output).parent
        plot_weight_distribution(train_df_weighted['sample_weight'], output_dir)
    
    print(f"\n" + "="*70)
    print("✅ HARD SAMPLE IDENTIFICATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

