"""
Error Analysis Tool for Toxicity Detection Models

Analyzes prediction errors and provides insights into model weaknesses:
- Confusion matrix and error breakdown
- Most confident errors
- Error patterns by text characteristics
- Comparison between baseline and hard sample mining models

Usage:
    python error_analysis.py \
        --baseline_pred output/predictions/detoxify_es.csv \
        --hard_sample_pred output/predictions/hard_sample_es.csv \
        --output_dir output/analysis/es
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)

warnings.filterwarnings("ignore")


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load predictions from CSV."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    
    # Standardize column names
    if 'y_true' in df.columns:
        df['true_label'] = df['y_true']
    elif 'toxic' in df.columns:
        df['true_label'] = df['toxic']
    
    if 'y_prob' in df.columns:
        df['pred_prob'] = df['y_prob']
    
    if 'y_pred' in df.columns:
        df['pred_label'] = df['y_pred']
    else:
        df['pred_label'] = (df['pred_prob'] >= 0.5).astype(int)
    
    return df


def compute_error_types(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Categorize errors into FP, FN, TP, TN."""
    fp = df[(df['true_label'] == 0) & (df['pred_label'] == 1)].copy()
    fn = df[(df['true_label'] == 1) & (df['pred_label'] == 0)].copy()
    tp = df[(df['true_label'] == 1) & (df['pred_label'] == 1)].copy()
    tn = df[(df['true_label'] == 0) & (df['pred_label'] == 0)].copy()
    
    return {
        'false_positive': fp,
        'false_negative': fn,
        'true_positive': tp,
        'true_negative': tn
    }


def print_error_summary(errors: Dict[str, pd.DataFrame], model_name: str = "Model"):
    """Print summary of errors."""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} ERROR SUMMARY")
    print(f"{'='*70}")
    
    total = sum(len(df) for df in errors.values())
    
    for error_type, df in errors.items():
        count = len(df)
        pct = count / total * 100 if total > 0 else 0
        print(f"{error_type:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Accuracy
    correct = len(errors['true_positive']) + len(errors['true_negative'])
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{'Accuracy':20s}: {correct:5d} ({accuracy:5.1f}%)")


def analyze_confidence(df: pd.DataFrame, text_col: str = 'comment_text') -> pd.DataFrame:
    """Analyze prediction confidence."""
    df = df.copy()
    
    # Compute confidence (distance from 0.5)
    df['confidence'] = np.abs(df['pred_prob'] - 0.5)
    
    # Correct prediction
    df['is_correct'] = (df['true_label'] == df['pred_label']).astype(int)
    
    return df


def find_most_confident_errors(
    df: pd.DataFrame,
    n: int = 20,
    text_col: str = 'comment_text'
) -> pd.DataFrame:
    """Find most confident errors (model is wrong but very confident)."""
    # Only errors
    errors = df[df['true_label'] != df['pred_label']].copy()
    
    if len(errors) == 0:
        print("No errors found!")
        return pd.DataFrame()
    
    # Sort by confidence
    errors['confidence'] = np.abs(errors['pred_prob'] - 0.5)
    errors = errors.nlargest(n, 'confidence')
    
    return errors


def compare_models(
    baseline_df: pd.DataFrame,
    hard_sample_df: pd.DataFrame,
    text_col: str = 'comment_text'
) -> Dict[str, pd.DataFrame]:
    """Compare baseline and hard sample mining models."""
    
    # Merge on text
    merged = baseline_df.merge(
        hard_sample_df,
        on=text_col,
        suffixes=('_baseline', '_hard_sample')
    )
    
    # Find cases where models differ
    both_correct = merged[
        (merged['pred_label_baseline'] == merged['true_label_baseline']) &
        (merged['pred_label_hard_sample'] == merged['true_label_hard_sample'])
    ]
    
    both_wrong = merged[
        (merged['pred_label_baseline'] != merged['true_label_baseline']) &
        (merged['pred_label_hard_sample'] != merged['true_label_hard_sample'])
    ]
    
    baseline_correct_only = merged[
        (merged['pred_label_baseline'] == merged['true_label_baseline']) &
        (merged['pred_label_hard_sample'] != merged['true_label_hard_sample'])
    ]
    
    hard_sample_correct_only = merged[
        (merged['pred_label_baseline'] != merged['true_label_baseline']) &
        (merged['pred_label_hard_sample'] == merged['true_label_hard_sample'])
    ]
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Both correct:           {len(both_correct):5d} ({len(both_correct)/len(merged)*100:5.1f}%)")
    print(f"Both wrong:             {len(both_wrong):5d} ({len(both_wrong)/len(merged)*100:5.1f}%)")
    print(f"Baseline correct only:  {len(baseline_correct_only):5d} ({len(baseline_correct_only)/len(merged)*100:5.1f}%)")
    print(f"Hard sample correct only: {len(hard_sample_correct_only):5d} ({len(hard_sample_correct_only)/len(merged)*100:5.1f}%)")
    
    net_improvement = len(hard_sample_correct_only) - len(baseline_correct_only)
    print(f"\n🎯 Net improvement: {net_improvement:+d} samples")
    
    return {
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'baseline_correct_only': baseline_correct_only,
        'hard_sample_correct_only': hard_sample_correct_only,
        'merged': merged
    }


def plot_confusion_matrices(
    baseline_df: pd.DataFrame,
    hard_sample_df: Optional[pd.DataFrame],
    output_dir: Path
):
    """Plot confusion matrices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_models = 2 if hard_sample_df is not None else 1
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # Baseline
    cm_baseline = confusion_matrix(baseline_df['true_label'], baseline_df['pred_label'])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Non-toxic', 'Toxic'],
                yticklabels=['Non-toxic', 'Toxic'])
    axes[0].set_title('Baseline Model')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Hard sample (if available)
    if hard_sample_df is not None:
        cm_hard = confusion_matrix(hard_sample_df['true_label'], hard_sample_df['pred_label'])
        sns.heatmap(cm_hard, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                    xticklabels=['Non-toxic', 'Toxic'],
                    yticklabels=['Non-toxic', 'Toxic'])
        axes[1].set_title('Hard Sample Mining Model')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plot_path = output_dir / 'confusion_matrices.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved confusion matrices to: {plot_path}")
    plt.close()


def plot_roc_curves(
    baseline_df: pd.DataFrame,
    hard_sample_df: Optional[pd.DataFrame],
    output_dir: Path
):
    """Plot ROC curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Baseline ROC
    fpr_base, tpr_base, _ = roc_curve(baseline_df['true_label'], baseline_df['pred_prob'])
    roc_auc_base = roc_auc_score(baseline_df['true_label'], baseline_df['pred_prob'])
    ax.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {roc_auc_base:.4f})', linewidth=2)
    
    # Hard sample ROC (if available)
    if hard_sample_df is not None:
        fpr_hard, tpr_hard, _ = roc_curve(hard_sample_df['true_label'], hard_sample_df['pred_prob'])
        roc_auc_hard = roc_auc_score(hard_sample_df['true_label'], hard_sample_df['pred_prob'])
        ax.plot(fpr_hard, tpr_hard, label=f'Hard Sample Mining (AUC = {roc_auc_hard:.4f})', linewidth=2)
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'roc_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Saved ROC curves to: {plot_path}")
    plt.close()


def plot_precision_recall_curves(
    baseline_df: pd.DataFrame,
    hard_sample_df: Optional[pd.DataFrame],
    output_dir: Path
):
    """Plot Precision-Recall curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Baseline PR
    precision_base, recall_base, _ = precision_recall_curve(
        baseline_df['true_label'], baseline_df['pred_prob']
    )
    ax.plot(recall_base, precision_base, label='Baseline', linewidth=2)
    
    # Hard sample PR (if available)
    if hard_sample_df is not None:
        precision_hard, recall_hard, _ = precision_recall_curve(
            hard_sample_df['true_label'], hard_sample_df['pred_prob']
        )
        ax.plot(recall_hard, precision_hard, label='Hard Sample Mining', linewidth=2)
    
    # Baseline (random classifier)
    baseline_ratio = baseline_df['true_label'].mean()
    ax.axhline(baseline_ratio, color='k', linestyle='--', 
               label=f'Random ({baseline_ratio:.2f})', linewidth=1)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'precision_recall_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Saved PR curves to: {plot_path}")
    plt.close()


def save_error_examples(
    errors: Dict[str, pd.DataFrame],
    output_dir: Path,
    text_col: str = 'comment_text',
    n: int = 50
):
    """Save example errors to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for error_type, df in errors.items():
        if len(df) == 0:
            continue
        
        # Take top N by confidence
        if 'confidence' not in df.columns:
            df = df.copy()
            df['confidence'] = np.abs(df['pred_prob'] - 0.5)
        
        df_sorted = df.nlargest(min(n, len(df)), 'confidence')
        
        # Save
        output_path = output_dir / f'{error_type}_examples.csv'
        cols_to_save = [text_col, 'true_label', 'pred_label', 'pred_prob', 'confidence']
        cols_to_save = [c for c in cols_to_save if c in df_sorted.columns]
        df_sorted[cols_to_save].to_csv(output_path, index=False)
        
        print(f"💾 Saved {len(df_sorted)} {error_type} examples to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Error analysis for toxicity detection models"
    )
    parser.add_argument(
        "--baseline_pred",
        type=str,
        required=True,
        help="Path to baseline predictions CSV"
    )
    parser.add_argument(
        "--hard_sample_pred",
        type=str,
        default=None,
        help="Path to hard sample mining predictions CSV (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="comment_text",
        help="Name of text column"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=50,
        help="Number of error examples to save"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Load predictions
    print(f"\n📂 Loading baseline predictions from {args.baseline_pred}...")
    baseline_df = load_predictions(Path(args.baseline_pred))
    print(f"  Samples: {len(baseline_df)}")
    
    hard_sample_df = None
    if args.hard_sample_pred:
        print(f"\n📂 Loading hard sample predictions from {args.hard_sample_pred}...")
        hard_sample_df = load_predictions(Path(args.hard_sample_pred))
        print(f"  Samples: {len(hard_sample_df)}")
    
    output_dir = Path(args.output_dir)
    
    # Analyze baseline
    baseline_analyzed = analyze_confidence(baseline_df, args.text_col)
    baseline_errors = compute_error_types(baseline_analyzed)
    print_error_summary(baseline_errors, "Baseline")
    
    # Most confident errors
    print(f"\n🎯 Most Confident Baseline Errors:")
    conf_errors = find_most_confident_errors(baseline_analyzed, n=10, text_col=args.text_col)
    for i, (idx, row) in enumerate(conf_errors.iterrows(), 1):
        text = row.get(args.text_col, '')[:80] + '...' if len(row.get(args.text_col, '')) > 80 else row.get(args.text_col, '')
        print(f"\n{i}. Confidence: {row['confidence']:.3f} | True: {int(row['true_label'])} | Pred: {row['pred_prob']:.3f}")
        print(f"   {text}")
    
    # Analyze hard sample model (if available)
    if hard_sample_df is not None:
        hard_sample_analyzed = analyze_confidence(hard_sample_df, args.text_col)
        hard_sample_errors = compute_error_types(hard_sample_analyzed)
        print_error_summary(hard_sample_errors, "Hard Sample Mining")
        
        # Compare models
        comparison = compare_models(baseline_analyzed, hard_sample_analyzed, args.text_col)
        
        # Save improved examples
        if len(comparison['hard_sample_correct_only']) > 0:
            improved_path = output_dir / 'improved_by_hard_sample.csv'
            output_dir.mkdir(parents=True, exist_ok=True)
            comparison['hard_sample_correct_only'].to_csv(improved_path, index=False)
            print(f"\n💾 Saved improved examples to: {improved_path}")
    else:
        hard_sample_errors = None
    
    # Generate plots
    print(f"\n📊 Generating visualizations...")
    plot_confusion_matrices(baseline_analyzed, hard_sample_analyzed if hard_sample_df else None, output_dir)
    plot_roc_curves(baseline_analyzed, hard_sample_analyzed if hard_sample_df else None, output_dir)
    plot_precision_recall_curves(baseline_analyzed, hard_sample_analyzed if hard_sample_df else None, output_dir)
    
    # Save error examples
    print(f"\n💾 Saving error examples...")
    save_error_examples(baseline_errors, output_dir / 'baseline', args.text_col, args.n_examples)
    if hard_sample_errors:
        save_error_examples(hard_sample_errors, output_dir / 'hard_sample', args.text_col, args.n_examples)
    
    print(f"\n{'='*70}")
    print("✅ ERROR ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()


