"""
Generate Baseline Predictions on Training Set

This script runs the baseline Detoxify model on the training set to generate
predictions that can be used for hard sample identification.

Why we need this:
- Hard sample mining requires baseline predictions on the SAME data being trained
- The original baseline predictions were on test set, not training set
- We need predictions with text content to match samples

Usage:
    python generate_train_baseline_predictions.py --languages es it tr
    
    # Or specify custom paths
    python generate_train_baseline_predictions.py \
        --languages es \
        --data_dir ../milestone3/data/processed \
        --output_dir ../milestone2/output/predictions
"""

import argparse
from pathlib import Path
from typing import List
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from detoxify import Detoxify
except ImportError:
    print("❌ Detoxify not installed!")
    print("Install with: pip install detoxify")
    exit(1)


def generate_predictions_for_language(
    lang: str,
    data_dir: Path,
    output_dir: Path,
    text_col: str = 'comment_text',
    label_col: str = 'toxic',
    batch_size: int = 32,
    model_type: str = 'multilingual'
):
    """
    Generate baseline predictions for a single language on training set.
    
    Args:
        lang: Language code (e.g., 'es', 'it', 'tr')
        data_dir: Directory containing processed data
        output_dir: Directory to save predictions
        text_col: Name of text column
        label_col: Name of label column
        batch_size: Batch size for prediction
        model_type: Detoxify model type ('multilingual' recommended)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {lang.upper()}")
    print(f"{'='*70}")
    
    # Load training data
    train_path = data_dir / lang / 'train.csv'
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    
    print(f"\n📂 Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    
    if text_col not in train_df.columns:
        raise KeyError(f"Text column '{text_col}' not found. Available: {list(train_df.columns)}")
    
    if label_col not in train_df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Available: {list(train_df.columns)}")
    
    n_samples = len(train_df)
    n_toxic = train_df[label_col].sum()
    print(f"  Samples: {n_samples}")
    print(f"  Toxic: {n_toxic} ({n_toxic/n_samples*100:.1f}%)")
    
    # Get texts
    texts = train_df[text_col].fillna('').astype(str).tolist()
    labels = train_df[label_col].astype(int).values
    
    # Load model (this will download the model if first time)
    print(f"\n🤖 Loading Detoxify '{model_type}' model...")
    model = Detoxify(model_type)
    print("  Model loaded!")
    
    # Run predictions in batches
    print(f"\n🔮 Generating predictions (batch_size={batch_size})...")
    all_probs = []
    
    for i in tqdm(range(0, n_samples, batch_size), desc=f"  {lang}"):
        batch_texts = texts[i:i+batch_size]
        
        # Get predictions
        results = model.predict(batch_texts)
        
        # Extract toxicity scores
        toxicity_scores = results['toxicity']
        
        # Handle both single value and array
        if isinstance(toxicity_scores, (float, int)):
            all_probs.append(toxicity_scores)
        else:
            all_probs.extend(toxicity_scores)
    
    all_probs = np.array(all_probs)
    
    # Generate binary predictions with threshold 0.5
    all_preds = (all_probs >= 0.5).astype(int)
    
    # Compute quick metrics
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    try:
        roc_auc = roc_auc_score(labels, all_probs)
        f1 = f1_score(labels, all_preds)
        precision = precision_score(labels, all_preds, zero_division=0)
        recall = recall_score(labels, all_preds, zero_division=0)
        
        print(f"\n📊 Baseline Performance on Training Set:")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
    except Exception as e:
        print(f"  ⚠️  Could not compute metrics: {e}")
    
    # Create prediction DataFrame with text content
    pred_df = pd.DataFrame({
        text_col: texts,              # ← Include text for matching
        'y_true': labels,
        'y_prob': all_probs,
        'y_pred': all_preds,
    })
    
    # Add language column if not present
    if 'lang' not in pred_df.columns:
        pred_df['lang'] = lang
    
    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'detoxify_{lang}_train.csv'
    pred_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved predictions to: {output_path}")
    print(f"  Rows: {len(pred_df)}")
    print(f"  Columns: {list(pred_df.columns)}")
    
    return {
        'language': lang,
        'n_samples': n_samples,
        'n_toxic': int(n_toxic),
        'roc_auc': roc_auc if 'roc_auc' in locals() else None,
        'f1': f1 if 'f1' in locals() else None,
        'output_path': str(output_path)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate baseline predictions on training set for hard sample mining"
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['es', 'it', 'tr'],
        help='Languages to process (default: es it tr)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/baseline_predictions',
        help='Directory to save predictions (default: data/baseline_predictions)'
    )
    parser.add_argument(
        '--text_col',
        type=str,
        default='comment_text',
        help='Name of text column (default: comment_text)'
    )
    parser.add_argument(
        '--label_col',
        type=str,
        default='toxic',
        help='Name of label column (default: toxic)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for prediction (default: 32)'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='multilingual',
        choices=['original', 'unbiased', 'multilingual'],
        help='Detoxify model type (default: multilingual)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*70)
    print("GENERATE BASELINE PREDICTIONS ON TRAINING SET")
    print("="*70)
    print(f"\nLanguages: {', '.join(args.languages)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: Detoxify '{args.model_type}'")
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please ensure milestone3 data is prepared.")
        exit(1)
    
    # Process each language
    results = []
    for lang in args.languages:
        try:
            result = generate_predictions_for_language(
                lang=lang,
                data_dir=data_dir,
                output_dir=output_dir,
                text_col=args.text_col,
                label_col=args.label_col,
                batch_size=args.batch_size,
                model_type=args.model_type
            )
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = output_dir / 'train_baseline_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
    
    print(f"\n{'='*70}")
    print("✅ BASELINE PREDICTIONS GENERATED")
    print(f"{'='*70}")
    print("\nNext step: Run hard sample identification")
    print("Example command:")
    print(f"  python identify_hard_samples.py \\")
    print(f"      --baseline_pred {output_dir}/detoxify_es_train.csv \\")
    print(f"      --train_data {data_dir}/es/train.csv \\")
    print(f"      --output {data_dir}/es/train_weighted.csv \\")
    print(f"      --plot")


if __name__ == '__main__':
    main()

