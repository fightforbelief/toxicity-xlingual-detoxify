"""
Unified evaluation script for toxicity detection models.
Computes ROC-AUC and F1 scores across languages and model types.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score
import argparse
import yaml
from pathlib import Path
import json
from typing import Dict, List, Tuple


def load_predictions(pred_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from file (CSV with 'prediction' and 'label' columns)."""
    df = pd.read_csv(pred_file)
    return df['prediction'].values, df['label'].values


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    # Basic metrics
    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int))
    
    # Precision-Recall metrics
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    # Additional metrics
    f1_macro = f1_score(y_true, (y_pred > 0.5).astype(int), average='macro')
    
    return {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'f1_macro': f1_macro,
        'average_precision': avg_precision,
        'precision_at_50': precision[np.argmin(np.abs(recall - 0.5))] if len(precision) > 0 else 0.0
    }


def evaluate_language_predictions(pred_dir: str, language: str) -> Dict[str, float]:
    """Evaluate predictions for a specific language."""
    pred_file = Path(pred_dir) / f"{language}_predictions.csv"
    
    if not pred_file.exists():
        print(f"Warning: Prediction file not found for {language}: {pred_file}")
        return {}
    
    y_true, y_pred = load_predictions(str(pred_file))
    metrics = compute_metrics(y_true, y_pred)
    
    print(f"\n{language.upper()} Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return metrics


def evaluate_model_across_languages(pred_dir: str, languages: List[str], model_name: str) -> Dict[str, Dict[str, float]]:
    """Evaluate a model across all languages."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    results = {}
    overall_metrics = []
    
    for language in languages:
        lang_metrics = evaluate_language_predictions(pred_dir, language)
        if lang_metrics:
            results[language] = lang_metrics
            overall_metrics.append(lang_metrics)
    
    # Compute macro-averaged metrics
    if overall_metrics:
        macro_metrics = {}
        for metric in overall_metrics[0].keys():
            values = [m[metric] for m in overall_metrics if not np.isnan(m[metric])]
            macro_metrics[f"macro_{metric}"] = np.mean(values) if values else 0.0
        
        results['macro_average'] = macro_metrics
        
        print(f"\n{model_name} - Macro-averaged Results:")
        for metric, value in macro_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def compare_models(results: Dict[str, Dict[str, Dict[str, float]]], output_file: str):
    """Compare results across different models."""
    comparison_data = []
    
    for model_name, model_results in results.items():
        for language, metrics in model_results.items():
            if language == 'macro_average':
                continue
                
            row = {
                'model': model_name,
                'language': language,
                **metrics
            }
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_file, index=False)
    
    # Create summary table
    summary_data = []
    for model_name, model_results in results.items():
        if 'macro_average' in model_results:
            row = {
                'model': model_name,
                **model_results['macro_average']
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file.replace('.csv', '_summary.csv'), index=False)
    
    print(f"\nComparison results saved to {output_file}")
    print(f"Summary results saved to {output_file.replace('.csv', '_summary.csv')}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate toxicity detection models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction files")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "it", "pt", "ru", "tr"],
                       help="Languages to evaluate")
    parser.add_argument("--models", nargs="+", default=["baseline", "detoxify_original", "detoxify_translated"],
                       help="Model names to evaluate")
    parser.add_argument("--output", type=str, default="output/evaluation_results.csv",
                       help="Output file for comparison results")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    all_results = {}
    
    # Evaluate each model
    for model_name in args.models:
        model_pred_dir = Path(args.pred_dir) / model_name
        if model_pred_dir.exists():
            model_results = evaluate_model_across_languages(
                str(model_pred_dir), 
                args.languages, 
                model_name
            )
            all_results[model_name] = model_results
        else:
            print(f"Warning: Prediction directory not found for {model_name}: {model_pred_dir}")
    
    # Compare models
    if all_results:
        compare_models(all_results, args.output)
        
        # Print final comparison
        print(f"\n{'='*80}")
        print("FINAL COMPARISON")
        print(f"{'='*80}")
        
        for model_name, model_results in all_results.items():
            if 'macro_average' in model_results:
                print(f"\n{model_name}:")
                for metric, value in model_results['macro_average'].items():
                    print(f"  {metric}: {value:.4f}")
    else:
        print("No valid results found for comparison.")


if __name__ == "__main__":
    main()
