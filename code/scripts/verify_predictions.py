"""
Quick verification script for baseline model predictions.
Checks that prediction files have correct format and reasonable values.
"""

import pandas as pd
from pathlib import Path
import sys


def verify_prediction_file(pred_file: Path, language: str) -> bool:
    """Verify a single prediction file."""
    print(f"\nVerifying {language} predictions: {pred_file}")
    
    if not pred_file.exists():
        print(f"  ❌ File not found!")
        return False
    
    try:
        # Load predictions
        df = pd.read_csv(pred_file)
        
        # Check required columns
        required_cols = ['id', 'y_true', 'y_prob', 'y_pred', 'best_threshold']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  ❌ Missing columns: {missing_cols}")
            return False
        
        # Check data types and ranges
        checks = []
        
        # Check y_true is binary
        if not df['y_true'].isin([0, 1]).all():
            checks.append("y_true not binary (0/1)")
        
        # Check y_prob is in [0, 1]
        if not ((df['y_prob'] >= 0) & (df['y_prob'] <= 1)).all():
            checks.append("y_prob not in [0, 1]")
        
        # Check y_pred is binary
        if not df['y_pred'].isin([0, 1]).all():
            checks.append("y_pred not binary (0/1)")
        
        # Check threshold is in [0, 1]
        threshold = df['best_threshold'].iloc[0]
        if not (0 <= threshold <= 1):
            checks.append(f"threshold {threshold} not in [0, 1]")
        
        if checks:
            print(f"  ❌ Data validation errors:")
            for check in checks:
                print(f"     - {check}")
            return False
        
        # Print statistics
        print(f"  ✅ File structure valid")
        print(f"     Samples: {len(df)}")
        print(f"     True toxic rate: {df['y_true'].mean()*100:.1f}%")
        print(f"     Pred toxic rate: {df['y_pred'].mean()*100:.1f}%")
        print(f"     Avg probability: {df['y_prob'].mean():.4f}")
        print(f"     Best threshold: {threshold:.3f}")
        
        # Calculate accuracy
        accuracy = (df['y_true'] == df['y_pred']).mean()
        print(f"     Accuracy: {accuracy*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False


def main():
    """Verify all prediction files."""
    pred_dir = Path("output/predictions")
    
    if not pred_dir.exists():
        print(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)
    
    print("="*70)
    print("BASELINE MODEL PREDICTIONS VERIFICATION")
    print("="*70)
    
    languages = ["en", "es", "fr", "de", "it", "pt", "ru", "tr"]
    results = {}
    
    for lang in languages:
        pred_file = pred_dir / f"baseline_{lang}.csv"
        results[lang] = verify_prediction_file(pred_file, lang)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    for lang, status in results.items():
        status_str = "✅ PASS" if status else "❌ FAIL"
        print(f"  {lang}: {status_str}")
    
    if passed == total:
        print("\n🎉 All prediction files verified successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} file(s) failed verification")
        sys.exit(1)


if __name__ == "__main__":
    main()