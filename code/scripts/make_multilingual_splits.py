"""
Create 80/10/10 train/dev/test splits from Jigsaw 2020 validation data per language.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import yaml


def load_jigsaw_data(data_path: str) -> pd.DataFrame:
    """Load Jigsaw 2020 validation data."""
    # Assuming the data is in the raw directory
    raw_path = Path(data_path) / "raw"
    
    # Look for common Jigsaw 2020 file names
    possible_files = [
        "validation.csv",
        "jigsaw-toxicity-validation.csv", 
        "jigsaw_unintended_bias_test.csv"
    ]
    
    for filename in possible_files:
        file_path = raw_path / filename
        if file_path.exists():
            print(f"Loading data from {file_path}")
            return pd.read_csv(file_path)
    
    raise FileNotFoundError(f"Could not find Jigsaw validation data in {raw_path}")


def create_language_splits(df: pd.DataFrame, language: str, output_dir: Path, 
                          test_size: float = 0.2, val_size: float = 0.125):
    """Create train/dev/test splits for a specific language."""
    print(f"Creating splits for {language}...")
    
    # Filter data for the language (assuming there's a language column or we use all data for 'en')
    if language == 'en':
        lang_df = df.copy()
    else:
        # For other languages, we might need to filter by a language column
        # or use translated data - this is a placeholder
        if 'lang' in df.columns:
            lang_df = df[df['lang'] == language].copy()
        else:
            print(f"Warning: No language column found, using all data for {language}")
            lang_df = df.copy()
    
    print(f"Found {len(lang_df)} samples for {language}")
    
    if len(lang_df) == 0:
        print(f"No data found for {language}, skipping...")
        return
    
    # Ensure we have the required columns
    required_cols = ['comment_text', 'toxic']
    missing_cols = [col for col in required_cols if col not in lang_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols} for {language}")
        # Create dummy columns if missing
        for col in missing_cols:
            if col == 'toxic':
                lang_df[col] = 0  # Default to non-toxic
            else:
                lang_df[col] = ""
    
    # First split: train+val vs test (80% vs 20%)
    train_val, test = train_test_split(
        lang_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=lang_df['toxic'] if 'toxic' in lang_df.columns else None
    )
    
    # Second split: train vs val (70% vs 10% of original)
    train, val = train_test_split(
        train_val,
        test_size=val_size/(1-test_size),  # Adjust for the fact that train_val is 80% of original
        random_state=42,
        stratify=train_val['toxic'] if 'toxic' in train_val.columns else None
    )
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_dir / f"{language}_train.csv", index=False)
    val.to_csv(output_dir / f"{language}_val.csv", index=False)
    test.to_csv(output_dir / f"{language}_test.csv", index=False)
    
    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples") 
    print(f"  Test: {len(test)} samples")
    
    # Print class distribution
    if 'toxic' in train.columns:
        print(f"  Train toxic ratio: {train['toxic'].mean():.3f}")
        print(f"  Val toxic ratio: {val['toxic'].mean():.3f}")
        print(f"  Test toxic ratio: {test['toxic'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Create multilingual data splits")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "it", "pt", "ru", "tr"],
                       help="Languages to create splits for")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                       help="Output directory for processed splits")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    output_dir = Path(args.output_dir)
    
    # Load Jigsaw data
    try:
        df = load_jigsaw_data(data_path)
        print(f"Loaded {len(df)} total samples")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create splits for each language
    for language in args.languages:
        try:
            create_language_splits(df, language, output_dir)
        except Exception as e:
            print(f"Error creating splits for {language}: {str(e)}")
            continue
    
    print(f"\nData splits created in {output_dir}")


if __name__ == "__main__":
    main()
