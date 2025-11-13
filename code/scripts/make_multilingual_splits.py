import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path
import yaml
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


def load_jigsaw_2020_data(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Jigsaw 2020 multilingual data.
    
    Returns:
        train_df: English training data
        val_df: Multilingual validation data with language labels
    """
    jigsaw_2020_dir = raw_dir / "jigsaw-2020"
    
    # Load English training data
    train_file = jigsaw_2020_dir / "jigsaw-toxic-comment-train.csv"
    if not train_file.exists():
        raise FileNotFoundError(
            f"English training data not found at {train_file}\n"
            "Please download from: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification"
        )
    
    print(f"Loading English training data from {train_file}")
    train_df = pd.read_csv(train_file)
    
    # Load multilingual validation data
    val_file = jigsaw_2020_dir / "validation.csv"
    if not val_file.exists():
        raise FileNotFoundError(
            f"Multilingual validation data not found at {val_file}\n"
            "Please download from: https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification"
        )
    
    print(f"Loading multilingual validation data from {val_file}")
    val_df = pd.read_csv(val_file)
    
    return train_df, val_df


def prepare_english_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare English dataset by combining train and validation subsets.
    
    Args:
        train_df: Jigsaw training data (only English, ~224k samples)
        val_df: Multilingual validation data
    
    Returns:
        Combined English dataset with 'comment_text', 'toxic', 'lang' columns
    """
    # Process training data
    en_train = train_df[['comment_text', 'toxic']].copy()
    en_train['lang'] = 'en'
    
    # Process validation subset for English
    en_val = val_df[val_df['lang'] == 'en'][['comment_text', 'toxic', 'lang']].copy()
    
    # Combine
    en_combined = pd.concat([en_train, en_val], ignore_index=True)
    
    print(f"English data: {len(en_train)} train + {len(en_val)} val = {len(en_combined)} total")
    
    return en_combined


def prepare_other_language_data(val_df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Prepare data for non-English languages from validation set.
    
    Args:
        val_df: Multilingual validation data
        language: Language code (es, fr, de, it, pt, ru, tr)
    
    Returns:
        Language-specific dataset
    """
    lang_df = val_df[val_df['lang'] == language][['comment_text', 'toxic', 'lang']].copy()
    
    print(f"{language.upper()} data: {len(lang_df)} samples from validation set")
    
    return lang_df


def create_splits_with_heldout(
    df: pd.DataFrame,
    language: str,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    heldout_ratio: float = 0.1,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test/heldout splits with stratification.
    
    Split strategy:
    1. First split: heldout (10%) vs rest (90%)
    2. Second split: test (10%) vs train+val (80%)
    3. Third split: val (10%) vs train (70%)
    
    Args:
        df: DataFrame with 'comment_text', 'toxic', 'lang' columns
        language: Language code
        output_dir: Output directory for splits
        train_ratio: Training set ratio (default 0.7)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        heldout_ratio: Held-out test ratio (default 0.1)
        random_state: Random seed
    
    Returns:
        Dictionary with 'train', 'val', 'test', 'heldout' DataFrames
    """
    print(f"\nCreating splits for {language.upper()}...")
    print(f"Total samples: {len(df)}")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio + heldout_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Check if we have enough samples
    min_samples_per_class = 10
    toxic_count = df['toxic'].sum()
    non_toxic_count = len(df) - toxic_count
    
    print(f"  Toxic samples: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")
    print(f"  Non-toxic samples: {non_toxic_count} ({non_toxic_count/len(df)*100:.1f}%)")
    
    if toxic_count < min_samples_per_class or non_toxic_count < min_samples_per_class:
        print(f"  WARNING: Insufficient samples for stratification, using random split")
        stratify = None
    else:
        stratify = df['toxic']
    
    # Split 1: heldout vs rest
    rest_df, heldout_df = train_test_split(
        df,
        test_size=heldout_ratio,
        random_state=random_state,
        stratify=stratify
    )
    
    # Recalculate stratify for rest_df
    if stratify is not None:
        stratify_rest = rest_df['toxic']
    else:
        stratify_rest = None
    
    # Split 2: test vs train+val
    train_val_df, test_df = train_test_split(
        rest_df,
        test_size=test_ratio / (1 - heldout_ratio),  # Adjust for already removed heldout
        random_state=random_state,
        stratify=stratify_rest
    )
    
    # Recalculate stratify for train_val_df
    if stratify is not None:
        stratify_train_val = train_val_df['toxic']
    else:
        stratify_train_val = None
    
    # Split 3: val vs train
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio / (train_ratio + val_ratio),  # Adjust for proportions
        random_state=random_state,
        stratify=stratify_train_val
    )
    
    # Verify splits
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'heldout': heldout_df
    }
    
    print(f"\nSplit sizes:")
    total_samples = len(df)
    for split_name, split_df in splits.items():
        toxic_ratio = split_df['toxic'].mean()
        print(f"  {split_name:8s}: {len(split_df):6d} samples "
              f"({len(split_df)/total_samples*100:5.1f}%) | "
              f"toxic: {toxic_ratio*100:5.1f}%")
    
    # Save splits
    lang_output_dir = output_dir / language
    lang_output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_df in splits.items():
        output_file = lang_output_dir / f"{split_name}.csv"
        split_df.to_csv(output_file, index=False)
        print(f"  Saved {split_name} to {output_file}")
    
    return splits


def create_dataset_summary(output_dir: Path, languages: list):
    """Create a summary CSV of all dataset splits."""
    summary_data = []
    
    for lang in languages:
        lang_dir = output_dir / lang
        if not lang_dir.exists():
            continue
        
        for split in ['train', 'val', 'test', 'heldout']:
            split_file = lang_dir / f"{split}.csv"
            if split_file.exists():
                df = pd.read_csv(split_file)
                summary_data.append({
                    'language': lang,
                    'split': split,
                    'samples': len(df),
                    'toxic': df['toxic'].sum(),
                    'toxic_ratio': df['toxic'].mean()
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "dataset_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nDataset summary saved to {summary_file}")
    print("\nSummary by language:")
    print(summary_df.pivot_table(
        index='language',
        columns='split',
        values='samples',
        aggfunc='sum'
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Create multilingual train/val/test/heldout splits from Jigsaw 2020 data"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es", "fr", "de", "it", "pt", "ru", "tr"],
        help="Languages to process (default: all 8 languages)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--heldout_ratio",
        type=float,
        default=0.1,
        help="Held-out test ratio (default: 0.1)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total = args.train_ratio + args.val_ratio + args.test_ratio + args.heldout_ratio
    if not np.isclose(total, 1.0):
        print(f"Error: Ratios must sum to 1.0, got {total}")
        return
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config - FIXED: use raw_dir and processed_dir
    raw_dir = Path(config['data']['raw_dir'])
    output_dir = Path(config['data']['processed_dir'])
    
    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load Jigsaw 2020 data
    try:
        train_df, val_df = load_jigsaw_2020_data(raw_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    print(f"\nLoaded data:")
    print(f"  English training: {len(train_df)} samples")
    print(f"  Multilingual validation: {len(val_df)} samples")
    print(f"  Validation languages: {sorted(val_df['lang'].unique())}")
    
    # Process each language
    failed_languages = []
    
    for language in args.languages:
        try:
            print(f"\n{'='*60}")
            print(f"Processing {language.upper()}")
            print(f"{'='*60}")
            
            # Prepare language-specific data
            if language == 'en':
                lang_df = prepare_english_data(train_df, val_df)
            else:
                lang_df = prepare_other_language_data(val_df, language)
            
            if len(lang_df) == 0:
                print(f"No data found for {language}, skipping...")
                failed_languages.append(language)
                continue
            
            # Create splits
            create_splits_with_heldout(
                lang_df,
                language,
                output_dir,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                heldout_ratio=args.heldout_ratio,
                random_state=args.random_state
            )
            
        except Exception as e:
            print(f"Error processing {language}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_languages.append(language)
            continue
    
    # Create summary
    successful_languages = [lang for lang in args.languages if lang not in failed_languages]
    if successful_languages:
        create_dataset_summary(output_dir, successful_languages)
    
    # Final report
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {', '.join(successful_languages)}")
    if failed_languages:
        print(f"Failed: {', '.join(failed_languages)}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()