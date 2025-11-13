"""
Check which languages are available in Jigsaw 2020 validation data.
"""

import pandas as pd
from pathlib import Path


def check_data_availability():
    """Check which languages have data available."""
    
    # Check validation.csv
    val_file = Path("data/raw/jigsaw-2020/validation.csv")
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        return
    
    print(f"✅ Found validation file: {val_file}")
    
    # Load validation data
    val_df = pd.read_csv(val_file)
    
    print(f"\nTotal samples: {len(val_df)}")
    print(f"Columns: {list(val_df.columns)}")
    
    # Check language distribution
    if 'lang' in val_df.columns:
        print("\n" + "="*60)
        print("LANGUAGE DISTRIBUTION IN VALIDATION SET")
        print("="*60)
        
        lang_counts = val_df['lang'].value_counts().sort_index()
        
        for lang, count in lang_counts.items():
            toxic_count = val_df[val_df['lang'] == lang]['toxic'].sum()
            toxic_pct = toxic_count / count * 100 if count > 0 else 0
            print(f"{lang:4s}: {count:5d} samples ({toxic_count:4d} toxic, {toxic_pct:5.1f}%)")
        
        print("\n" + "="*60)
        
        # Check which expected languages are missing
        expected_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'tr']
        available_languages = set(lang_counts.index)
        missing_languages = set(expected_languages) - available_languages
        
        if missing_languages:
            print("\n⚠️  MISSING LANGUAGES:")
            for lang in sorted(missing_languages):
                print(f"  - {lang}")
        else:
            print("\n✅ All expected languages are present!")
    else:
        print("\n❌ No 'lang' column found in validation data!")
    
    # Check processed data
    print("\n" + "="*60)
    print("PROCESSED DATA AVAILABILITY")
    print("="*60)
    
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print(f"❌ Processed directory not found: {processed_dir}")
        return
    
    expected_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'tr']
    
    for lang in expected_languages:
        lang_dir = processed_dir / lang
        
        if not lang_dir.exists():
            print(f"❌ {lang}: directory not found")
            continue
        
        train_file = lang_dir / "train.csv"
        val_file = lang_dir / "val.csv"
        test_file = lang_dir / "test.csv"
        heldout_file = lang_dir / "heldout.csv"
        
        files_exist = [
            ("train", train_file.exists()),
            ("val", val_file.exists()),
            ("test", test_file.exists()),
            ("heldout", heldout_file.exists())
        ]
        
        status = "✅" if all(exists for _, exists in files_exist) else "⚠️"
        files_status = ", ".join([f"{name}:{'✓' if exists else '✗'}" for name, exists in files_exist])
        
        print(f"{status} {lang}: {files_status}")
        
        # If all files exist, show sample counts
        if all(exists for _, exists in files_exist):
            train_df = pd.read_csv(train_file)
            val_df = pd.read_csv(val_file)
            test_df = pd.read_csv(test_file)
            heldout_df = pd.read_csv(heldout_file)
            
            total = len(train_df) + len(val_df) + len(test_df) + len(heldout_df)
            print(f"       Total: {total} samples "
                  f"(train:{len(train_df)}, val:{len(val_df)}, "
                  f"test:{len(test_df)}, heldout:{len(heldout_df)})")


if __name__ == "__main__":
    check_data_availability()
