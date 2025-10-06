"""
Create manifest for translated training files.
This script prepares an index of translated training data for multilingual toxicity detection.
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import yaml
from typing import Dict, List


def create_translation_manifest(translated_data_dir: str, output_file: str, 
                               languages: List[str]) -> Dict:
    """Create a manifest file for translated training data."""
    
    manifest = {
        "languages": languages,
        "files": {},
        "statistics": {}
    }
    
    translated_dir = Path(translated_data_dir)
    
    for language in languages:
        lang_files = []
        lang_stats = {
            "total_samples": 0,
            "toxic_samples": 0,
            "files": []
        }
        
        # Look for translated files for this language
        # Common patterns: {lang}_train.csv, train_{lang}.csv, etc.
        possible_patterns = [
            f"{language}_train.csv",
            f"train_{language}.csv", 
            f"{language}_translated.csv",
            f"translated_{language}.csv"
        ]
        
        for pattern in possible_patterns:
            file_path = translated_dir / pattern
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    lang_files.append({
                        "filename": pattern,
                        "path": str(file_path),
                        "samples": len(df),
                        "columns": list(df.columns)
                    })
                    
                    lang_stats["files"].append(pattern)
                    lang_stats["total_samples"] += len(df)
                    
                    # Count toxic samples if column exists
                    if 'toxic' in df.columns:
                        lang_stats["toxic_samples"] += df['toxic'].sum()
                    
                    print(f"Found {pattern}: {len(df)} samples")
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
        
        manifest["files"][language] = lang_files
        manifest["statistics"][language] = lang_stats
        
        if lang_files:
            print(f"{language}: {lang_stats['total_samples']} total samples, "
                  f"{lang_stats['toxic_samples']} toxic")
        else:
            print(f"Warning: No translated files found for {language}")
    
    # Save manifest
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nTranslation manifest saved to {output_file}")
    return manifest


def validate_translation_files(manifest: Dict) -> Dict[str, List[str]]:
    """Validate that translation files have required structure."""
    issues = {}
    
    for language, files in manifest["files"].items():
        lang_issues = []
        
        for file_info in files:
            file_path = Path(file_info["path"])
            
            try:
                df = pd.read_csv(file_path)
                
                # Check required columns
                required_cols = ['comment_text', 'toxic']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    lang_issues.append(f"{file_info['filename']}: Missing columns {missing_cols}")
                
                # Check for empty files
                if len(df) == 0:
                    lang_issues.append(f"{file_info['filename']}: Empty file")
                
                # Check for missing values in critical columns
                if 'comment_text' in df.columns and df['comment_text'].isna().any():
                    lang_issues.append(f"{file_info['filename']}: Missing values in comment_text")
                
                if 'toxic' in df.columns and df['toxic'].isna().any():
                    lang_issues.append(f"{file_info['filename']}: Missing values in toxic")
                
            except Exception as e:
                lang_issues.append(f"{file_info['filename']}: Error reading file - {str(e)}")
        
        if lang_issues:
            issues[language] = lang_issues
    
    return issues


def create_sample_output(manifest: Dict, output_dir: str, samples_per_lang: int = 5):
    """Create sample output files for inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for language, files in manifest["files"].items():
        if not files:
            continue
            
        # Take first file for this language
        file_info = files[0]
        df = pd.read_csv(file_info["path"])
        
        # Sample data
        sample_df = df.sample(n=min(samples_per_lang, len(df)), random_state=42)
        sample_file = output_path / f"{language}_sample.csv"
        sample_df.to_csv(sample_file, index=False)
        
        print(f"Sample for {language} saved to {sample_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare translated data manifest")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--translated_dir", type=str, required=True, 
                       help="Directory containing translated training files")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "it", "pt", "ru", "tr"],
                       help="Languages to include in manifest")
    parser.add_argument("--output", type=str, default="data/processed/translation_manifest.json",
                       help="Output file for manifest")
    parser.add_argument("--validate", action="store_true", help="Validate translation files")
    parser.add_argument("--create_samples", action="store_true", help="Create sample files for inspection")
    parser.add_argument("--samples_dir", type=str, default="data/processed/samples",
                       help="Directory for sample files")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create manifest
    manifest = create_translation_manifest(
        args.translated_dir, 
        args.output, 
        args.languages
    )
    
    # Validate files if requested
    if args.validate:
        print("\nValidating translation files...")
        issues = validate_translation_files(manifest)
        
        if issues:
            print("\nValidation issues found:")
            for language, lang_issues in issues.items():
                print(f"\n{language}:")
                for issue in lang_issues:
                    print(f"  - {issue}")
        else:
            print("All files validated successfully!")
    
    # Create sample files if requested
    if args.create_samples:
        print("\nCreating sample files...")
        create_sample_output(manifest, args.samples_dir)
    
    print(f"\nManifest creation complete!")
    print(f"Total languages: {len(manifest['languages'])}")
    print(f"Total files: {sum(len(files) for files in manifest['files'].values())}")


if __name__ == "__main__":
    main()
