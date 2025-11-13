"""
Simple baseline model using TF-IDF + Logistic Regression for multilingual toxicity detection.

Features:
- Language-specific training (separate model per language)
- Text preprocessing (lowercase, URL/mention removal, optional stopwords)
- TF-IDF vectorization with bigrams
- Logistic Regression with balanced class weights
- Threshold optimization on validation set (maximize F1)
- Prediction output with probabilities and binary labels
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    accuracy_score,
    classification_report
)
import joblib
import re
from pathlib import Path
import argparse
import yaml
import warnings
from typing import Tuple, Dict

warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Text preprocessing utilities for toxicity detection."""
    
    # Map ISO language codes to NLTK stopwords language names
    NLTK_LANGUAGE_MAP = {
        'en': 'english',
        'es': 'spanish',
        'fr': 'french',
        'de': 'german',
        'it': 'italian',
        'pt': 'portuguese',
        'ru': 'russian',
        'tr': 'turkish'
    }
    
    def __init__(self, lowercase=True, remove_urls=True, remove_mentions=True, 
                 remove_hashtags=False, use_stopwords=False, language='en'):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.use_stopwords = use_stopwords
        self.language = language
        
        # Try to load stopwords if requested
        self.stopwords = None
        if use_stopwords:
            try:
                from nltk.corpus import stopwords
                import nltk
                
                # Convert ISO language code to NLTK language name
                nltk_lang = self.NLTK_LANGUAGE_MAP.get(language, language)
                
                try:
                    self.stopwords = set(stopwords.words(nltk_lang))
                    print(f"  Loaded {len(self.stopwords)} stopwords for {language} ({nltk_lang})")
                except LookupError:
                    print(f"  Downloading stopwords for {language} ({nltk_lang})...")
                    nltk.download('stopwords', quiet=True)
                    try:
                        self.stopwords = set(stopwords.words(nltk_lang))
                        print(f"  Loaded {len(self.stopwords)} stopwords for {language} ({nltk_lang})")
                    except LookupError:
                        print(f"  Warning: Stopwords not available for {language}, skipping")
                        self.stopwords = None
            except ImportError:
                print("  Warning: NLTK not installed, skipping stopwords removal")
                self.stopwords = None
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing to a single text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords (if enabled and available)
        if self.stopwords is not None:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)
        
        return text
    
    def preprocess_series(self, texts: pd.Series) -> pd.Series:
        """Apply preprocessing to a pandas Series."""
        return texts.apply(self.preprocess)


def load_data(data_dir: Path, language: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test data for a specific language."""
    lang_dir = data_dir / language
    
    train_path = lang_dir / "train.csv"
    val_path = lang_dir / "val.csv"
    test_path = lang_dir / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded {language} data:")
    print(f"  Train: {len(train_df)} samples ({train_df['toxic'].mean()*100:.1f}% toxic)")
    print(f"  Val:   {len(val_df)} samples ({val_df['toxic'].mean()*100:.1f}% toxic)")
    print(f"  Test:  {len(test_df)} samples ({test_df['toxic'].mean()*100:.1f}% toxic)")
    
    return train_df, val_df, test_df


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, 
                       metric: str = 'f1') -> Tuple[float, float]:
    """
    Find the best classification threshold by maximizing a metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', or 'recall')
    
    Returns:
        best_threshold: Optimal threshold value
        best_score: Best metric score achieved
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # Test 81 thresholds from 0.1 to 0.9
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def train_model(X_train: pd.Series, y_train: pd.Series, language: str,
                use_stopwords: bool = False) -> Tuple[TextPreprocessor, TfidfVectorizer, LogisticRegression]:
    """
    Train TF-IDF + Logistic Regression model for a specific language.
    
    Args:
        X_train: Training texts
        y_train: Training labels
        language: Language code
        use_stopwords: Whether to use stopwords removal
    
    Returns:
        preprocessor: Text preprocessor
        vectorizer: TF-IDF vectorizer
        model: Trained logistic regression model
    """
    print(f"\nTraining model for {language}...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False,
        use_stopwords=use_stopwords,
        language=language  # Pass ISO language code directly
    )
    
    # Preprocess texts
    print("  Preprocessing texts...")
    X_train_clean = preprocessor.preprocess_series(X_train)
    
    # TF-IDF vectorization
    print("  Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=200000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        sublinear_tf=True,  # Apply sublinear tf scaling
        use_idf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression
    print("  Training logistic regression...")
    model = LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight='balanced',
        solver='lbfgs',
        n_jobs=-1
    )
    
    model.fit(X_train_tfidf, y_train)
    print("  Training complete!")
    
    return preprocessor, vectorizer, model


def evaluate_model(preprocessor: TextPreprocessor, 
                  vectorizer: TfidfVectorizer,
                  model: LogisticRegression,
                  X_val: pd.Series, 
                  y_val: pd.Series, 
                  language: str,
                  find_threshold: bool = True) -> Dict[str, float]:
    """
    Evaluate model performance on validation set.
    
    Args:
        preprocessor: Text preprocessor
        vectorizer: TF-IDF vectorizer
        model: Trained model
        X_val: Validation texts
        y_val: Validation labels
        language: Language code
        find_threshold: Whether to find optimal threshold
    
    Returns:
        results: Dictionary with metrics and best threshold
    """
    print(f"\nEvaluating on validation set...")
    
    # Preprocess and vectorize
    X_val_clean = preprocessor.preprocess_series(X_val)
    X_val_tfidf = vectorizer.transform(X_val_clean)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_val_tfidf)[:, 1]
    
    # Find best threshold on validation set
    if find_threshold:
        best_threshold, best_f1 = find_best_threshold(y_val, y_pred_proba, metric='f1')
        print(f"  Best threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
    else:
        best_threshold = 0.5
        best_f1 = None
    
    # Predict with best threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    
    results = {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'best_threshold': best_threshold
    }
    
    print(f"\n{language.upper()} Validation Results:")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Threshold:  {best_threshold:.3f}")
    
    return results


def predict_and_save(preprocessor: TextPreprocessor,
                    vectorizer: TfidfVectorizer,
                    model: LogisticRegression,
                    X_test: pd.Series,
                    y_test: pd.Series,
                    best_threshold: float,
                    output_path: Path,
                    language: str):
    """
    Generate predictions on test set and save to CSV.
    
    Args:
        preprocessor: Text preprocessor
        vectorizer: TF-IDF vectorizer
        model: Trained model
        X_test: Test texts
        y_test: Test labels
        best_threshold: Optimal threshold from validation
        output_path: Path to save predictions
        language: Language code
    """
    print(f"\nGenerating predictions on test set...")
    
    # Preprocess and vectorize
    X_test_clean = preprocessor.preprocess_series(X_test)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Predict binary labels with best threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Calculate test metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{language.upper()} Test Results:")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'id': range(len(y_test)),
        'y_true': y_test.values,
        'y_prob': y_pred_proba,
        'y_pred': y_pred,
        'best_threshold': best_threshold
    })
    
    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"  Predictions saved to: {output_path}")
    
    return {
        'roc_auc': roc_auc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train simple TF-IDF + LogReg baseline models for toxicity detection"
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
        help="Languages to train models for"
    )
    parser.add_argument(
        "--use_stopwords",
        action="store_true",
        help="Use stopwords removal (requires NLTK)"
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    data_dir = Path(config['data']['processed_dir'])
    model_output_dir = Path(config['training']['output_dir']) / "baseline"
    pred_output_dir = Path(config['evaluation']['output_dir'])
    
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Model output: {model_output_dir}")
    print(f"  Predictions output: {pred_output_dir}")
    print(f"  Languages: {', '.join(args.languages)}")
    print(f"  Use stopwords: {args.use_stopwords}")
    
    # Create output directories
    model_output_dir.mkdir(parents=True, exist_ok=True)
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results for all languages
    all_results = {}
    
    # Train model for each language
    for language in args.languages:
        print(f"\n{'='*70}")
        print(f"PROCESSING LANGUAGE: {language.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load data
            train_df, val_df, test_df = load_data(data_dir, language)
            
            # Prepare features and labels
            X_train = train_df['comment_text']
            y_train = train_df['toxic']
            X_val = val_df['comment_text']
            y_val = val_df['toxic']
            X_test = test_df['comment_text']
            y_test = test_df['toxic']
            
            # Train model
            preprocessor, vectorizer, model = train_model(
                X_train, y_train, language, 
                use_stopwords=args.use_stopwords
            )
            
            # Evaluate on validation set and find best threshold
            val_results = evaluate_model(
                preprocessor, vectorizer, model,
                X_val, y_val, language,
                find_threshold=True
            )
            
            # Get best threshold
            best_threshold = val_results['best_threshold']
            
            # Predict on test set
            test_results = predict_and_save(
                preprocessor, vectorizer, model,
                X_test, y_test, best_threshold,
                pred_output_dir / f"baseline_{language}.csv",
                language
            )
            
            # Save model artifacts
            lang_model_dir = model_output_dir / language
            lang_model_dir.mkdir(exist_ok=True)
            
            joblib.dump(preprocessor, lang_model_dir / "preprocessor.pkl")
            joblib.dump(vectorizer, lang_model_dir / "vectorizer.pkl")
            joblib.dump(model, lang_model_dir / "model.pkl")
            joblib.dump({'best_threshold': best_threshold}, 
                       lang_model_dir / "threshold.pkl")
            
            print(f"\nModel artifacts saved to: {lang_model_dir}")
            
            # Store results
            all_results[language] = {
                'val_roc_auc': val_results['roc_auc'],
                'val_f1': val_results['f1_score'],
                'val_precision': val_results['precision'],
                'val_recall': val_results['recall'],
                'test_roc_auc': test_results['roc_auc'],
                'test_f1': test_results['f1_score'],
                'test_precision': test_results['precision'],
                'test_recall': test_results['recall'],
                'best_threshold': best_threshold
            }
            
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print(f"Skipping {language}...")
            all_results[language] = {'error': str(e)}
            continue
        except Exception as e:
            print(f"\nUnexpected error processing {language}: {e}")
            import traceback
            traceback.print_exc()
            all_results[language] = {'error': str(e)}
            continue
    
    # Save results summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL LANGUAGES")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(4)
    
    # Save to CSV
    summary_path = model_output_dir / "baseline_results_summary.csv"
    results_df.to_csv(summary_path)
    print(f"Results summary saved to: {summary_path}")
    
    # Print summary table
    print("\nValidation Results:")
    if 'val_roc_auc' in results_df.columns:
        print(results_df[['val_roc_auc', 'val_f1', 'val_precision', 'val_recall', 'best_threshold']])
    
    print("\nTest Results:")
    if 'test_roc_auc' in results_df.columns:
        print(results_df[['test_roc_auc', 'test_f1', 'test_precision', 'test_recall']])
    
    print(f"\n{'='*70}")
    print("BASELINE TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()