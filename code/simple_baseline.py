"""
Simple baseline model using TF-IDF + Logistic Regression for multilingual toxicity detection.
Trains separate models for each language.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import joblib
import os
from pathlib import Path
import argparse
import yaml


def load_data(data_path, language):
    """Load training and validation data for a specific language."""
    train_path = Path(data_path) / "processed" / f"{language}_train.csv"
    val_path = Path(data_path) / "processed" / f"{language}_val.csv"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    return train_df, val_df


def train_model(X_train, y_train, language):
    """Train TF-IDF + Logistic Regression model for a specific language."""
    print(f"Training model for {language}...")
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=None,  # Will be language-specific
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Logistic Regression
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    model.fit(X_train_tfidf, y_train)
    
    return vectorizer, model


def evaluate_model(vectorizer, model, X_val, y_val, language):
    """Evaluate model performance on validation set."""
    X_val_tfidf = vectorizer.transform(X_val)
    y_pred_proba = model.predict_proba(X_val_tfidf)[:, 1]
    y_pred = model.predict(X_val_tfidf)
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)
    
    print(f"\n{language} Results:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    return roc_auc, f1


def main():
    parser = argparse.ArgumentParser(description="Train simple baseline models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "fr", "de", "it", "pt", "ru", "tr"], 
                       help="Languages to train models for")
    parser.add_argument("--output_dir", type=str, default="output/runs", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for language in args.languages:
        print(f"\n{'='*50}")
        print(f"Processing {language}")
        print(f"{'='*50}")
        
        try:
            # Load data
            train_df, val_df = load_data(data_path, language)
            
            # Prepare features and labels
            X_train = train_df['comment_text'].fillna('')
            y_train = train_df['toxic']
            X_val = val_df['comment_text'].fillna('')
            y_val = val_df['toxic']
            
            # Train model
            vectorizer, model = train_model(X_train, y_train, language)
            
            # Evaluate model
            roc_auc, f1 = evaluate_model(vectorizer, model, X_val, y_val, language)
            
            # Save model
            model_dir = output_dir / f"baseline_{language}"
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
            joblib.dump(model, model_dir / "model.pkl")
            
            results[language] = {
                'roc_auc': roc_auc,
                'f1_score': f1
            }
            
            print(f"Model saved to {model_dir}")
            
        except Exception as e:
            print(f"Error processing {language}: {str(e)}")
            results[language] = {'error': str(e)}
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_df.to_csv(output_dir / "baseline_results.csv")
    print(f"\nResults summary saved to {output_dir / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()
