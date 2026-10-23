#!/usr/bin/env python3
"""
main_pipeline.py
End-to-end execution script for the social media engagement prediction pipeline.
Runs all stages: data acquisition → preprocessing → feature engineering → 
sentiment analysis → model training → explainability → dashboard export.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_acquisition import create_synthetic_fallback
from preprocessing import TextCleaner
from feature_engineering import FeatureEngineer
from sentiment_analysis import SentimentAnalyzer
from model_training import ModelTrainer
from explainability import SHAPExplainer
from dashboard_export import DashboardExporter
from serving_export import export_serving_artifacts

def run_pipeline(use_synthetic: bool = True, n_posts: int = 11000):
    """Execute full pipeline."""
    print("="*60)
    print("SOCIAL MEDIA ENGAGEMENT PREDICTION PIPELINE")
    print("="*60)

    # Stage 1: Data Acquisition
    print("\n[1/8] Data Acquisition...")
    if use_synthetic:
        df = create_synthetic_fallback(n_posts=n_posts)
    else:
        # Would load real data here
        raise NotImplementedError("Real data loading not configured.")
    print(f"  Loaded {len(df)} posts")

    # Stage 2: Preprocessing
    print("\n[2/8] Text Preprocessing...")
    cleaner = TextCleaner()
    cleaned_texts, metadata = cleaner.batch_clean(df["text"].tolist())
    df["cleaned_text"] = cleaned_texts
    print(f"  Cleaned {len(cleaned_texts)} texts")

    # Stage 3: Feature Engineering
    print("\n[3/8] Feature Engineering...")
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df, cleaned_texts, metadata)
    print(f"  Engineered {features.shape[1]} features")
    print(f"  Class balance: {features['engagement_label'].value_counts().to_dict()}")

    # Stage 4: Sentiment Analysis
    print("\n[4/8] Sentiment Analysis...")
    analyzer = SentimentAnalyzer(use_sbert=False)  # Use fallback for speed
    vader_df, embeddings = analyzer.analyze(df["text"].tolist())

    # Add sentiment features
    for col in ["neg", "neu", "pos", "compound"]:
        features[f"vader_{col}"] = vader_df[col].values

    # Add embedding features efficiently (avoid repeated DataFrame inserts)
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=features.index)
    features = pd.concat([features, emb_df], axis=1)

    print(f"  Added {embeddings.shape[1]} embedding dimensions (concatenated)")

    # Stage 5: Serving Artefact Export (for streaming_app.py's live scoring)
    # Separate feature contract from the dashboard pipeline below -- see
    # src/serving_export.py docstring for why it can't reuse `features`/`X`.
    # Placed before Model Training/SHAP so a downstream failure there
    # (e.g. SHAP not supporting whichever model wins the dashboard bake-off)
    # can't prevent the live-serving artefact from being produced.
    print("\n[5/8] Serving Artefact Export...")
    _, serving_metrics = export_serving_artifacts(
        df, cleaned_texts, metadata, features["engagement_label"],
        output_path="models/serving_artifacts.pkl")
    print(f"  Serving model: {serving_metrics['model_type']} "
          f"({serving_metrics['n_features']} features)")
    print(f"  Test F1: {serving_metrics['f1']:.4f} | "
          f"Accuracy: {serving_metrics['accuracy']:.4f} | "
          f"ROC-AUC: {serving_metrics['roc_auc']:.4f}")

    # Stage 6: Model Training
    print("\n[6/8] Model Training...")
    X = features.drop(["engagement_label", "interaction_score"], axis=1, errors="ignore")
    y = features["engagement_label"]

    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    results = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n  Results:")
    print(results.to_string(index=False))

    # Stage 7: Explainability
    print("\n[7/8] SHAP Explainability...")
    explainer = SHAPExplainer(trainer.best_model, X.columns.tolist())
    explainer.fit_explainer(X_train.sample(min(100, len(X_train))))
    explainer.compute_values(X_test)

    importance = explainer.global_importance(X_test)
    print("\n  Top 10 Features:")
    print(importance.head(10).to_string(index=False))

    # Check for target leakage
    leakage = explainer.detect_target_leakage(X_test)

    # Stage 8: Dashboard Export
    print("\n[8/8] Dashboard Export...")
    exporter = DashboardExporter()

    # Add predictions to dataframe
    df["predicted_class"] = trainer.best_model.predict(X)
    df["predicted_proba"] = trainer.best_model.predict_proba(X)[:, 1]

    exporter.export_scored_posts(df)
    exporter.export_model_metrics(results)
    exporter.export_shap_importance(importance)

    # Save model
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/best_model.joblib")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Best F1-Score: {results.loc[results['Model']==trainer.best_model_name, 'F1-Score'].values[0]:.4f}")
    print(f"Target Leakage Detected: {leakage}")
    print("\nOutputs:")
    print("  - models/best_model.joblib")
    print("  - models/serving_artifacts.pkl")
    print("  - data/dashboard/scored_posts.csv")
    print("  - data/dashboard/model_metrics.csv")
    print("  - data/dashboard/shap_importance.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run engagement prediction pipeline")
    parser.add_argument("--synthetic", action="store_true", default=True,
                       help="Use synthetic data (default: True)")
    parser.add_argument("--n-posts", type=int, default=1000,
                       help="Number of synthetic posts to generate")
    args = parser.parse_args()

    run_pipeline(use_synthetic=args.synthetic, n_posts=args.n_posts)
