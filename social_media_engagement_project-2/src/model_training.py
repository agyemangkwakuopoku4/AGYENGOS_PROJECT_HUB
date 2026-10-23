"""
model_training.py
Multi-model training, hyperparameter optimisation, and evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
try:
    import xgboost as xgb
    _XGBOOST_AVAILABLE = True
except Exception:
    xgb = None
    _XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed — XGBoost model will be skipped.")
import joblib

class ModelTrainer:
    """Trains and compares multiple classifiers for engagement prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                     test_size: float = 0.2) -> Tuple:
        """Stratified train-test split."""
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

    def define_models(self) -> Dict:
        """Define model configurations and hyperparameter grids."""
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l2"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [200, 400],
                    "max_depth": [6, 10, None],
                    "min_samples_leaf": [1, 3]
                }
            },
        }

        # Add XGBoost model only if xgboost is available in the environment
        if _XGBOOST_AVAILABLE and xgb is not None:
            try:
                # Try to instantiate an XGBoost classifier to ensure native libs load
                test_model = xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=self.random_state
                )
                # If instantiation succeeds, add to models
                models["XGBoost"] = {
                    "model": test_model,
                    "params": {
                        "n_estimators": [200, 400],
                        "max_depth": [3, 5, 7],
                        "learning_rate": [0.01, 0.1, 0.2]
                    }
                }
            except Exception as e:
                print(f"Warning: XGBoost present but failed to initialise ({e}). Skipping XGBoost.")

        return models

    def train_and_evaluate(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """Train all models with cross-validated grid search."""
        models_config = self.define_models()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        results = []
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            # Use single-process search for XGBoost where native libs sometimes
            # fail to load in worker subprocesses on macOS. Default to parallel.
            grid_n_jobs = -1 if name != "XGBoost" else 1
            grid = GridSearchCV(
                config["model"],
                config["params"],
                cv=cv,
                scoring="f1",
                n_jobs=grid_n_jobs,
                verbose=1
            )
            grid.fit(X_train, y_train)

            best = grid.best_estimator_
            self.models[name] = best

            # Evaluate on test set
            y_pred = best.predict(X_test)
            y_proba = best.predict_proba(X_test)[:, 1]

            result = {
                "Model": name,
                "Best Params": grid.best_params_,
                "CV F1-Score": grid.best_score_,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }
            results.append(result)
            print(f"  Test F1: {result['F1-Score']:.4f} | ROC-AUC: {result['ROC-AUC']:.4f}")

        results_df = pd.DataFrame(results)

        # Identify best model
        best_idx = results_df["F1-Score"].idxmax()
        self.best_model_name = results_df.loc[best_idx, "Model"]
        self.best_model = self.models[self.best_model_name]
        print(f"\n✓ Best model: {self.best_model_name}")

        self.results = results_df
        return results_df

    def save_model(self, path: str = "models/best_model.joblib"):
        """Save best model to disk."""
        if self.best_model is None:
            raise ValueError("No model trained yet.")
        joblib.dump(self.best_model, path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path: str) -> object:
        """Load model from disk."""
        return joblib.load(path)

if __name__ == "__main__":
    print("ModelTrainer class defined.")
