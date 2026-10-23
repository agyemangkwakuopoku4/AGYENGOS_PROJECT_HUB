"""
explainability.py
SHAP-based model explainability for engagement prediction.
"""
import numpy as np
import pandas as pd
try:
    import shap
    _SHAP_AVAILABLE = True
except Exception:
    shap = None
    _SHAP_AVAILABLE = False
    print("Warning: shap not installed — using feature-importance fallback for explainability.")
import matplotlib.pyplot as plt
from typing import Optional

class SHAPExplainer:
    """Post-hoc explainability using SHAP (SHapley Additive exPlanations)."""

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit_explainer(self, X_background: pd.DataFrame):
        """Initialise TreeSHAP explainer."""
        if _SHAP_AVAILABLE and shap is not None:
            self.explainer = shap.TreeExplainer(self.model)
            print("✓ TreeSHAP explainer initialised")
        else:
            # Fallback: rely on model feature_importances_ if available
            self.explainer = None
            if hasattr(self.model, "feature_importances_"):
                print("✓ Using model.feature_importances_ as fallback importance")
            else:
                print("⚠ No SHAP and no feature_importances_ available; explainability will be limited")

    def compute_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for given instances."""
        if self.explainer is None:
            # Fallback behaviour: try to use model.feature_importances_
            if hasattr(self.model, "feature_importances_"):
                fi = np.asarray(self.model.feature_importances_)
                # Create a sample-wise matrix with importances replicated
                self.shap_values = np.tile(fi, (len(X), 1))
                return self.shap_values

            raise ValueError("Explainer not fitted or no fallback available. Call fit_explainer first.")

        raw = self.explainer.shap_values(X)

        # Normalise several possible SHAP return shapes into a single
        # array of shape (n_samples, n_features) or a list [class0, class1].
        if isinstance(raw, list):
            # Typical case for classifier: list of arrays per class
            self.shap_values = raw
            return self.shap_values

        arr = np.asarray(raw)

        # If arr is 3D (n_samples, n_features, n_classes) choose class 1
        if arr.ndim == 3:
            # try last axis as classes
            if arr.shape[2] >= 2:
                self.shap_values = arr[:, :, 1]
                return self.shap_values
            # otherwise collapse last axis
            self.shap_values = arr.mean(axis=2)
            return self.shap_values

        # If arr is 2D but has twice the feature count, try to split in half
        n_feats = len(self.feature_names)
        if arr.ndim == 2 and arr.shape[1] == 2 * n_feats:
            # assume concatenated class0|class1 -> take second half (class1)
            self.shap_values = arr[:, n_feats:]
            return self.shap_values

        # Otherwise, use as-is
        self.shap_values = arr
        return self.shap_values

    def global_importance(self, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
        """
        Compute global feature importance (mean absolute SHAP).

        Returns:
            DataFrame with feature names and mean |SHAP| values.
        """
        if self.shap_values is None:
            self.compute_values(X)

        # For binary classification, use values for class 1
        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values

        importance = np.abs(sv).mean(axis=0)
        # Ensure importance is 1-D
        importance = np.asarray(importance).ravel()

        # Align feature names and importance length
        if len(self.feature_names) != importance.shape[0]:
            print(f"Warning: feature_names length ({len(self.feature_names)}) != importance length ({importance.shape[0]}). Truncating/expanding to match.")
            min_len = min(len(self.feature_names), importance.shape[0])
            features = list(self.feature_names)[:min_len]
            importance = importance[:min_len]
        else:
            features = list(self.feature_names)

        imp_df = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": importance
        }).sort_values("mean_abs_shap", ascending=False)

        return imp_df

    def plot_summary(self, X: pd.DataFrame, max_display: int = 20, 
                     save_path: Optional[str] = None):
        """Generate and optionally save SHAP summary plot."""
        if self.shap_values is None:
            self.compute_values(X)

        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values

        if _SHAP_AVAILABLE and shap is not None:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(sv, X, feature_names=self.feature_names, 
                             max_display=max_display, show=False)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Summary plot saved to {save_path}")
            plt.close()
        else:
            # Fallback: simple bar plot of mean absolute importances
            importance = np.abs(sv).mean(axis=0)
            imp_df = pd.DataFrame({"feature": self.feature_names, "mean_abs": importance})
            imp_df = imp_df.sort_values("mean_abs", ascending=False).head(max_display)
            plt.figure(figsize=(8, max(4, 0.3 * len(imp_df))))
            plt.barh(imp_df["feature"][::-1], imp_df["mean_abs"][::-1])
            plt.xlabel("Mean abs importance")
            plt.title("Feature importance (fallback)")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Summary plot saved to {save_path}")
            plt.close()

    def plot_waterfall(self, X: pd.DataFrame, instance_idx: int = 0,
                       save_path: Optional[str] = None):
        """Generate waterfall plot for single prediction."""
        if self.shap_values is None:
            self.compute_values(X)

        sv = self.shap_values[1] if isinstance(self.shap_values, list) else self.shap_values

        if _SHAP_AVAILABLE and shap is not None:
            plt.figure(figsize=(12, 6))
            shap.waterfall_plot(shap.Explanation(
                values=sv[instance_idx],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                data=X.iloc[instance_idx],
                feature_names=self.feature_names
            ), show=False)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Waterfall plot saved to {save_path}")
            plt.close()
        else:
            # Fallback: show top features as bar chart for the single instance
            vals = sv[instance_idx]
            df = pd.DataFrame({"feature": self.feature_names, "value": vals})
            df = df.reindex(df["value"].abs().sort_values(ascending=False).index).head(20)
            plt.figure(figsize=(10, max(4, 0.3 * len(df))))
            plt.barh(df["feature"][::-1], df["value"][::-1])
            plt.xlabel("Importance (signed)")
            plt.title("Instance importance (fallback)")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✓ Waterfall plot saved to {save_path}")
            plt.close()

    def detect_target_leakage(self, X: pd.DataFrame, threshold_ratio: float = 5.0) -> bool:
        """
        Detect potential target leakage by checking if one feature
        dominates SHAP importance disproportionately.

        Returns:
            True if potential leakage detected.
        """
        imp_df = self.global_importance(X)
        top_feature = imp_df.iloc[0]
        second_feature = imp_df.iloc[1]

        ratio = top_feature["mean_abs_shap"] / second_feature["mean_abs_shap"]
        is_leakage = ratio > threshold_ratio

        if is_leakage:
            print(f"⚠ Target leakage suspected: '{top_feature['feature']}' "
                  f"dominates with ratio {ratio:.1f}x over next feature.")
        else:
            print(f"✓ No target leakage detected. Top feature ratio: {ratio:.1f}x")

        return is_leakage

if __name__ == "__main__":
    print("SHAPExplainer class defined.")
