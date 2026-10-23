"""
dashboard_export.py
Export scored data for Power BI dashboard ingestion.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

class DashboardExporter:
    """Exports data in formats optimised for Power BI consumption."""

    def __init__(self, output_dir: str = "data/dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_scored_posts(self, df: pd.DataFrame, filename: str = "scored_posts.csv"):
        """Export scored posts for descriptive/predictive tiers."""
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        print(f"✓ Exported {len(df)} scored posts to {path}")
        return path

    def export_model_metrics(self, metrics_df: pd.DataFrame, 
                            filename: str = "model_metrics.csv"):
        """Export model performance metrics."""
        path = self.output_dir / filename
        metrics_df.to_csv(path, index=False)
        print(f"✓ Exported model metrics to {path}")
        return path

    def export_shap_importance(self, shap_df: pd.DataFrame,
                               filename: str = "shap_importance.csv"):
        """Export SHAP feature importance for dashboard visualisation."""
        path = self.output_dir / filename
        shap_df.to_csv(path, index=False)
        print(f"✓ Exported SHAP importance to {path}")
        return path

    def export_prescriptive_rules(self, rules: list, 
                                  filename: str = "prescriptive_rules.json"):
        """Export prescriptive rules as JSON for dashboard integration."""
        import json
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(rules, f, indent=2)
        print(f"✓ Exported prescriptive rules to {path}")
        return path

    def export_kmeans_segments(self, segments_df: pd.DataFrame,
                               filename: str = "audience_segments.csv"):
        """Export K-Means audience segments for prescriptive tier."""
        path = self.output_dir / filename
        segments_df.to_csv(path, index=False)
        print(f"✓ Exported {len(segments_df)} audience segments to {path}")
        return path

if __name__ == "__main__":
    print("DashboardExporter class defined.")
