"""
feature_engineering.py
Constructs structural, temporal, and interaction-based features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    """Engineers features from cleaned text and metadata."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.fitted = False

    def compute_structural(self, cleaned_texts: List[str], metadata: List[dict]) -> pd.DataFrame:
        """Compute structural text features."""
        features = []
        for text, meta in zip(cleaned_texts, metadata):
            words = text.split()
            features.append({
                "post_length_chars": len(text),
                "word_count": max(len(words), 1),
                "avg_word_len": len(text) / max(len(words), 1),
                "hashtag_count": meta.get("hashtag_count", 0),
                "mention_count": meta.get("mention_count", 0),
            })
        return pd.DataFrame(features)

    def compute_temporal(self, timestamps: pd.Series) -> pd.DataFrame:
        """Compute temporal features from timestamps."""
        dt = pd.to_datetime(timestamps)
        features = pd.DataFrame({
            "posting_hour": dt.dt.hour,
            "posting_day": dt.dt.dayofweek,
            "is_weekend": dt.dt.dayofweek.isin([5, 6]).astype(int),
        })
        # One-hot encode day of week
        day_dummies = pd.get_dummies(dt.dt.dayofweek, prefix="posting_day")
        features = pd.concat([features, day_dummies], axis=1)
        return features

    def compute_interaction_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Weighted composite interaction score.
        Comments and shares weighted higher than likes (greater user effort).
        """
        score = (
            0.20 * df.get("likes", 0) +
            0.35 * df.get("comments", 0) +
            0.35 * df.get("shares", 0) +
            0.05 * df.get("retweets", 0) +
            0.05 * df.get("replies", 0)
        )
        return score

    def create_target(self, interaction_score: pd.Series) -> pd.Series:
        """Binary engagement label via median split."""
        median = interaction_score.median()
        return (interaction_score >= median).astype(int)

    def fit_transform(self, df: pd.DataFrame, cleaned_texts: List[str], 
                      metadata: List[dict]) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        structural = self.compute_structural(cleaned_texts, metadata)
        temporal = self.compute_temporal(df["timestamp"])

        interaction_score = self.compute_interaction_score(df)
        engagement_label = self.create_target(interaction_score)

        # Platform encoding
        platform_encoded = pd.get_dummies(df["platform"], prefix="platform")

        # Combine all features
        features = pd.concat([
            structural.reset_index(drop=True),
            temporal.reset_index(drop=True),
            platform_encoded.reset_index(drop=True),
        ], axis=1)

        features["interaction_score"] = interaction_score.values
        features["engagement_label"] = engagement_label.values

        # Scale structural features
        structural_cols = ["post_length_chars", "word_count", "avg_word_len", 
                          "hashtag_count", "mention_count"]
        features[structural_cols] = self.scaler.fit_transform(features[structural_cols])
        self.fitted = True

        return features

    def transform(self, df: pd.DataFrame, cleaned_texts: List[str],
                  metadata: List[dict]) -> pd.DataFrame:
        """Transform new data using fitted scaler."""
        if not self.fitted:
            raise ValueError("Engineer must be fitted before transform.")
        # Same as fit_transform but uses fitted scaler
        structural = self.compute_structural(cleaned_texts, metadata)
        temporal = self.compute_temporal(df["timestamp"])
        platform_encoded = pd.get_dummies(df["platform"], prefix="platform")

        features = pd.concat([
            structural.reset_index(drop=True),
            temporal.reset_index(drop=True),
            platform_encoded.reset_index(drop=True),
        ], axis=1)

        structural_cols = ["post_length_chars", "word_count", "avg_word_len",
                          "hashtag_count", "mention_count"]
        features[structural_cols] = self.scaler.transform(features[structural_cols])
        return features

if __name__ == "__main__":
    print("FeatureEngineer class defined.")
