"""
data_acquisition.py
Handles loading and harmonising Twitter/X and Facebook brand post datasets.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def load_facebook_metrics(path: str = "data/raw/facebook_metrics.csv") -> pd.DataFrame:
    """Load UCI Facebook Metrics dataset."""
    df = pd.read_csv(path)
    df["platform"] = "Facebook"
    # Standardise column names
    df = df.rename(columns={
        "Page total likes": "likes",
        "comment": "comments",
        "share": "shares",
        "Post Hour": "posting_hour",
        "Post Weekday": "posting_day",
    })
    df["retweets"] = 0
    df["replies"] = 0
    return df

def load_twitter_corpus(path: str = "data/raw/twitter_corpus.csv") -> pd.DataFrame:
    """Load Twitter/X brand interaction corpus."""
    df = pd.read_csv(path)
    df["platform"] = "Twitter"
    df = df.rename(columns={
        "favorite_count": "likes",
        "retweet_count": "retweets",
        "reply_count": "replies",
    })
    df["comments"] = 0
    df["shares"] = 0
    return df

def harmonise_schemas(fb_df: pd.DataFrame, tw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Facebook and Twitter datasets into unified schema."""
    common_cols = ["text", "likes", "comments", "shares", "retweets", "replies", 
                   "timestamp", "platform"]
    fb_subset = fb_df[[c for c in common_cols if c in fb_df.columns]]
    tw_subset = tw_df[[c for c in common_cols if c in tw_df.columns]]
    merged = pd.concat([fb_subset, tw_subset], ignore_index=True)
    return merged

def create_synthetic_fallback(n_posts: int = 11000, 
                               start_date: str = "2023-01-01",
                               end_date: str = "2023-12-31") -> pd.DataFrame:
    """Generate schema-identical synthetic data when live sources unavailable."""
    np.random.seed(42)
    platforms = np.random.choice(["Twitter", "Facebook"], size=n_posts)

    data = {
        "text": [f"Sample post content {i}" for i in range(n_posts)],
        "likes": np.random.poisson(lam=150, size=n_posts),
        "comments": np.random.poisson(lam=25, size=n_posts),
        "shares": np.random.poisson(lam=15, size=n_posts),
        "retweets": np.random.poisson(lam=30, size=n_posts),
        "replies": np.random.poisson(lam=10, size=n_posts),
        "timestamp": pd.date_range(start=start_date, end=end_date, periods=n_posts),
        "platform": platforms,
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Example usage
    synth = create_synthetic_fallback(n_posts=100)
    print(f"Generated {len(synth)} synthetic posts")
    print(synth.head())
