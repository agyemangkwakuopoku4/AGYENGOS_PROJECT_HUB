"""
serving_export.py
Fits the TF-IDF+SVD embedder and classifier used by streaming_app.py's live
serving path, and packages them into models/serving_artifacts.pkl.

This is a SEPARATE feature pipeline from feature_engineering.py /
model_training.py (which serve the Power BI / SHAP dashboard export). The
dashboard pipeline z-scores nothing (MinMaxScaler), one-hot encodes every
platform and day-of-week seen in training, and embeds with SBERT. The
serving path in streaming_app.py instead expects an exact fixed-width
vector: 7 z-scored structural fields, 3 raw binary fields, 4 VADER
sentiment scores, then a TF-IDF+SVD embedding -- built live from a single
posting_hour/has_media/platform/hashtags/mentions form input or CSV row.
Reusing the dashboard pipeline's features would not match that contract, so
this module rebuilds the exact vector streaming_app.build_feature_vector
constructs, fits an embedder and classifier against it, and exports
structural_means/structural_stds alongside them for the app's z-scoring.

KNOWN LIMITATION: has_media has no signal anywhere upstream -- it is never
collected by data_acquisition.py for either real or synthetic data. It is
included here as an uncorrelated random binary purely so the trained
classifier's input width matches the serving contract; it will carry
~zero real predictive weight until a genuine media flag is collected.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NOTE: deliberately NOT xgboost for the serving model. XGBoost's compiled
# libxgboost.dylib depends on libomp, which this machine doesn't have
# (no Homebrew installed). Its import happens to succeed inside
# main_pipeline.py because enough other libraries are already loaded by
# that point to satisfy the dlopen -- but it reliably FAILS to import in a
# leaner process, e.g. streaming_app.py loading this pickle directly. That
# failure isn't cosmetic: streaming_app.py's loader silently regenerates a
# random-noise mock artifact on any load exception, clobbering a real
# trained one. RandomForestClassifier has no native-library dependency and
# unpickles the same way everywhere, so it's used here even though
# main_pipeline.py's separate dashboard pipeline may still pick XGBoost
# when available for its own (differently-shaped) model.


def _structural_frame(df, cleaned_texts, metadata):
    """Same 7 fields, same order, as streaming_app.build_feature_vector's
    structural_vec: post_length_chars, word_count, avg_word_len,
    hashtag_count, mention_count, posting_hour, posting_day."""
    dt = pd.to_datetime(df["timestamp"])
    rows = []
    for text, meta in zip(cleaned_texts, metadata):
        word_count = max(len(text.split()), 1)
        post_length_chars = len(text)
        rows.append({
            "post_length_chars": post_length_chars,
            "word_count": word_count,
            "avg_word_len": post_length_chars / word_count,
            "hashtag_count": meta.get("hashtag_count", 0),
            "mention_count": meta.get("mention_count", 0),
        })
    structural = pd.DataFrame(rows)
    structural["posting_hour"] = dt.dt.hour.values
    structural["posting_day"] = dt.dt.dayofweek.values
    return structural[["post_length_chars", "word_count", "avg_word_len",
                        "hashtag_count", "mention_count", "posting_hour",
                        "posting_day"]]


def _extra_frame(df, structural, random_state):
    """3 raw binary fields, same order as extra_vec: is_weekend,
    media_indicator, platform_twitter. media_indicator is synthetic --
    see module docstring."""
    rng = np.random.RandomState(random_state)
    is_weekend = structural["posting_day"].isin([5, 6]).astype(int)
    media_indicator = rng.randint(0, 2, size=len(df))
    platform_twitter = (df["platform"].astype(str).str.lower() == "twitter").astype(int)
    return pd.DataFrame({
        "is_weekend": is_weekend.values,
        "media_indicator": media_indicator,
        "platform_twitter": platform_twitter.values,
    })


def _sentiment_frame(raw_texts):
    """4 VADER scores computed on RAW (uncleaned) text, matching
    build_feature_vector which scores `text`, not `clean`."""
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(str(t)) for t in raw_texts]
    return pd.DataFrame(scores)[["neg", "neu", "pos", "compound"]]


def export_serving_artifacts(df, cleaned_texts, metadata, engagement_label,
                              output_path="models/serving_artifacts.pkl",
                              tfidf_max_features=500, svd_n_components=100,
                              random_state=42):
    """Fit the serving-path embedder + classifier and pickle them to
    output_path in the exact dict shape streaming_app.py expects."""
    structural = _structural_frame(df, cleaned_texts, metadata)
    extra = _extra_frame(df, structural, random_state)
    sentiment = _sentiment_frame(df["text"])

    means = structural.mean().to_dict()
    stds = structural.std(ddof=0).replace(0, 1.0).to_dict()
    structural_z = (structural - structural.mean()) / structural.std(ddof=0).replace(0, 1.0)

    tfidf = TfidfVectorizer(max_features=tfidf_max_features, min_df=2, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(cleaned_texts)

    # TruncatedSVD silently caps n_components below min(n_samples, n_features);
    # size the classifier off the ACTUAL fitted output width, not the request.
    n_components = min(svd_n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(tfidf_matrix)
    emb_cols = [f"emb_{i}" for i in range(embedding.shape[1])]
    embedding_df = pd.DataFrame(embedding, columns=emb_cols, index=structural.index)

    X = pd.concat([
        structural_z.reset_index(drop=True),
        extra.reset_index(drop=True),
        sentiment.reset_index(drop=True),
        embedding_df.reset_index(drop=True),
    ], axis=1)
    y = pd.Series(engagement_label).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_features": X.shape[1],
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "model_type": type(model).__name__,
    }

    artefacts = {
        "model": model,
        "tfidf_vectorizer": tfidf,
        "svd_model": svd,
        "structural_means": means,
        "structural_stds": stds,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(artefacts, f)

    return artefacts, metrics


if __name__ == "__main__":
    print("serving_export module defines export_serving_artifacts(); "
          "run via main_pipeline.py.")
