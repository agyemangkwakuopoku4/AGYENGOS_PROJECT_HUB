"""
mock_artifact_generator.py
Generates mock model artefacts for testing the Streamlit app without
running the full training pipeline.
"""
import os
import pickle
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def generate_mock_artifacts(output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    # Create diverse vocabulary
    base_words = [
        "amazing", "terrible", "great", "bad", "good", "love", "hate", "excited",
        "disappointed", "happy", "sad", "angry", "joyful", "worried", "thrilled",
        "product", "service", "launch", "update", "event", "offer", "deal", "sale",
        "customer", "brand", "company", "team", "support", "help", "thanks",
        "new", "old", "best", "worst", "first", "last", "next", "previous",
        "today", "tomorrow", "yesterday", "now", "soon", "later", "early", "late",
        "big", "small", "huge", "tiny", "large", "little", "massive", "mini",
        "fast", "slow", "quick", "rapid", "instant", "delayed", "speedy", "swift",
        "beautiful", "ugly", "pretty", "handsome", "elegant", "plain", "fancy", "simple",
        "strong", "weak", "powerful", "fragile", "tough", "delicate", "hard", "soft",
        "hot", "cold", "warm", "cool", "freezing", "burning", "chilly", "mild",
        "bright", "dark", "shiny", "dull", "clear", "cloudy", "sunny", "rainy"
    ]

    all_words = base_words[:]
    for i in range(600):
        w1 = random.choice(base_words)
        w2 = random.choice(base_words)
        all_words.append(f"{w1}_{w2}_{i}")
    all_words = list(set(all_words))

    # Generate training texts
    dummy_texts = []
    for i in range(300):
        text = " ".join(random.sample(all_words, k=random.randint(8, 20)))
        dummy_texts.append(text)

    # Fit TF-IDF
    tfidf = TfidfVectorizer(max_features=500, min_df=1)
    tfidf.fit(dummy_texts)
    tfidf_matrix = tfidf.transform(dummy_texts)

    # Fit SVD
    svd = TruncatedSVD(n_components=384, random_state=42)
    embedding = svd.fit_transform(tfidf_matrix)

    # Create classifier with exact feature dimensions
    # App structure: 7 structural + 3 extra + 4 sentiment + embedding dims
    # NOTE: TruncatedSVD can silently fit fewer components than requested
    # when n_samples/n_features are small, so size off the actual fitted
    # embedding width (embedding.shape[1]), not the requested n_components —
    # otherwise the classifier and the embedder disagree on feature count
    # at inference time.
    n_samples = 300
    n_features = 7 + 3 + 4 + embedding.shape[1]
    X_full = np.random.randn(n_samples, n_features)
    labels = [random.choice([0, 1]) for _ in range(n_samples)]

    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_full, labels)

    # Normalisation parameters
    means = {
        "post_length_chars": 120.0, "word_count": 15.0, "avg_word_len": 5.5,
        "hashtag_count": 2.0, "mention_count": 1.0, "posting_hour": 12.0, "posting_day": 3.0,
    }
    stds = {
        "post_length_chars": 40.0, "word_count": 8.0, "avg_word_len": 2.0,
        "hashtag_count": 2.5, "mention_count": 2.0, "posting_hour": 6.0, "posting_day": 2.0,
    }

    artefacts = {
        "model": model,
        "tfidf_vectorizer": tfidf,
        "svd_model": svd,
        "structural_means": means,
        "structural_stds": stds,
    }

    artifact_path = os.path.join(output_dir, "serving_artifacts.pkl")
    with open(artifact_path, "wb") as f:
        pickle.dump(artefacts, f)

    print(f"✓ Mock artefacts saved to {artifact_path}")
    print(f"  Model expects {model.n_features_in_} features")
    return artefacts


if __name__ == "__main__":
    generate_mock_artifacts()
