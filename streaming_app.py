"""
streaming_app.py — Real-time + Batch serving layer for the dissertation project
"Predicting Social Media Engagement and Brand-Building Outcomes Using XGBoost,
Sentiment Analysis, and Interactive Power BI Dashboards".

WHY THIS FILE EXISTS
---------------------
Power BI can visualise data beautifully but it CANNOT execute live Python
against user-typed input or a freshly uploaded CSV — it is a reporting layer,
not an inference layer. This Streamlit app IS the actual interactive,
real-time part of the system. Power BI (Part 11) reads the CSVs this
pipeline/app produces; it does not run the model itself.

MODES
-----
1. Single-post real-time scoring: type a post, get predicted engagement
   class, confidence, sentiment, and a transparent rule-based
   recommendation.
2. Batch "fuse new data" mode: upload a CSV of new posts, every row is
   scored immediately against the ALREADY-TRAINED model (no retraining),
   with a results table, bar chart, and CSV download.

Run with:  streamlit run streaming_app.py
Expects the following artefacts to exist alongside this file's ../models,
../feature_engineering, ../sentiment directories (produced by the main
notebook): XGBoost.joblib (or best model), scaler.joblib, and the fitted
TF-IDF+SVD (or SBERT) embedder + feature metadata, loaded via
serving_artifacts.pkl (built by Part 9 of the notebook).
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Social Media Engagement Predictor", layout="wide")

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "serving_artifacts.pkl")

# -----------------------------------------------------------------------------
# Load trained artefacts (model, scaler, embedder, feature stats, rules)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)

try:
    ART = load_artifacts()
    LOAD_OK = True
except Exception as e:
    LOAD_OK = False
    LOAD_ERR = str(e)

STOPWORDS = set("""
a an the and or but if while is are was were be been being of at by for with
about against between into through during before after above below to from
up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not
only own same so than too very s t can will just don should now i me my
myself we our ours you your yours he him his she her it its they them their
this that these those am do does did doing having have has had
""".split())
URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
PUNCT_RE = re.compile(r"[^\w\s]")
SUFFIXES = ["ing", "edly", "ies", "ied", "es", "ed", "ly", "s"]

def simple_lemmatize(token):
    for suf in SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token

def clean_text(raw):
    t = str(raw).lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    tokens = [tok for tok in t.split() if tok not in STOPWORDS and len(tok) > 1]
    return " ".join(simple_lemmatize(tok) for tok in tokens)


def build_feature_vector(text, posting_hour, has_media, platform, hashtags, mentions):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    clean = clean_text(text)
    tfidf_vec = ART["tfidf_vectorizer"].transform([clean])
    embedding = ART["svd_model"].transform(tfidf_vec)[0]

    post_length_chars = len(clean)
    word_count = max(len(clean.split()), 1)
    avg_word_len = post_length_chars / word_count
    hashtag_count = len(str(hashtags).split()) if str(hashtags).strip() else 0
    mention_count = len(str(mentions).split()) if str(mentions).strip() else 0
    posting_day = pd.Timestamp.now().dayofweek
    is_weekend = int(posting_day in [5, 6])
    media_indicator = int(bool(has_media))
    platform_twitter = int(str(platform).lower() == "twitter")

    means, stds = ART["structural_means"], ART["structural_stds"]
    def z(name, val):
        return (val - means[name]) / stds[name]

    structural_vec = [
        z("post_length_chars", post_length_chars), z("word_count", word_count),
        z("avg_word_len", avg_word_len), z("hashtag_count", hashtag_count),
        z("mention_count", mention_count), z("posting_hour", posting_hour),
        z("posting_day", posting_day),
    ]
    extra_vec = [is_weekend, media_indicator, platform_twitter]
    vs = analyzer.polarity_scores(text)
    sentiment_vec = [vs["neg"], vs["neu"], vs["pos"], vs["compound"]]

    full_vec = np.array(structural_vec + extra_vec + sentiment_vec + list(embedding),
                         dtype=np.float64).reshape(1, -1)
    return full_vec, vs, hashtag_count, media_indicator


def predict(full_vec):
    model = ART["model"]
    proba = model.predict_proba(full_vec)[0]
    pred_class = "High Engagement" if proba[1] >= 0.5 else "Low Engagement"
    return pred_class, float(max(proba))


def recommend(vs, hashtag_count, posting_hour, media_indicator, pred_class):
    """Transparent, rule-based prescriptive recommendation — kept in sync
    with the pre-publish content-flag rules exported to Power BI (Part 11)."""
    tips = []
    if vs["compound"] <= -0.4:
        tips.append("⚠️ Strongly negative tone detected — consider a crisis-response review before posting.")
    if hashtag_count == 0:
        tips.append("No hashtag detected — adding 1–2 relevant hashtags is associated with higher reach.")
    if hashtag_count > 4:
        tips.append("Hashtag count is high — over-tagging can look spammy and may suppress reach.")
    if not (17 <= posting_hour <= 22 or 8 <= posting_hour <= 11):
        tips.append(f"Posted at hour {posting_hour} — historical data shows low-activity windows outside "
                     f"08:00–11:00 and 17:00–22:00.")
    if media_indicator == 0:
        tips.append("No image/video attached — media posts trend toward higher engagement in this corpus.")
    if pred_class == "Low Engagement" and not tips:
        tips.append("No obvious red flags — engagement may simply be topic-dependent; consider A/B testing copy.")
    if not tips:
        tips.append("✅ Looks good — strong tone, timing and structure align with historically high-engagement posts.")
    return tips


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("📈 Social Media Engagement Predictor")
st.caption("XGBoost + Sentiment Analysis serving layer — real-time scoring for new, unpublished posts. "
           "This app is the live-interactivity counterpart to the Power BI dashboards, which are "
           "historical/descriptive-predictive reporting only (Power BI cannot execute live Python).")

if not LOAD_OK:
    st.error(f"Could not load trained model artefacts from `{ARTIFACT_PATH}`.\n\n"
             f"Run the main notebook through Part 9 first to generate `serving_artifacts.pkl`.\n\n"
             f"Error: {LOAD_ERR}")
    st.stop()

tab1, tab2 = st.tabs(["🔮 Single-Post Real-Time Scoring", "📦 Batch: Fuse New Data"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        text = st.text_area("Post text", height=120,
                             placeholder="e.g. Absolutely love the new product launch! #excited")
        hashtags = st.text_input("Hashtags (space-separated, optional)", "")
        mentions = st.text_input("Mentions (space-separated, optional)", "")
    with col2:
        platform = st.selectbox("Platform", ["Twitter", "Facebook"])
        posting_hour = st.slider("Posting hour (24h)", 0, 23, 12)
        has_media = st.checkbox("Includes image/video", value=False)
        run_btn = st.button("Predict engagement", type="primary", use_container_width=True)

    if run_btn:
        if not text.strip():
            st.warning("Please enter some post text.")
        else:
            full_vec, vs, hcount, media_ind = build_feature_vector(
                text, posting_hour, has_media, platform, hashtags, mentions)
            pred_class, confidence = predict(full_vec)
            tone = "Positive" if vs["compound"] >= 0.05 else ("Negative" if vs["compound"] <= -0.05 else "Neutral")

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", pred_class)
            m2.metric("Confidence", f"{confidence:.1%}")
            m3.metric("Sentiment", tone, f"compound={vs['compound']:.2f}")

            st.subheader("Prescriptive recommendations")
            for tip in recommend(vs, hcount, posting_hour, media_ind, pred_class):
                st.write("- " + tip)

with tab2:
    st.write("Upload a CSV of new, unpublished posts. Every row is scored **immediately** against the "
             "already-trained model — no retraining happens here.")
    st.caption("Required columns: `text`. Optional: `posting_hour`, `has_media`, `platform`, `hashtags`, `mentions`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "text" not in batch_df.columns:
            st.error("CSV must contain a `text` column.")
        else:
            with st.spinner(f"Scoring {len(batch_df)} posts..."):
                preds, confs, tones = [], [], []
                for _, row in batch_df.iterrows():
                    full_vec, vs, hcount, media_ind = build_feature_vector(
                        row.get("text", ""), row.get("posting_hour", 12),
                        row.get("has_media", False), row.get("platform", "Twitter"),
                        row.get("hashtags", ""), row.get("mentions", ""))
                    pred_class, confidence = predict(full_vec)
                    tone = "Positive" if vs["compound"] >= 0.05 else ("Negative" if vs["compound"] <= -0.05 else "Neutral")
                    preds.append(pred_class); confs.append(confidence); tones.append(tone)

                batch_df["predicted_class"] = preds
                batch_df["confidence"] = confs
                batch_df["sentiment_tone"] = tones

            st.success(f"Scored {len(batch_df)} new posts.")
            st.dataframe(batch_df, use_container_width=True)

            fig = px.histogram(batch_df, x="predicted_class", color="sentiment_tone",
                                title="Predicted Engagement Distribution — New Batch",
                                barmode="group")
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download scored results (CSV)", csv_bytes,
                                file_name="scored_new_posts.csv", mime="text/csv")

st.divider()
st.caption("Serving layer for: Predicting Social Media Engagement and Brand-Building Outcomes Using "
           "XGBoost, Sentiment Analysis, and Interactive Power BI Dashboards (MSc Dissertation).")
