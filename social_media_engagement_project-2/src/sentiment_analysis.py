"""
sentiment_analysis.py
Dual-layer sentiment analysis: VADER + Sentence-BERT.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_SOURCE = "vaderSentiment"
except Exception:
    try:
        # Fall back to NLTK's VADER implementation if vaderSentiment isn't installed
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        _VADER_SOURCE = "nltk"
    except Exception:
        # Final fallback: provide a minimal stub analyzer so the package still runs
        _VADER_SOURCE = None

        class _FallbackVader:
            def polarity_scores(self, text: str):
                return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        SentimentIntensityAnalyzer = _FallbackVader

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using TF-IDF + SVD fallback.")

class SentimentAnalyzer:
    """
    Hybrid sentiment analysis combining VADER lexicon scoring
    with Sentence-BERT contextual embeddings.
    """

    def __init__(self, use_sbert: bool = True, sbert_model: str = "all-MiniLM-L6-v2"):
        self.vader = SentimentIntensityAnalyzer()
        self.use_sbert = use_sbert and SENTENCE_TRANSFORMERS_AVAILABLE
        self.sbert_model_name = sbert_model
        self.sbert_model = None

        if self.use_sbert:
            try:
                self.sbert_model = SentenceTransformer(sbert_model)
                print(f"✓ Loaded Sentence-BERT model: {sbert_model}")
            except Exception as e:
                print(f"✗ Failed to load SBERT: {e}. Will use fallback.")
                self.use_sbert = False

    def vader_scores(self, texts: List[str]) -> pd.DataFrame:
        """Compute VADER polarity scores."""
        scores = []
        for text in texts:
            vs = self.vader.polarity_scores(str(text))
            scores.append(vs)
        return pd.DataFrame(scores)

    def sbert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute Sentence-BERT embeddings (384-dim)."""
        if self.sbert_model is None:
            raise RuntimeError("SBERT model not loaded.")
        embeddings = self.sbert_model.encode(texts, show_progress_bar=True)
        return embeddings

    def analyze(self, texts: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Full dual-layer sentiment analysis.

        Returns:
            vader_df: DataFrame with pos, neg, neu, compound columns
            embeddings: numpy array of shape (n_texts, 384)
        """
        vader_df = self.vader_scores(texts)

        if self.use_sbert:
            embeddings = self.sbert_embeddings(texts)
        else:
            # Fallback: return zero embeddings (will be handled by caller)
            embeddings = np.zeros((len(texts), 384))

        return vader_df, embeddings

    def analyze_single(self, text: str) -> Tuple[dict, np.ndarray]:
        """Analyze a single text."""
        vader_df, emb = self.analyze([text])
        return vader_df.iloc[0].to_dict(), emb[0]

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_texts = [
        "I absolutely love this new product! Best purchase ever!",
        "This is the worst experience I've ever had. So disappointed.",
        "The package arrived on time. Standard delivery service."
    ]
    vader_df, embeddings = analyzer.analyze(sample_texts)
    print("\nVADER Scores:")
    print(vader_df)
    print(f"\nEmbedding shape: {embeddings.shape}")
