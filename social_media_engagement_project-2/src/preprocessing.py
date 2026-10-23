"""
preprocessing.py
NLP text cleaning pipeline for social media posts.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
PUNCT_RE = re.compile(r"[^\w\s]")

class TextCleaner:
    """Standardised text preprocessing for social media corpora."""

    def __init__(self):
        self.url_pattern = URL_RE
        self.mention_pattern = MENTION_RE
        self.hashtag_pattern = HASHTAG_RE
        self.punct_pattern = PUNCT_RE
        self.stopwords = STOPWORDS
        self.lemmatizer = LEMMATIZER

    def clean(self, raw_text: str) -> Tuple[str, dict]:
        """
        Clean raw post text and extract metadata.

        Returns:
            cleaned_text: Preprocessed text string
            metadata: Dict with hashtag_count, mention_count
        """
        text = str(raw_text)

        # Extract counts before removal
        hashtags = self.hashtag_pattern.findall(text)
        mentions = self.mention_pattern.findall(text)
        metadata = {
            "hashtag_count": len(hashtags),
            "mention_count": len(mentions),
        }

        # Clean text
        text = text.lower()
        text = self.url_pattern.sub(" ", text)
        text = self.mention_pattern.sub(" ", text)
        text = self.hashtag_pattern.sub(" ", text)
        text = self.punct_pattern.sub(" ", text)

        # Tokenise and filter
        tokens = [tok for tok in text.split() 
                  if tok not in self.stopwords and len(tok) > 1]

        # Lemmatise
        tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens]

        cleaned = " ".join(tokens)
        return cleaned, metadata

    def batch_clean(self, texts: List[str]) -> Tuple[List[str], List[dict]]:
        """Process multiple texts."""
        cleaned_texts = []
        metadata_list = []
        for text in texts:
            c, m = self.clean(text)
            cleaned_texts.append(c)
            metadata_list.append(m)
        return cleaned_texts, metadata_list

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Absolutely LOVE the new #product launch!!! Check it out @brandname https://t.co/xyz"
    cleaned, meta = cleaner.clean(sample)
    print(f"Original: {sample}")
    print(f"Cleaned:  {cleaned}")
    print(f"Metadata: {meta}")
