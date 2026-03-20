"""
utils.py
--------
Shared utilities for the Fake News Detection project:
  - Text preprocessing (cleaning, stopword removal)
  - Tokenizer save/load helpers
  - Fact-check rule-based scoring
  - Combined prediction helper
"""

import re
import string
import pickle
import os

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore



MAX_VOCAB_SIZE = 20000   # maximum number of unique tokens
MAX_SEQ_LENGTH = 500     # maximum sequence length after padding
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")

FAKE_NEWS_KEYWORDS = [
    "breaking", "shocking", "viral", "must see", "you won't believe",
    "exclusive", "secret", "exposed", "hoax", "conspiracy", "unbelievable",
    "bombshell", "scandal", "they don't want you to know", "urgent",
    "warning", "alert", "leaked", "miracle", "cure", "banned",
]

MAX_KEYWORD_HITS = 5


def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs, punctuation, extra whitespace.
    Returns a cleaned string ready for tokenisation.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text



def save_tokenizer(tokenizer, path: str = TOKENIZER_PATH) -> None:
    """Persist a fitted Keras Tokenizer to disk."""
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: str = TOKENIZER_PATH):
    """Load a previously saved Keras Tokenizer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)



def texts_to_padded_sequences(texts, tokenizer, maxlen: int = MAX_SEQ_LENGTH):
    """
    Convert a list of raw text strings to a padded numpy array.

    Parameters
    ----------
    texts : list[str]
    tokenizer : keras Tokenizer (already fitted)
    maxlen : int

    Returns
    -------
    np.ndarray  shape (n, maxlen)
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")



def fact_check_score(text: str) -> float:
    """
    Return a suspicion score in [0.0, 1.0] based on the presence of
    sensationalist / fake-news keywords.

    A higher score means the text looks *more* like fake news according
    to simple heuristics.
    """
    lowered = text.lower()
    hits = sum(1 for kw in FAKE_NEWS_KEYWORDS if kw in lowered)
    return min(hits / MAX_KEYWORD_HITS, 1.0)



def combined_prediction(ml_prob: float, text: str, alpha: float = 0.15):
    """
    Blend the ML model's fake-news probability with the rule-based score.

    Parameters
    ----------
    ml_prob : float   Model output in [0, 1]; higher = more likely FAKE
                      (matches label encoding: 1 = FAKE, 0 = REAL).
    text    : str     Original (uncleaned) news text.
    alpha   : float   Weight given to the rule-based score (0–1).
                      Default 0.15 gives the ML model 85% of the influence
                      while still allowing sensationalist-keyword signals
                      to nudge borderline predictions.

    Returns
    -------
    label      : str   "FAKE" or "REAL"
    confidence : float percentage confidence in the returned label
    """
    rb_score = fact_check_score(text)
    blended = (1 - alpha) * ml_prob + alpha * rb_score
    label = "FAKE" if blended >= 0.5 else "REAL"
    confidence = 50.0 + abs(blended - 0.5) * 100.0
    return label, round(confidence, 1)
