"""
model.py
--------
TensorFlow / Keras model definition for Fake News Detection.

Architecture:
  Embedding → SpatialDropout1D → LSTM → Dropout → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Embedding,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
)

from utils import MAX_VOCAB_SIZE, MAX_SEQ_LENGTH


def build_model(
    vocab_size: int = MAX_VOCAB_SIZE,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build and compile the LSTM-based fake-news classifier.

    Parameters
    ----------
    vocab_size    : int   Size of the vocabulary (number of unique tokens + 1).
    embedding_dim : int   Dimension of the embedding vectors.
    lstm_units    : int   Number of LSTM memory units.
    dropout_rate  : float Dropout fraction applied after the LSTM layer.

    Returns
    -------
    model : tf.keras.Model   Compiled and ready-to-train model.
    """
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="embedding",
            ),
            SpatialDropout1D(dropout_rate, name="spatial_dropout"),
            LSTM(lstm_units, name="lstm"),
            Dropout(dropout_rate, name="dropout"),
            Dense(1, activation="sigmoid", name="output"),
        ],
        name="fake_news_detector",
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
