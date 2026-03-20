"""
train.py
--------
End-to-end training script for the Fake News Detection model.

Usage
-----
    python train.py

Expects the Kaggle fake-news dataset CSVs inside the ./data/ directory:
    data/Fake.csv   (label = 0)
    data/True.csv   (label = 1)

After training the script saves:
    model.h5          – trained Keras model
    tokenizer.pkl     – fitted Keras Tokenizer
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

from utils import (
    clean_text,
    save_tokenizer,
    texts_to_padded_sequences,
    MAX_VOCAB_SIZE,
    MAX_SEQ_LENGTH,
)
from model import build_model


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAKE_CSV = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATA_DIR, "True.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

EPOCHS = 7
BATCH_SIZE = 64
TEST_SIZE = 0.20
RANDOM_STATE = 42



def load_dataset():
    """Load Fake.csv and True.csv, combine, and return X, y arrays."""
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        print(
            "\n[ERROR] Dataset not found!\n"
            f"  Expected:\n    {FAKE_CSV}\n    {TRUE_CSV}\n\n"
            "  Please download the Kaggle 'Fake and real news dataset' and place\n"
            "  Fake.csv and True.csv inside the data/ directory.\n"
            "  See data/README.md for instructions.\n"
        )
        sys.exit(1)

    df_fake = pd.read_csv(FAKE_CSV)
    df_real = pd.read_csv(TRUE_CSV)

    df_fake["label"] = 1  # 1 = FAKE
    df_real["label"] = 0  # 0 = REAL

    df = pd.concat([df_fake, df_real], ignore_index=True)

    df["content"] = df.get("title", "").fillna("") + " " + df["text"].fillna("")
    df = df[["content", "label"]].dropna()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df["content"].tolist(), df["label"].tolist()



def preprocess(texts, labels, test_size=TEST_SIZE):
    """Clean text, tokenize, pad, and split into train/test sets."""
    print("Cleaning text...")
    cleaned = [clean_text(t) for t in texts]

    print("Fitting tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")  # OOV = out-of-vocabulary
    tokenizer.fit_on_texts(cleaned)
    save_tokenizer(tokenizer)
    print(f"  Vocabulary size: {len(tokenizer.word_index):,}")

    print("Padding sequences...")
    X = texts_to_padded_sequences(texts, tokenizer, maxlen=MAX_SEQ_LENGTH)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, tokenizer



def train(X_train, y_train, X_test, y_test):
    """Build, compile, and train the model."""
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs...\n")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )
    return model, history



def evaluate(model, X_test, y_test):
    """Print accuracy, precision, recall, F1, confusion matrix."""
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("=" * 50)



if __name__ == "__main__":
    texts, labels = load_dataset()
    X_train, X_test, y_train, y_test, tokenizer = preprocess(texts, labels)
    model, history = train(X_train, y_train, X_test, y_test)
    evaluate(model, X_test, y_test)
    print(f"\nModel saved to: {MODEL_PATH}")
