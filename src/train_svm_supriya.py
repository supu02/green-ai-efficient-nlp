# src/train_svm_supriya.py

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.src.ensemble_svm import WordCharEnsemble

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def build_word_model(
    max_features: int = 30_000,
    ngram_range=(1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    C: float = 1.0,
):
    """Word-level TF-IDF + LinearSVC."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        strip_accents="unicode",
        dtype=np.float32,
    )

    clf = LinearSVC(
        C=C,
        max_iter=5000,
        class_weight=None,
    )

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])


def build_char_model(
    max_features: int = 25_000,
    ngram_range=(3, 5),
    C: float = 0.7,
):
    """
    Char-level TF-IDF + LinearSVC.

    analyzer='char_wb' captures local character patterns
    (punctuation, elongations, emoticons, suffixes…).
    """
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )

    clf = LinearSVC(
        C=C,
        max_iter=5000,
        class_weight=None,
    )

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])


def main():
    print("📥 Loading training data.")
    X, y = load_data()
    X = np.array(X)
    y = np.array(y)
    n_samples = len(X)
    print(f" → {n_samples} samples loaded")

    # 1) Train/val split for ensemble tuning
    print("✂️ Splitting train/val for ensemble tuning (10% val).")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )
    print(f" → Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 2) Train individual models on TRAIN
    print("\n🛠 Training word-level SVM on TRAIN…")
    word_clf = build_word_model()
    word_clf.fit(X_train, y_train)

    print("🛠 Training char-level SVM on TRAIN…")
    char_clf = build_char_model()
    char_clf.fit(X_train, y_train)

    # 3) Compute decision scores on VAL
    print("\n📊 Evaluating on VAL for ensemble weights…")
    word_scores_val = word_clf.decision_function(X_val)
    char_scores_val = char_clf.decision_function(X_val)

    # Standardize scale (std on val set)
    scale_word = np.std(word_scores_val) + 1e-8
    scale_char = np.std(char_scores_val) + 1e-8

    word_z = word_scores_val / scale_word
    char_z = char_scores_val / scale_char

    # Simple grid search for weights
    candidate_ws = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_w = 0.6
    best_acc = -1.0

    for w in candidate_ws:
        scores = w * word_z + (1.0 - w) * char_z
        preds = (scores > 0).astype(int)
        acc = accuracy_score(y_val, preds)
        print(f"   w_word={w:.1f} → Val accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_w = w

    print("\n🏆 Best ensemble weight on VAL:")
    print(f"   w_word={best_w:.2f}, w_char={1.0 - best_w:.2f}")
    print(f"   Ensemble VAL accuracy: {best_acc:.4f}")

    # 4) Retrain both models on FULL data
    print("\n🚀 Retraining word-level SVM on FULL data…")
    word_clf_full = build_word_model()
    word_clf_full.fit(X, y)

    print("🚀 Retraining char-level SVM on FULL data…")
    char_clf_full = build_char_model()
    char_clf_full.fit(X, y)

    # Recompute scales on FULL data (for more stable normalization)
    word_scores_full = word_clf_full.decision_function(X)
    char_scores_full = char_clf_full.decision_function(X)
    scale_word_full = np.std(word_scores_full) + 1e-8
    scale_char_full = np.std(char_scores_full) + 1e-8

    # 5) Build ensemble object and save it as "svm_model.pkl"
    ensemble = WordCharEnsemble(
        word_clf=word_clf_full,
        char_clf=char_clf_full,
        w_word=best_w,
        w_char=1.0 - best_w,
        scale_word=scale_word_full,
        scale_char=scale_char_full,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ensemble, f)

    print(f"\n💾 Ensemble SVM model saved to {MODEL_PATH}")
    print(f"📏 Model size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
