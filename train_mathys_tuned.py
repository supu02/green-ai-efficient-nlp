import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.mathys import MathysModel, MathysConfig

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "mathys.joblib"


def main():
    print("📥 Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(int).tolist()
    print(f" → {len(X)} samples loaded")

    print("✂️ Train/val split (10% val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )
    print(f" → Train: {len(X_train)} | Val: {len(X_val)}")

    # ---------- petit grid "sobre" ----------
    CANDIDATES = [
        # proche de ton setup actuel
        {"max_features": 20_000, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9, "C": 0.7},
        {"max_features": 20_000, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9, "C": 1.0},

        # vocab un peu plus riche
        {"max_features": 25_000, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9, "C": 0.7},
        {"max_features": 25_000, "ngram_range": (1, 2), "min_df": 2, "max_df": 0.9, "C": 1.0},

        # on teste les trigrams
        {"max_features": 20_000, "ngram_range": (1, 3), "min_df": 2, "max_df": 0.9, "C": 0.7},
        {"max_features": 20_000, "ngram_range": (1, 3), "min_df": 2, "max_df": 0.9, "C": 1.0},
    ]

    best_acc = -1.0
    best_cfg_dict = None

    print("\n🔎 Hyperparameter search (offline)...")
    for i, cfg in enumerate(CANDIDATES, start=1):
        print(f"\n  Candidate {i}/{len(CANDIDATES)}: {cfg}")

        config = MathysConfig(
            max_features=cfg["max_features"],
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            C=cfg["C"],
        )

        model = MathysModel(model_path=None, config=config)  # ne charge pas de modèle disque
        model.train(X_train, y_train)

        y_val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"   → Val accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_cfg_dict = cfg

    print("\n🏆 Best config found:")
    print(f"   {best_cfg_dict}")
    print(f"   Validation accuracy: {best_acc:.4f}")

    # ---------- retrain sur TOUT le dataset ----------
    print("\n🚀 Training BEST MathysModel on FULL data...")
    best_config = MathysConfig(
        max_features=best_cfg_dict["max_features"],
        ngram_range=best_cfg_dict["ngram_range"],
        min_df=best_cfg_dict["min_df"],
        max_df=best_cfg_dict["max_df"],
        C=best_cfg_dict["C"],
    )

    best_model = MathysModel(model_path=None, config=best_config)
    best_model.train(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_model.save(str(MODEL_PATH))

    print(f"💾 Final model saved to {MODEL_PATH}")
    print(f"✅ This is the one loaded by evaluation.py")


if __name__ == "__main__":
    main()
