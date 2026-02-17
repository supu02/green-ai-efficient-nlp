# src/svm_model.py

import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "svm_model.pkl"


class SVMModel:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SVM model not found at {MODEL_PATH}. Did you run train_svm.py?"
            )
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text: str) -> int:
        return int(self.model.predict([text])[0])
