"""
WacilModel : TF-IDF + LinearSVC pour le Green AI Hack

- Entraînement sur ./data/train.csv
- Bon compromis accuracy / coût (modèle linéaire, TF-IDF sparse)
- Interface simple :
    - En local : lancer `python src/wacil.py` pour entraîner + sauvegarder
    - En prod (CI) : `from src.wacil import WacilModel` puis `model.predict(text)`
"""

from __future__ import annotations
#Hello word
import os
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# === chemins par défaut ===
DEFAULT_TRAIN_PATH = "./data/train.csv"               # ton vrai train
DEFAULT_MODEL_PATH = "./models/wacil_tfidf_logreg.joblib"  # on garde le même nom pour la CI/DVC


TextLike = Union[str, Sequence[str]]


@dataclass
class WacilConfig:
    # Tu peux tweaker ces hyperparams pour gratter de l'accuracy
    max_features: int = 50000        # au lieu de 40000
    ngram_range: tuple = (1, 3)       # 1-3, pas 1-4 (1, 2)
    min_df: int = 3                   # on vire un peu plus de bruit
    max_df: float = 0.9               # on vire aussi les mots trop fréquents
    C: float = 1                   # régularisation un peu plus forte


class WacilModel:
    """
    Modèle "baseline ++" pour le hackathon :
    - Vectorisation TF-IDF (unigrams + bigrams)
    - Classifieur LinearSVC
    - Sauvegarde + chargement via joblib
    """

    def __init__(
        self,
        model_path: Optional[str] = DEFAULT_MODEL_PATH,
        config: Optional[WacilConfig] = None,
        load: bool = True,
    ):
        self.config = config or WacilConfig()
        self.pipeline: Optional[Pipeline] = None

        # En mode inference (CI / evaluation.py), on veut charger un modèle déjà entraîné
        if load:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained model not found at {model_path}. "
                    "Train it once locally by running `python src/wacil.py`."
                )
            self.load(model_path)

    # -------------------------------------------------------------------------
    # Construction du pipeline sklearn
    # -------------------------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=True,
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",          # 🔥 very important
        )

        clf = LinearSVC(
            C=self.config.C,
        )

        pipe = Pipeline(
            steps=[
                ("tfidf", tfidf),
                ("clf", clf),
            ]
        )
        return pipe

    # -------------------------------------------------------------------------
    # Entraînement à partir de X, y
    # -------------------------------------------------------------------------
    def train(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))

    # -------------------------------------------------------------------------
    # Prédiction
    # -------------------------------------------------------------------------
    def predict(self, x: TextLike) -> Union[int, List[int]]:
        """
        - entrée str      -> sortie int (0 ou 1)
        - entrée séquence -> liste d'ints
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded / trained (pipeline is None).")

        if isinstance(x, str):
            preds = self.pipeline.predict([x])
            return int(preds[0])

        preds = self.pipeline.predict(list(x))
        return [int(p) for p in preds]

    # -------------------------------------------------------------------------
    # Sauvegarde / chargement
    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Sauvegarde uniquement le pipeline sklearn (TF-IDF + LinearSVC).
        Pas de WacilConfig dans le fichier => pas de problème de pickle.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save: pipeline is None.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        """
        Charge uniquement le pipeline sklearn.
        """
        self.pipeline = joblib.load(path)


    # -------------------------------------------------------------------------
    # Helper : entraînement complet depuis un CSV + validation
    # -------------------------------------------------------------------------
    @classmethod
    def train_from_csv(
        cls,
        csv_path: str = DEFAULT_TRAIN_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        text_col: str = "text",
        label_col: str = "label",
        config: Optional[WacilConfig] = None,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> "WacilModel":
        """
        - lit un CSV avec colonnes text/label
        - split train/val
        - entraîne WacilModel
        - affiche l'accuracy de validation
        - sauvegarde le modèle sur disque
        """
        df = pd.read_csv(csv_path)
        assert {text_col, label_col}.issubset(df.columns), \
            f"CSV must contain columns: {text_col}, {label_col}"

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=val_size,
            random_state=random_state,
            stratify=labels,
        )

        model = cls(model_path=None, config=config, load=False)
        model.train(X_train, y_train)

        # Éval locale
        y_val_pred = model.pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[WacilModel] Validation accuracy: {acc:.4f}")
        print(classification_report(y_val, y_val_pred))

        # Sauvegarde
        model.save(model_path)
        print(f"[WacilModel] Saved model to {model_path}")

        return model
    

    @classmethod
    def train_full(
        cls,
        csv_path: str = DEFAULT_TRAIN_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        text_col: str = "text",
        label_col: str = "label",
        config: Optional[WacilConfig] = None,
    ) -> "WacilModel":
        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        model = cls(model_path=None, config=config, load=False)
        model.train(texts, labels)
        model.save(model_path)
        print(f"[WacilModel] Trained on full data, saved to {model_path}")
        return model

# -------------------------------------------------------------------------
# Entrée script : entraîne le modèle localement
# -------------------------------------------------------------------------
if __name__ == "__main__":
    WacilModel.train_from_csv()
