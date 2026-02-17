"""
MathysModel : modèle TF-IDF + LinearSVC optimisé pour IMDB
- Entraînement rapide
- Bonne accuracy
- Inference très peu coûteuse (bon score carbone)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

TextLike = Union[str, Sequence[str]]

DEFAULT_MODEL_PATH = "models/mathys.joblib"


@dataclass
class MathysConfig:
    max_features: int = 15000
    ngram_range: tuple = (1, 2)
    min_df: int = 3
    max_df: float = 0.9
    C: float = 0.8


class MathysModel:
    """
    Modèle "baseline ++" pour le Green AI Hack :
    - Vectorisation TF-IDF (unigrams + bigrams)
    - Classifieur LinearSVC
    - Interface simple : train / predict / save / load
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[MathysConfig] = None):
        self.config = config or MathysConfig()
        self.pipeline: Optional[Pipeline] = None

        # Cas 1 : chemin explicite fourni
        if model_path is not None:
            self.load(model_path)
        else:
            # Cas 2 : aucun chemin fourni -> on tente de charger le modèle par défaut
            try:
                self.load(DEFAULT_MODEL_PATH)
            except FileNotFoundError:
                # OK en phase d'entraînement, le modèle sera créé par train()
                pass

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
            stop_words="english",       # 🔹 comme ton pote
            dtype=np.float32,           # 🔹 plus light
        )

        clf = LinearSVC(
            C=self.config.C,
            max_iter=3000,              # 🔹 pour être safe sur la convergence
        )

        pipe = Pipeline(
            steps=[
                ("tfidf", tfidf),
                ("clf", clf),
            ]
        )
        return pipe


    # -------------------------------------------------------------------------
    # Entraînement
    # -------------------------------------------------------------------------
    def train(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))

    # -------------------------------------------------------------------------
    # Prédiction
    # -------------------------------------------------------------------------
    def predict(self, x: TextLike) -> Union[int, List[int]]:
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas chargé / entraîné (pipeline = None).")

        if isinstance(x, str):
            preds = self.pipeline.predict([x])
            return int(preds[0])

        preds = self.pipeline.predict(list(x))
        return [int(p) for p in preds]

    # ---------- marge de décision (pour la cascade) --------------------------
    def decision_margin(self, x: TextLike) -> Union[float, List[float]]:
        """
        Retourne la marge de décision (distance à l'hyperplan) de LinearSVC.
        Sert à détecter les cas "ambigus".
        """
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas chargé / entraîné (pipeline = None).")

        if isinstance(x, str):
            scores = self.pipeline.decision_function([x])  # shape (1,)
            return float(scores[0])

        scores = self.pipeline.decision_function(list(x))
        return [float(s) for s in scores]

    # -------------------------------------------------------------------------
    # Sauvegarde / chargement
    # -------------------------------------------------------------------------
    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Impossible de sauvegarder : pipeline = None.")
        joblib.dump(
            {
                "config": self.config,
                "pipeline": self.pipeline,
            },
            path,
        )

    def load(self, path: str) -> None:
        obj = joblib.load(path)
        if isinstance(obj, dict) and "pipeline" in obj:
            self.config = obj.get("config", MathysConfig())
            self.pipeline = obj["pipeline"]
        else:
            self.pipeline = obj

    # -------------------------------------------------------------------------
    # Helper pour entraînement depuis un CSV
    # -------------------------------------------------------------------------
    @staticmethod
    def train_from_csv(
        csv_path: str,
        text_col: str = "text",
        label_col: str = "label",
        model_path: Optional[str] = None,
        config: Optional[MathysConfig] = None,
    ) -> "MathysModel":
        import pandas as pd

        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        model = MathysModel(config=config)
        model.train(texts, labels)

        if model_path is not None:
            model.save(model_path)

        return model


# ===========================
#  Modèle "expert" en cascade
# ===========================

class CascadeModel:
    """
    Modèle en cascade :
    - MathysModel (rapide) pour les cas confiants
    - CharModel (plus coûteux mais plus robuste) pour les cas ambigus
    """

    def __init__(
        self,
        fast_model_path: str = DEFAULT_MODEL_PATH,
        expert_model_path: str = "models/char.joblib",
        margin_threshold: float = 1,
    ):
        from src.char_model import CharModel  # import tardif pour éviter les cycles

        self.fast = MathysModel(model_path=fast_model_path)
        self.expert = CharModel(model_path=expert_model_path)
        self.margin_threshold = margin_threshold

    def predict(self, x: TextLike) -> Union[int, List[int]]:
        single = isinstance(x, str)
        texts = [x] if single else list(x)

        preds: List[int] = []
        for t in texts:
            margin = self.fast.decision_margin(t)
            if abs(margin) >= self.margin_threshold:
                preds.append(self.fast.predict(t))
            else:
                preds.append(self.expert.predict(t))

        return preds[0] if single else preds
