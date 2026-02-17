from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union, Optional

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TextLike = Union[str, Sequence[str]]
DEFAULT_CHAR_MODEL_PATH = "models/char.joblib"


@dataclass
class CharConfig:
    max_features: int = 30000
    ngram_range: tuple = (3, 5)      # n-grammes de caractères
    min_df: int = 2
    max_df: float = 0.95
    C: float = 1.0


class CharModel:
    """
    Modèle 'expert' plus robuste :
    - TF-IDF caractère (3–5)
    - LogisticRegression
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[CharConfig] = None):
        self.config = config or CharConfig()
        self.pipeline: Optional[Pipeline] = None

        if model_path is not None:
            self.load(model_path)
        else:
            try:
                self.load(DEFAULT_CHAR_MODEL_PATH)
            except FileNotFoundError:
                pass

    def _build_pipeline(self) -> Pipeline:
        tfidf = TfidfVectorizer(
            analyzer="char",
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
        )

        clf = LogisticRegression(
            C=self.config.C,
            max_iter=1000,
            n_jobs=-1,
        )

        return Pipeline(
            steps=[
                ("tfidf", tfidf),
                ("clf", clf),
            ]
        )

    def train(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        if self.pipeline is None:
            self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))

    def predict(self, x: TextLike) -> Union[int, List[int]]:
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas chargé / entraîné (pipeline = None).")

        if isinstance(x, str):
            preds = self.pipeline.predict([x])
            return int(preds[0])

        preds = self.pipeline.predict(list(x))
        return [int(p) for p in preds]

    def save(self, path: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("Impossible de sauvegarder : pipeline = None.")
        joblib.dump(self.pipeline, path)

    def load(self, path: str) -> None:
        self.pipeline = joblib.load(path)

    @staticmethod
    def train_from_csv(
        csv_path: str,
        text_col: str = "text",
        label_col: str = "label",
        model_path: Optional[str] = None,
        config: Optional[CharConfig] = None,
    ) -> "CharModel":
        import pandas as pd

        df = pd.read_csv(csv_path)
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()

        model = CharModel(config=config)
        model.train(texts, labels)

        if model_path is not None:
            model.save(model_path)

        return model
