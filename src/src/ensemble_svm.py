# src/ensemble_svm.py

import numpy as np


class WordCharEnsemble:
    """
    Ensemble of:
      - word-level TF-IDF + LinearSVC
      - char-level TF-IDF + LinearSVC

    We combine their decision_function scores with learned weights.
    """

    def __init__(self, word_clf, char_clf,
                 w_word=0.6, w_char=0.4,
                 scale_word=1.0, scale_char=1.0):
        self.word_clf = word_clf
        self.char_clf = char_clf
        self.w_word = w_word
        self.w_char = w_char
        self.scale_word = scale_word
        self.scale_char = scale_char

    def _scores(self, texts):
        """Return combined decision scores for a list of texts."""
        word_scores = self.word_clf.decision_function(texts)
        char_scores = self.char_clf.decision_function(texts)

        # Normalize by std so both models have comparable scale
        word_scores = word_scores / (self.scale_word + 1e-8)
        char_scores = char_scores / (self.scale_char + 1e-8)

        scores = self.w_word * word_scores + self.w_char * char_scores
        return scores

    def predict(self, X):
        """
        Accept:
          - single string → returns int
          - list/array of strings → returns np.array of ints
        """
        # Normalize to list for sklearn compatibility
        if isinstance(X, str):
            texts = [X]
            single = True
        else:
            texts = list(X)
            single = False

        scores = self._scores(texts)
        # LinearSVC convention: score > 0 → class 1, else class 0
        preds = (scores > 0).astype(int)

        return int(preds[0]) if single else preds
