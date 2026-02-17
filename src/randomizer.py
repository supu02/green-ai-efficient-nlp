import random
import time


class RandomizerModel:

    def __init__(self):
        random.seed(42)

    def predict(self, text: str) -> int:
        time.sleep(0.01)
        return random.randint(0, 1)
