"""
This script is meant to be executed on the CI server.
"""
import os
import pandas as pd

from src.randomizer import RandomizerModel
from src.wacil import WacilModel
from src.svm_model import SVMModel
from src.wacil import WacilModel

TEST_DATASET_URL = os.environ['TEST_DATASET_TEXT_URL']


def evaluate():
    df = pd.read_csv(TEST_DATASET_URL)
    model = WacilModel()

    predictions = []
    for row in df.itertuples():
        pred = model.predict(row.text)
        predictions.append({
            'id': row.id,
            'label': pred
        })

    pd.DataFrame(predictions).to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    evaluate()
