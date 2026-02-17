"""
This script is meant to be executed on a local environment.
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker

from src.randomizer import RandomizerModel

TEST_DATASET_PATH = './data/sample_test.csv'


def evaluate():
    tracker = EmissionsTracker()
    df = pd.read_csv(TEST_DATASET_PATH)
    model = RandomizerModel()

    predictions = []
    tracker.start()
    for row in df.itertuples():
        pred = model.predict(row.text)
        predictions.append({
            'id': row.id,
            'label': pred
        })
    emissions = tracker.stop()

    preds = pd.DataFrame(predictions)['label'].to_list()
    targets = df['label'].to_list()

    print('ACCURACY:', accuracy_score(targets, preds))
    print('GWP OPERATIONAL IMPACT:', round(emissions * 1000, 3), 'g CO2eq.')


if __name__ == '__main__':
    evaluate()
