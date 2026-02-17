import os
from src.char_model import CharModel

if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"
    MODEL_PATH = "models/char.joblib"

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = CharModel.train_from_csv(
        csv_path=TRAIN_PATH,
        text_col="text",
        label_col="label",
        model_path=MODEL_PATH,
    )

    print("✅ Modèle expert (char) entraîné et sauvegardé dans", MODEL_PATH)
