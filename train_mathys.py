import os
from src.mathys import MathysModel

if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"
    MODEL_PATH = "models/mathys.joblib"

    # création auto du dossier
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = MathysModel.train_from_csv(
        csv_path=TRAIN_PATH,
        text_col="text",   # adapte selon ton CSV
        label_col="label",
        model_path=MODEL_PATH,
    )

    print("✅ Modèle entraîné et sauvegardé dans", MODEL_PATH)
