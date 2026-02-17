# src/big_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BigModel:
    def __init__(self, model_dir="models/distilbert_cascade"):
        # CPU only for CI to keep impact controlled
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts):
        preds = []
        # 🔧 slightly bigger batch, fewer forward passes
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                # 🔧 shorter sequence: cheaper, usually same accuracy
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)

        return [int(p) for p in preds]
