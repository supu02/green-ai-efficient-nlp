import pickle
import torch
from transformers import AutoTokenizer

class CascadeModel:
    def __init__(self):

        print("🔄 Loading Cascade Model (SVM + TinyBERT)")

        # ----- Load SVM (FAST MODEL) -----
        with open("models/svm_model.pkl", "rb") as f:
            self.svm = pickle.load(f)

        # ----- Load TinyBERT tokenizer -----
        model_dir = "models/tinybert_finetuned"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # ----- Load FULL QUANTIZED TinyBERT -----
        quant_path = "models/tinybert_quantized_full.pt"
        self.bert = torch.load(quant_path, map_location="cpu")
        self.bert.eval()

        self.device = torch.device("cpu")

        # Thresholds (safe values)
        self.conf_threshold = 0.75   # 75% confidence from SVM
        self.length_threshold = 12   # reviews longer than 12 words → BERT

        print("Cascade Ready ✔")

    def svm_confidence(self, text):
        """
        Estimate confidence using SVM decision function magnitude.
        Higher absolute value ⇒ higher confidence.
        """
        score = self.svm.decision_function([text])[0]
        # Normalize to 0–1 confidence
        conf = min(1, abs(score) / 5)
        return conf

    def predict(self, text):
        # Quick length check
        if len(text.split()) <= self.length_threshold:
            return int(self.svm.predict([text])[0])

        # Confidence check
        conf = self.svm_confidence(text)
        if conf >= self.conf_threshold:
            return int(self.svm.predict([text])[0])

        # Otherwise → use TinyBERT
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():
            outputs = self.bert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        pred = torch.argmax(outputs.logits, dim=1).item()
        return pred