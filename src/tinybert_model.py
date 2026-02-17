import torch
from transformers import AutoTokenizer

class TinyBertModel:
    def __init__(self):
        print("Loading FULL quantized TinyBERT...")

        model_dir = "models/tinybert_finetuned"
        quant_path = "models/tinybert_quantized_full.pt"

        self.device = torch.device("cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Load FULL quantized model (architecture + weights)
        self.model = torch.load(quant_path, map_location="cpu")
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128  # Energy efficient!
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        pred = torch.argmax(outputs.logits, dim=1).item()
        return pred