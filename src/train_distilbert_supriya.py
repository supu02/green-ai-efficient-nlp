import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from pathlib import Path

DATA_PATH = Path("../data/train.csv")
MODEL_DIR = Path("models/distilbert")
QUANT_PATH = Path("models/distilbert_quantized.pt")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]] 
    return Dataset.from_pandas(df)


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


def main():
    print("📥 Loading dataset...")
    dataset = load_data()

    print("🔠 Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("🔧 Tokenizing dataset...")
    tokenized = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    tokenized = tokenized.train_test_split(test_size=0.1)

    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]

    train_ds = train_ds.remove_columns(["text", "__index_level_0__"])
    eval_ds = eval_ds.remove_columns(["text", "__index_level_0__"])

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    print("🤖 Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results_distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_steps=200,
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # mixed precision if GPU available
    )

    print("🚀 Training DistilBERT...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer
    )

    trainer.train()

    print("💾 Saving model...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("⚖️ Applying dynamic quantization...")
    base = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    base.eval()

    quantized = torch.quantization.quantize_dynamic(
        base,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    torch.save(quantized.state_dict(), QUANT_PATH)
    print(f"✅ Quantized model saved to {QUANT_PATH}")


if __name__ == "__main__":
    main()