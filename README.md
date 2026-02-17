## 🧠 Green AI: Energy-Efficient NLP Under Deployment Constraints

Energy-aware sentiment analysis benchmarking classical and transformer models under CPU-only deployment constraints.

⸻

Overview

This project investigates how to balance predictive performance and environmental impact in modern NLP systems.

While transformer models achieve strong accuracy, they often require substantial computational resources. This work evaluates whether compact architectures and quantization techniques can maintain competitive performance while significantly reducing carbon footprint and inference cost.

The focus is not only accuracy — but deployment realism.

⸻

## 🎯 Objective

To evaluate the trade-offs between:

	•	Predictive performance
	•	Inference efficiency
	•	Carbon footprint (GWP)
	•	CPU-only feasibility

Key questions:

	•	How much accuracy is gained by transformers over classical baselines?
	•	What is the environmental cost of that gain?
	•	Can quantization reduce impact while preserving performance?
	•	What configuration offers the best accuracy-to-carbon ratio?

⸻

## 📦 Experimental Setup

Dataset

	•	IMDb Movie Review Dataset
	•	Binary sentiment classification (positive / negative)

Deployment Constraint

	•	CPU-only inference
	•	No GPU acceleration
	•	Lightweight runtime environment

All models were evaluated under consistent runtime conditions.

⸻

## 🧪 Model Variants

Model Variant	Description
TF-IDF + Linear SVM	Classical sparse feature baseline
TinyBERT (Fine-Tuned)	Compact transformer model
DistilBERT Cascade	Two-stage confidence-based pipeline
Quantized TinyBERT (INT8)	Post-training quantized transformer


⸻

## ⚙ Methodology

	1.	Text preprocessing and tokenization
	2.	TF-IDF feature extraction for classical baseline
	3.	Fine-tuning compact transformer models
	4.	Post-training INT8 quantization
	5.	Benchmarking under CPU-only constraints
	6.	Evaluation using:
	•	Accuracy
	•	Carbon footprint (GWP)
	•	Efficiency trade-off analysis

Carbon impact values were obtained from official competition evaluation logs.

⸻

## 📊 Evaluation Results

Green AI Benchmark — Team “nous”

| Model Variant              | Accuracy    | Total CO₂ (GWP)      | Key Observation                          |
|----------------------------|------------|----------------------|------------------------------------------|
| TF-IDF + Linear SVM        | 0.78–0.79  | ~0.00001–0.00002     | Extremely low footprint, strong baseline |
| Quantized Transformer      | ~0.79      | ~0.00007–0.00008     | Best accuracy–impact trade-off           |
| Fine-Tuned Transformer     | ~0.80      | ~0.00013+            | Highest accuracy, highest impact         |

All results correspond to official competition submissions recorded under team name “nous”.

⸻

📈 Key Findings

	•	Classical models remain highly competitive under strict efficiency constraints.
	•	Compact transformers provide measurable accuracy gains.
	•	Quantization significantly reduces environmental cost.
	•	Accuracy improvements come with non-trivial carbon trade-offs.
	•	Deployment-aware benchmarking changes model selection decisions.

⸻

## 🧠 Engineering Perspective

This project emphasizes:

	•	Deployment-aware model selection
	•	Fair benchmarking under fixed runtime conditions
	•	Carbon-aware ML evaluation
	•	Trade-off analysis over leaderboard chasing
	•	Practical system constraints

⸻

## 🏗 Project Structure

```
green-ai-efficient-nlp/
├── README.md
├── src/
│   ├── train_svm_supriya.py
│   ├── train_distilbert_supriya.py
│   ├── ensemble_svm.py
│   ├── tinybert_model.py
│   ├── cascade_model.py
│   ├── green_ai_model.py
│   └── char_model.py
├── results/
├── figures/
└── docs/
```

⸻

## 🚀 Possible Extensions

	•	Knowledge distillation comparison
	•	Structured pruning experiments
	•	ONNX export for optimized CPU runtime
	•	Direct energy instrumentation
	•	Extended carbon benchmarking

⸻

## 🧠 What This Project Demonstrates

	•	Practical model compression
	•	Quantization-aware evaluation
	•	Efficiency-focused transformer deployment
	•	Classical vs deep learning trade-off analysis
	•	Real-world ML decision thinking

⸻

## Status

Competition prototype completed.
Code structured for reproducibility and public demonstration.


