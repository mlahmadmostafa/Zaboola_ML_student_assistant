# Zaboola_ML_student_assistant

## 🧠 Project: Fine-Tuning Phi-4 Mini & TinyLlama for Question Answering

### 🔍 Overview

This project focused on fine-tuning compact open-source language models (**Phi-4 Mini** and **TinyLlama**) for a **question answering task** using a custom dataset generated from educational material. The goal was to build an efficient, accurate assistant for answering questions from machine learning course content.

---

### ⚙️ Models Used

* **Phi-4 Mini**: A lightweight variant of Microsoft's Phi-4 model
* **TinyLlama-1.1B-Chat**: Lightweight chat-tuned model suitable for fine-tuning
* **Architecture**: AutoModelForCausalLM (causal decoder-only)
* **Tokenizer**: AutoTokenizer for each respective model

---

### 📁 Dataset

* **Sources**:

  * Stanford University lectures, slides and notes
  * DeepLearning.AI courses by Andrew Ng
  * Associated textbooks and official documentation
* **Method**:

  * Gemini Flash 1.5 was used to extract and convert lecture slides, books, and notes into structured Q\&A format.
* **Format Example**:

  ```json
  {
    "question": "What is the bias-variance tradeoff?",
    "answer": "The bias-variance tradeoff refers to the balance between a model's ability to minimize bias and variance, which affects generalization performance.",
    "text": "Q: What is the bias-variance tradeoff?\nA: The bias-variance tradeoff refers to the balance..."
  }
  ```

---

### 🧪 Training Setup

* **Framework**: Hugging Face Transformers

* **Trainer**: `Trainer`

* **Loss Function**: CrossEntropyLoss

* **TrainingArguments**:

  ```python
  TrainingArguments(
      output_dir="checkpoints",
      num_train_epochs=3,
      per_device_train_batch_size=1,
      per_device_eval_batch_size=1,
      gradient_accumulation_steps=4,
      save_strategy="epoch",
      evaluation_strategy="epoch",
      logging_dir="logs",
      learning_rate=5e-5,
      bf16=True,
      report_to="tensorboard"
  )
  ```

* **Metrics Tracked**:

  * Accuracy
  * F1
  * BLEU
  * ROUGE
  * BERTScore

---

### ✅ Results

* **Phi-4 Mini Final Loss**: \~0.20
* **TinyLlama**: Comparable performance, slightly better generalization
* **Observations**:

  * Model learned cleanly from highly structured content
  * Outputs were fluent and instructional

---

### 🛠️ Optimization Tricks

* Used `local_files_only=True` to avoid re-downloading
* Cached models/tokenizers with Hugging Face cache
* Mixed precision training with bf16 for performance

---

### 📌 Summary

A successful small-model fine-tuning project for educational Q\&A using high-quality instructional content and Gemini-generated data. The result was a lightweight assistant capable of answering foundational ML course questions efficiently.
