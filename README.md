# Zaboola_ML_student_assistant
**Project Title:** Zaboola student assistant - Fine-Tuning TinyLlama and Phi-4 Models for Question-Answering on AI and Deep Learning Content

**Objective:**
To develop a lightweight, high-accuracy question-answering model by fine-tuning pre-trained transformer architectures (TinyLlama and Phi-4 Mini) on curated academic and instructional content focused on AI and deep learning.
![image](https://github.com/user-attachments/assets/017a0d31-e6f8-4f36-8092-643df0a831bf)

---

**1. Dataset Creation**

* **Sources:**

  * Stanford CS229, CS230, CS221, CS228 course notes, exams, and slides
  * DeepLearning.AI course PDFs (C1M1 to C5M3), Andrew Ng's "Machine Learning Yearning"
  * AI4E, NLP Slides, DLS Slides, MLS Slides, M4ML, optimization documents, and GANS Slides
  * Career-focused content ("How to Build a Career in AI")

* **Generation Method:**

  * Used Gemini 1.5 Flash to generate high-quality question-answer pairs from PDFs
  * Performed preprocessing: extracted raw text, removed noisy content, sectioned by topic
  * Validated generated Q\&A for correctness and academic depth

* **Structure:**

  * Each sample is a dictionary with keys: `question`, `answer`
  * Example:

    ```json
    {"question": "What is the main advantage of ReLU over sigmoid?", "answer": "ReLU mitigates the vanishing gradient problem."}
    ```

---

**2. Model Training**

* **Phase 1: Phi-4 Mini Fine-tuning**

  * Base: Phi-4 Mini (transformer causal decoder)
  * Tokenized prompts with format: `Q: <question>\nA:`
  * Trained for 3 epochs with early stopping
  * Final training loss: \~0.20

* **Phase 2: TinyLlama Fine-tuning**

  * Base: TinyLlama-1.1B-Chat
  * Reformatted inputs into system-user-assistant chat format
  * Included optional system prompt: "You are a helpful deep learning assistant."

* **Training Details:**

  * Optimizer: AdamW
  * Batch size: 1 (gradient accumulation used)
  * Evaluation using BLEU, ROUGE-1/L, BERTScore, F1, and Accuracy

---

**3. Results and Metrics**

* Final evaluation on test set:

  ```json
  {
    "bleu": 0.063,
    "rouge1": 0.252,
    "rougeL": 0.213,
    "bertscore_f1": 0.847
  }
  ```
* Qualitative inspection confirms precise, concise, and context-aware answers

---

**4. Inference & Deployment**

* Implemented `predict_answer(question: str)` inference function
* Support for adding system prompts in chat-style inputs
* Lightweight and suitable for on-device or API deployment

