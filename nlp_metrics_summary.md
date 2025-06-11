
# 📊 NLP Evaluation Metrics: Definitions, Interpretations, and Use Cases

This document summarizes key evaluation metrics used in Natural Language Processing (NLP), especially for classification and text generation tasks.

---

## 🔹 1. **Precision**
- **Definition**: Fraction of predicted positives that are truly positive.
- **Formula**: `Precision = TP / (TP + FP)`
- **Interpretation**: High precision means few false positives.
- **Use When**: False positives are costly (e.g., spam detection).

---

## 🔹 2. **Recall (Sensitivity)**
- **Definition**: Fraction of actual positives correctly identified.
- **Formula**: `Recall = TP / (TP + FN)`
- **Interpretation**: High recall means few false negatives.
- **Use When**: Missing positives is costly (e.g., medical diagnosis).

---

## 🔹 3. **Specificity**
- **Definition**: Fraction of actual negatives correctly identified.
- **Formula**: `Specificity = TN / (TN + FP)`
- **Interpretation**: High specificity = few false positives.
- **Use When**: You want to confidently rule out negatives.

---

## 🔹 4. **F1 Score**
- **Definition**: Harmonic mean of precision and recall.
- **Formula**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **Interpretation**: Balanced metric for imbalanced datasets.
- **Use When**: You want a single score combining precision and recall.

---

## 🔹 5. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
Used in **text summarization** to measure n-gram overlaps between generated and reference summaries.

### Variants:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence
- **ROUGE-Lsum**: LCS at summary level

**Use When**: Evaluating summarization or paraphrasing tasks.

---

## 🔹 6. **BLEU (Bilingual Evaluation Understudy)**
Measures **n-gram precision** between generated text and one or more references, with a **brevity penalty**.

- **Interpretation**: Higher BLEU = closer match to reference
- **Use When**: Evaluating **machine translation** and sometimes summarization (less ideal for abstractive outputs)

---

## 🔹 7. **BERTScore**
Uses BERT embeddings to compare generated and reference texts at the **semantic level**.

- **Precision**: Are predicted tokens semantically similar to reference?
- **Recall**: Does the prediction capture most of the reference?
- **F1**: Balanced semantic overlap

**Use When**: You want to evaluate **semantic similarity**, especially for abstractive summarization, generation, QA, etc.

---

## 📘 Summary Table

| Metric     | Best For                        | Measures                | Notes                                      |
|------------|----------------------------------|--------------------------|--------------------------------------------|
| Precision  | Classification (e.g., spam)     | TP / (TP + FP)          | Penalizes false positives                  |
| Recall     | Classification (e.g., diagnosis)| TP / (TP + FN)          | Penalizes false negatives                  |
| Specificity| Classification (e.g., screening)| TN / (TN + FP)          | Rules out negatives                        |
| F1 Score   | Imbalanced classification       | Harmonic of P & R       | Trade-off between P and R                  |
| ROUGE      | Summarization                   | n-gram/LCS overlap       | Measures lexical overlap                   |
| BLEU       | Machine Translation             | n-gram precision + BP    | Reference overlap + length penalty         |
| BERTScore  | Semantic evaluation             | Contextual similarity    | Captures meaning better than n-gram overlap|

---

**Note**: Always choose metrics based on the **task** and **what kind of errors matter most** for your application.
