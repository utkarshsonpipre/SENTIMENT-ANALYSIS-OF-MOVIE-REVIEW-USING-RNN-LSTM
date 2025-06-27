# 🧠 Character-Level Sentiment Classification using CNN + BiLSTM  
**Sentiment Analysis on IMDB Reviews using Deep Character-Level Models**

---

## 🔍 Overview

This project implements character-level deep learning models for sentiment classification.

The model processes text at the **character level**, transforming reviews into hierarchical encodings that capture sentence-level and document-level semantics.

> ✅ **Achieves 98% training accuracy and 87% validation accuracy** on the IMDB dataset.

---

## 📚 Architecture Summary

### ➤ Sentence Encoding (Character-Level)

Each sentence is:
- Truncated/padded to **512 characters**
- Encoded via:
  - **Bi-directional LSTM**, OR
  - **CNN Sentence Encoder** with 3 Conv1D + ReLU + MaxPooling + Dropout layers

### ➤ Document Encoding

- Sentences (max 15 per review) are encoded using `TimeDistributed` layers
- Sentence encodings are passed to a **second BiLSTM** for document-level understanding

### ➤ Output

- A final dense layer with **sigmoid activation** gives binary classification (positive/negative sentiment)

---

## 🧪 Model Variants

### ✅ **CNN + BiLSTM (Default)**  
- CNN encodes sentence → BiLSTM encodes document  
- Achieves **98% training accuracy**, **87% validation accuracy**

### ✅ **CNN-Only Sentence Encoder**  
- Dual-stream CNN (different filter sizes) + Temporal MaxPooling + Dense layers

> Switch models via configuration flags inside `model.ipynb`

---

## 🗃️ Dataset

**IMDB Movie Reviews Dataset**  
- `labeledTrainData.tsv` with **25,000 labeled reviews**
- Train: 20,000  
- Validation: 2,500  
- Test: 2,500  

---

## 🧹 Preprocessing Steps

1. Remove HTML tags  
2. Replace non-ASCII characters  
3. Split reviews into sentences  
4. Limit: 15 sentences/review, 512 characters/sentence  
5. Prepare `(doc, sentence, char)` format for input

---

## 📈 Performance

| Metric             | Accuracy |
|--------------------|----------|
| **Training**       | 98%      |
| **Validation**     | 87%      |

---

## ⚙️ Requirements

```txt
pandas==0.20.3  
tensorflow==1.4.0  
keras==2.0.8  
numpy==1.14.0  
