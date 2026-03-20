# Fake News Detection System

A complete, production-ready **Fake News Detection** system built with **TensorFlow / Keras**, **NLP** text processing, a rule-based fact-check layer, and a **Flask** web UI.

---

## ✨ Features

| Feature | Detail |
|---|---|
| Deep-learning classifier | Embedding → LSTM → Dense(sigmoid) |
| Text preprocessing | Lowercase, URL/HTML stripping, punctuation removal, padding |
| Fact-check layer | Rule-based keyword scoring blended with ML probability |
| Explanation output | Human-readable reason for the verdict |
| Flask web app | Responsive single-page UI with confidence bar |
| REST API | `/predict` JSON endpoint for programmatic access |

---

## 📁 Project Structure

```
Fake-News-Detection/
├── data/
│   └── README.md        ← instructions for downloading the dataset
├── templates/
│   └── index.html       ← Flask HTML template (UI)
├── static/
│   └── style.css        ← Stylesheet
├── utils.py             ← Text cleaning, tokeniser helpers, fact-check scoring
├── model.py             ← Keras model definition
├── train.py             ← Training script (loads data, trains, evaluates, saves)
├── app.py               ← Flask web application
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 – Clone & install dependencies

```bash
git clone https://github.com/2025cssiddharth24641-wq/Fake-News-Detection.git
cd Fake-News-Detection
pip install -r requirements.txt
```

### 2 – Download the dataset

Follow the instructions in [`data/README.md`](data/README.md) to download
`Fake.csv` and `True.csv` from Kaggle and place them inside the `data/` folder.

### 3 – Train the model

```bash
python train.py
```

This will:
- Clean and tokenise the news articles
- Build the LSTM model
- Train for up to 7 epochs with early stopping
- Save `model.h5` and `tokenizer.pkl` in the project root
- Print accuracy, precision, recall, F1 and confusion matrix

### 4 – Run the web app

```bash
python app.py
```

Open http://localhost:5000 in your browser, paste any news text, and click **Check News**.

---

## 🔌 REST API

### `POST /predict`

```json
// Request body
{ "text": "Scientists discover cure for all diseases in shocking breakthrough…" }

// Response
{
  "label": "FAKE",
  "confidence": 87.3,
  "ml_probability": 82.1,
  "rule_based_score": 40.0,
  "explanation": "The ML model assigned a 82.1% fake-news probability. The text also contains sensationalist language (rule-based score: 40%)."
}
```

---

## 🧠 Model Architecture

```
Embedding(20000, 128, input_length=500)
    ↓
SpatialDropout1D(0.3)
    ↓
LSTM(64)
    ↓
Dropout(0.3)
    ↓
Dense(1, activation='sigmoid')
```

- **Loss:** binary crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy

---

## 🔍 Fact-Check Layer

Beyond the ML model, a simple rule-based scorer flags articles containing
sensationalist keywords such as *"breaking"*, *"shocking"*, *"viral"*,
*"conspiracy"*, *"leaked"*, etc.

The final verdict is a weighted blend:

```
blended_score = 0.85 × ml_probability + 0.15 × rule_based_score
```

---

## 📊 Expected Results (on Kaggle dataset)

| Metric | Score |
|---|---|
| Accuracy | ~98 % |
| Precision | ~98 % |
| Recall | ~98 % |
| F1-Score | ~98 % |

---

## 📋 Requirements

- Python 3.8+
- TensorFlow ≥ 2.10
- Flask ≥ 2.2
- pandas, NumPy, scikit-learn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is released under the MIT License.