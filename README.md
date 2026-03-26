# Email_Spam_Analysis# SMS Spam Classifier

A simple spam classifier built with Streamlit and Scikit-learn. Message paste karo — batayega spam hai ya nahi.

---

## Project Structure

```
spam_classifier/
├── app.py            # Main Streamlit app
├── spam.csv          # Kaggle SMS Spam dataset (5573 messages)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Setup & Run

### 1. Clone / Download
Download all files and put them in one folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

App automatically opens at **http://localhost:8501**

---

## How It Works

| Step | Detail |
|------|--------|
| Dataset | Kaggle SMS Spam Collection — 5573 messages (747 spam, 4827 ham) |
| Vectorizer | TF-IDF with bigrams (`ngram_range=(1,2)`, `max_features=10000`) |
| Model | Multinomial Naive Bayes |
| Train/Test split | 80% train, 20% test |
| Accuracy | ~98% on test set |

---

## Dependencies

```
streamlit
scikit-learn
pandas
numpy
```

---

## Dataset

Dataset used: [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

- Total messages: 5573
- Spam: 747
- Ham (not spam): 4827

---

## Usage

1. App khulne ke baad text area mein message paste karo
2. **Check Spam** button dabao
3. Result dikhega — `SPAM` ya `NOT SPAM` confidence ke saath
4. Spam probability bar bhi dikhega

---

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~98% |
| Precision (Spam) | ~97% |
| Recall (Spam) | ~93% |
| F1 Score (Spam) | ~95% |

---

Made with Streamlit + Scikit-learn