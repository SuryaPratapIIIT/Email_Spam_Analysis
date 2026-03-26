import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.title("Spam Classifier")
st.write("Message likho — batayega spam hai ya nahi.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("spam.csv")
    df.columns = df.columns.str.strip()

    if "Category" in df.columns and "Message" in df.columns:
        df = df[["Category", "Message"]].dropna()
        df["label"] = (df["Category"].str.lower() == "spam").astype(int)
        df["text"] = df["Message"]
    elif "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].dropna()
        df["label"] = (df["v1"].str.lower() == "spam").astype(int)
        df["text"] = df["v2"]
    else:
        st.error("CSV format pehchana nahi.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)),
        ("clf", MultinomialNB()),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

    total = len(df)
    spam_count = int(df["label"].sum())
    ham_count = total - spam_count

    return model, acc, report, total, spam_count, ham_count

model, acc, report, total, spam_count, ham_count = load_and_train()

with st.expander("Dataset info"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total messages", total)
    c2.metric("Spam", spam_count)
    c3.metric("Ham", ham_count)
    c4.metric("Accuracy", f"{round(acc * 100, 1)}%")

    c5, c6 = st.columns(2)
    c5.metric("Spam precision", f"{round(report['Spam']['precision']*100,1)}%")
    c6.metric("Spam recall",    f"{round(report['Spam']['recall']*100,1)}%")

st.divider()

user_input = st.text_area("Message yahan paste karo:", height=120,
                          placeholder="Type or paste your message here...")

if st.button("Check Spam"):
    if user_input.strip() == "":
        st.warning("Kuch likho pehle.")
    else:
        pred = model.predict([user_input])[0]
        prob = model.predict_proba([user_input])[0]
        spam_prob  = round(prob[1] * 100, 1)
        confidence = round(max(prob) * 100, 1)

        if pred == 1:
            st.error(f"SPAM — {confidence}% confidence")
        else:
            st.success(f"NOT SPAM — {confidence}% confidence")

        st.progress(int(spam_prob), text=f"Spam probability: {spam_prob}%")

st.divider()
st.caption("Model: Naive Bayes + TF-IDF | Kaggle SMS Spam dataset (5573 messages)")