import pickle, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading spam.csv...")
df = pd.read_csv("spam.csv")
df.columns = df.columns.str.strip()

if "Category" in df.columns:
    df["label"] = (df["Category"].str.lower() == "spam").astype(int)
    df["text"] = df["Message"]
else:
    df["label"] = (df["v1"].str.lower() == "spam").astype(int)
    df["text"] = df["v2"]

df = df.dropna(subset=["text", "label"])
print(f"Rows: {len(df)} | Spam: {df.label.sum()}")

print("Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["text"])

print("Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X, df["label"])

# Quick sanity check
sample = tfidf.transform(["you won 1000000 dollar prize claim it now"])
pred = model.predict(sample)[0]
print("Test prediction:", "SPAM" if pred == 1 else "NOT SPAM")

base = r"C:\Users\Suraj\PycharmProjects\PythonProject\SMS_SPAM_CLASSIFICATION"
with open(os.path.join(base, "vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(base, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("Done! vectorizer.pkl and model.pkl saved to", base)
