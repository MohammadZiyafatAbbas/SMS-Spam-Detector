# ========== 🔹 1. IMPORTS ========== #
import pandas as pd
import spacy
from cleantext import clean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle
import os

# ========== 🔹 2. LOAD SPACY ========== #
nlp = spacy.load("en_core_web_sm")

# ========== 🔹 3. TEXT PREPROCESSING FUNCTION ========== #
def preprocess_text(text):
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=True,
        lower=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=True,
        no_currency_symbols=True,
        no_punct=False,
        lang="en"
    )
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    return " ".join(tokens)

# ========== 🔹 4. LOAD DATASET ========== #
df = pd.read_csv("data\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert labels: ham -> 0, spam -> 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(inplace=True)

print("Original shape:", df.shape)
print("Label counts:\n", df['label'].value_counts().to_frame(name='count'))

# ========== 🔹 5. PREPROCESS TEXTS ========== #
df['clean_text'] = df['text'].apply(preprocess_text)

# ========== 🔹 6. SPLIT DATA ========== #
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== 🔹 7. VECTORIZER ========== #
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== 🔹 8. HANDLE CLASS IMBALANCE ========== #
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_tfidf, y_train)

# ========== 🔹 9. TRAIN MODEL ========== #
model = MultinomialNB()
model.fit(X_resampled, y_resampled)

# ========== 🔹 10. EVALUATION ========== #
y_pred = model.predict(X_test_tfidf)

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔍 Show messages that were missed as spam
print("\n❌ Missed Spam Messages:")
for orig_text, pred, true in zip(X_test, y_pred, y_test):
    if pred == 0 and true == 1:
        print("   ", orig_text)

# ========== 🔹 11. SAVE MODEL ========== #
os.makedirs("model", exist_ok=True)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model and vectorizer saved to 'model/' directory.")
