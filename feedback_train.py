# retrain_model.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
from cleantext import clean
import spacy

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

def load_data():
    """Load original training data and feedback data"""
    try:
        # Load original data
        df_original = pd.read_csv("data\spam.csv")  
        
        # Load feedback data
        df_feedback = pd.read_csv("data\feedback.csv", header=None, 
                                names=["message", "model_prediction", "user_feedback"])
        
        # Combine datasets
        df_combined = pd.concat([
            df_original[['message', 'label']].rename(columns={'label': 'true_label'}),
            df_feedback[['message', 'user_feedback']].rename(columns={'user_feedback': 'true_label'})
        ])
        
        return df_combined
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

def preprocess(text):
    """Clean and preprocess text"""
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
    return " ".join([token.lemma_ for token in doc if token.is_alpha])

def retrain_model():
    """Retrain model with combined dataset"""
    print("Loading data...")
    df = load_data()
    if df is None:
        return

    print("Preprocessing data...")
    df['processed_text'] = df['message'].apply(preprocess)
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['true_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.2f}")
    print(f"Test accuracy: {test_score:.2f}")
    
    # Save new model
    print("Saving models...")
    with open("model/retrained_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open("model/retrained_spam_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Retraining complete! New models saved as:")
    print("- model/retrained_vectorizer.pkl")
    print("- model/retrained_spam_classifier.pkl")

if __name__ == "__main__":
    print("Starting model retraining process...")
    retrain_model()
    