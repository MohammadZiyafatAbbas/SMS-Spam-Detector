import streamlit as st
st.set_page_config(
    page_title="SMS Spam Shield",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

import pickle
import pandas as pd
from cleantext import clean
from pathlib import Path

# --- Load custom CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

local_css("styles.css")

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model/spam_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# --- Preprocess ---
def preprocess(text):
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
        no_punct=True,
        lang="en"
    )
    tokens = [word for word in text.split() if word.isalpha()]
    return " ".join(tokens)

# --- Predict ---
def predict_spam(message):
    processed = preprocess(message)
    vect = vectorizer.transform([processed])
    return model.predict(vect)[0]

# --- Save Feedback ---
def save_feedback(message, prediction, correct_label):
    feedback_path = Path("data/feedback.csv")
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "message": [message],
        "model_prediction": [prediction],
        "user_feedback": [correct_label]
    }
    df = pd.DataFrame(entry)

    try:
        if feedback_path.exists():
            df.to_csv(feedback_path, mode='a', index=False, header=False)
        else:
            df.to_csv(feedback_path, index=False)
    except Exception as e:
        st.error(f"❌ Could not save feedback: {e}")

# --- UI Header ---
def header_section():
    st.markdown("""
    <div class="header">
        <h1>🛡️ SMS Spam Shield</h1>
        <p class="subheader">Protecting you from unwanted messages</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# --- Input Section ---
def input_section():
    st.markdown("### 📝 Enter Your Message")

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""

    st.session_state.message_input = st.text_area(
        "Type or paste your SMS message here:",
        value=st.session_state.message_input,
        height=150,
        placeholder="e.g., 'Congratulations! You've won a $1000 gift card...'",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔍 Analyze Message", use_container_width=True):
            st.session_state.prediction = predict_spam(st.session_state.message_input)
            st.session_state.feedback_mode = False
            st.session_state.feedback_done = False

    with col2:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# --- Result Section ---
def result_section():
    pred = st.session_state.prediction
    message = st.session_state.message_input

    if pred == 1:
        st.error("## 🚨 Spam Detected!")
    else:
        st.success("## ✅ Legitimate Message")

    st.markdown("---")
    st.markdown("### 💬 Help Improve Our Service")
    st.markdown("Was this classification correct?")

    col1, col2, _ = st.columns([1, 1, 2])

    if not st.session_state.feedback_done:
        with col1:
            if st.button("👍 Correct"):
                save_feedback(message, pred, pred)
                st.session_state.feedback_done = True
                st.toast("✅ Thank you for your feedback!", icon="🙏")

        with col2:
            if st.button("👎 Incorrect"):
                st.session_state.feedback_mode = True

        if st.session_state.feedback_mode:
            correct_label = st.radio(
                "What should it be?",
                ["Ham (Not Spam)", "Spam"],
                horizontal=True,
                key="feedback_radio"
            )
            if st.button("✅ Submit Correction"):
                correct_value = 0 if correct_label == "Ham (Not Spam)" else 1
                save_feedback(message, pred, correct_value)
                st.session_state.feedback_done = True
                st.toast("✨ Thanks for helping us improve!")

    else:
        st.success("✅ Feedback received. Thank you!")

# --- Main App ---
def main():
    header_section()
    input_section()

    if "prediction" in st.session_state and st.session_state.message_input.strip():
        st.markdown("### 📊 Analysis Results")
        result_section()

if __name__ == "__main__":
    main()
