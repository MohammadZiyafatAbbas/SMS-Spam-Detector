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
import base64
import requests
import json
import io

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

# --- Save Feedback to GitHub ---
def save_feedback(message, prediction, correct_label):
    entry = {
        "message": [message],
        "model_prediction": [prediction],
        "user_feedback": [correct_label]
    }
    new_df = pd.DataFrame(entry)

    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    GITHUB_USERNAME = st.secrets["GITHUB_USERNAME"]
    REPO_NAME = st.secrets["REPO_NAME"]
    BRANCH = st.secrets["BRANCH"]
    FILE_PATH = st.secrets["FILE_PATH"]

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{FILE_PATH}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = base64.b64decode(response.json()["content"]).decode()
        sha = response.json()["sha"]
        existing_df = pd.read_csv(io.StringIO(content))
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif response.status_code == 404:
        updated_df = new_df
        sha = None
    else:
        st.error(f"❌ Error fetching file: {response.json()}")
        return

    csv_data = updated_df.to_csv(index=False)
    encoded = base64.b64encode(csv_data.encode()).decode()

    payload = {
        "message": "Update feedback.csv from Streamlit app",
        "content": encoded,
        "branch": BRANCH
    }
    if sha:
        payload["sha"] = sha

    put_response = requests.put(url, headers=headers, data=json.dumps(payload))
    if put_response.status_code in [200, 201]:
        st.success("✅ Feedback successfully saved to GitHub!")
    else:
        st.error(f"❌ Failed to update GitHub: {put_response.json()}")

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
