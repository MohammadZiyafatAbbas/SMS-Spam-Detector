# 🛡️ SMS Spam Detector – Streamlit App for Spam Detection

**SMS Spam Detector** is a beginner-friendly **Streamlit web application** that detects spam SMS messages using machine learning and natural language processing (NLP). It provides an intuitive interface for classifying messages as "Spam" or "Ham (Not Spam)" and allows users to provide real-time feedback to improve the model.

![App Screenshot](img/app.png)

## 🚀 Features

- 🔍 Spam detection using a trained Naive Bayes classifier.
- 🧠 Built-in feedback mechanism for user corrections.
- 📊 Option to retrain the model using user-submitted feedback.
- 🌐 Deployed using Streamlit – no coding needed to use!
- 💡 Clean UI with custom CSS.

---

## 🧰 Tech Stack

- **Python**
- **scikit-learn**
- **SpaCy NLP**
- **Streamlit**
- **clean-text**
- **pandas**
- **TfidfVectorizer**
- **Naive Bayes Classifier**
- **Streamlit for UI**

---

## 📂 Project Structure

sms-spam-detector/
├── app.py  # Main Streamlit app

├── train.py  # Training script for base model
├── feedback_train.py  # Retrain model with feedback
├── requirements.txt  # Dependencies
├── styles.css  # Custom styles
├── model/
│ ├── spam_classifier.pkl
│ └── vectorizer.pkl
├── data/
│ ├── spam.csv  # Original training data
│ └── feedback.csv  # User feedback data
└── README.md


---

## 🛠️ Local Setup Instructions

To run the SMS Spam Shield on your local machine:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sms-spam-detector.git
cd sms-spam-detector

python -m venv venv
# Activate venv:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

streamlit run app.py

python train.py  # Train using base dataset
python feedback_train.py  # Retrain with feedback

---

🙋‍♂️ Feedback & Contribution
Have suggestions or want to contribute?

Fork this repo

Create a branch: git checkout -b feature/your-feature

Push to your branch: git push origin feature/your-feature

Open a Pull Request

---

🤖 Keywords for SEO
sms spam detection, spam sms classifier, streamlit spam detector, python sms spam app, ham or spam message filter, naive bayes sms spam, nlp sms classification, retrainable spam classifier, machine learning spam detection project, github sms spam detection

---

📄 License
This project is licensed under the MIT License.

---

📫 Contact
Made by Mohammad Ziyafat Abbas


---

Let me know if you'd like help customizing this with your name, GitHub URL, or adding a `.gitignore` and license file for publishing on GitHub.
