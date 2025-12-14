# 🧠 Twitter Sentiment Analyzer

A Machine Learning web application that predicts the sentiment of tweets
(Negative, Neutral, Positive) using Natural Language Processing (NLP).

## 🚀 Live Demo
https://huggingface.co/spaces/theekaka/twitter-sentiment-analyzer

## 📌 Features
- Text preprocessing and cleaning
- TF-IDF vectorization
- Logistic Regression classifier
- Interactive Streamlit web interface
- Deployed on HuggingFace Spaces

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- HuggingFace Spaces

## 📊 Model Performance
- Accuracy: **~80%**
- Evaluation metrics: Precision, Recall, F1-score

## ⚙️ How It Works
1. User enters a tweet
2. Text is cleaned and vectorized
3. ML model predicts sentiment
4. Result is displayed in the UI

## 🧪 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
