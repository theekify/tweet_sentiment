import streamlit as st
import pickle

# Load model
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfdif.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Sentiment Analyzer")
st.title("Twitter Sentiment Analyzer")
st.write("Enter a tweet and get sentiment prediction")

tweet = st.text_area("Enter tweet text")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter some text")
    else:
        vector = vectorizer.transform([tweet.lower()])
        pred = model.predict(vector)[0]

        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Prediction: {mapping[pred]}")
