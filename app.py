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

        # Handle both string and numeric predictions
        if isinstance(pred, str):
            sentiment = pred.capitalize()
        else:
            mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = mapping.get(pred, "Unknown")
        
        # Display with appropriate styling
        if sentiment == "Positive":
            st.success(f"Prediction: {sentiment}")
        elif sentiment == "Negative":
            st.error(f"Prediction: {sentiment}")
        else:
            st.info(f"Prediction: {sentiment}")
