from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from pathlib import Path


# Get the directory where this script is located
BASE_DIR = Path(__file__).parent

with open(BASE_DIR / 'lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(BASE_DIR / 'tfdif.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = FastAPI()


class Tweet(BaseModel):
    text: str


@app.post("/predict")
def predict_sentiment(tweet: Tweet):
    cleaned_text = tweet.text.lower()  # simple cleaning
    vector = vectorizer.transform([cleaned_text])
    pred = model.predict(vector)[0]
    
    # Handle both numeric and string predictions
    if isinstance(pred, str):
        return {"prediction": pred.capitalize()}
    else:
        mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return {"prediction": mapping.get(pred, "Unknown")}
