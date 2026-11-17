# Library imports
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from numpy import array

from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template, session
from keras_preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords, words
from keras_preprocessing.text import tokenizer_from_json
import io
import json

POSITIVE_WORDS = {
    "amazing", "awesome", "excellent", "fantastic", "brilliant",
    "great", "good", "wonderful", "beautiful", "love", "loved",
    "enjoy", "enjoyed", "perfect", "best", "interesting",
    "fun", "impressive", "masterpiece"
}

NEGATIVE_WORDS = {
    "bad", "boring", "terrible", "worst", "awful", "poor",
    "hate", "hated", "disappointing", "disappointed",
    "slow", "waste", "horrible", "dull", "weak", "confusing"
}

stopwords_list = set(stopwords.words('english'))
maxlen = 100

# Load model
model_path ='c1_lstm_model_acc_0.864.h5'
pretrained_lstm_model = load_model(model_path)

# Loading
with open('b3_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)


# Create the app object
app = Flask(__name__)
app.secret_key = "sentiment-analysis-secret-key"


# creating function for data cleaning
from b2_preprocessing_function import CustomPreprocess
custom = CustomPreprocess()


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    query_asis = [str(x) for x in request.form.values()]

    # Extract words for review insights
    review_text = query_asis[0].lower()

    words = set(
    word.strip(".,!?;:'\"()[]{}")
    for word in review_text.split()
)

    positive_words = sorted(words.intersection(POSITIVE_WORDS))
    negative_words = sorted(words.intersection(NEGATIVE_WORDS))

    # Preprocess review
    query_processed_list = []

    for query in query_asis:
        query_processed = custom.preprocess_text(query)
        query_processed_list.append(query_processed)

    # Tokenize
    query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)

    # Pad sequence
    query_padded = pad_sequences(
        query_tokenized,
        padding='post',
        maxlen=maxlen
    )

    # LSTM prediction
    query_sentiments = pretrained_lstm_model.predict(
        query_padded,
        verbose=0
    )

    probability = float(query_sentiments[0][0])

    # Determine sentiment
    if probability > 0.5:
        sentiment = "Positive"
        confidence = probability * 100
    else:
        sentiment = "Negative"
        confidence = (1 - probability) * 100

    # Convert probability into 1–5 star rating
    rating = round(probability * 5)
    rating = max(1, min(5, rating))

    # Save analysis to session history
    history = session.get("history", [])

    history.append({
    "review": review_text,
    "sentiment": sentiment,
    "confidence": round(confidence, 2),
    "rating": rating
    })

# Keep only the latest 10 analyses
    session["history"] = history[-10:]
    session.modified = True

    stars = "⭐" * rating

    return render_template(
    'index.html',
    prediction_text=(
        f"{sentiment} Review | "
        f"Confidence: {confidence:.2f}% | "
        f"Predicted Rating: {stars} ({rating}/5)"
    ),
    sentiment=sentiment,
    confidence=confidence,
    rating=rating,
    stars=stars,
    positive_words=positive_words,
    negative_words=negative_words
    )
@app.route('/history')
def history():
    history_data = session.get("history", [])
    return render_template(
        'history.html',
        history=history_data
    )

if __name__ == "__main__":
    app.run(debug=True)
