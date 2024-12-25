import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index=imdb.get_word_index()
model=load_model('simplernn.keras')

def preproces_text(sentence):
    encoded_words=[word_index.get(word,2)+3 for word in sentence.lower().split()]
    padded_word_encoding=sequence.pad_sequences([encoded_words],maxlen=500)
    return padded_word_encoding

def predict_sentiment(review):
    encoded_review=preproces_text(review)
    prediction=model.predict(encoded_review)
    sentiment="Negative"
    if(prediction[0][0]>0.5):
        sentiment="Positive"
    return sentiment, prediction[0][0]

st.title("Movie review sentiment analysis")
review = st.text_input("Please enter your review:")
st.write(f"Entered review: {review}")
sentiment, score = predict_sentiment(review)
st.write(f"Model score: {score}")
st.write(f"Model sentiment: {sentiment}")