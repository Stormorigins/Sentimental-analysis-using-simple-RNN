#load libraies

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load index
word_index=imdb.get_word_index()
reverse_index={j:i for i,j in word_index.items()}#dict comprehension

#model load
model=load_model("simple_rnn_imdb.h5")

#Helper functions
def preprocess_text(text):
    words=text.lower().split()
    encode=[word_index.get(i,2) + 3 for i in words]
    padding=sequence.pad_sequences([encode],maxlen=500)
    return padding

def decoded_review (x):
    return ' '.join([reverse_index.get(i - 3, '?') for i in x])

#prediction functions

def predic_sentiment(review):
    processed_input= preprocess_text(review)
    prediction=model.predict(processed_input)
    sentiment="Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment,prediction[0][0]

#Stremlit

import streamlit as st
st.title("IMDB Movie review sentiment analysis")
st.write("Enter the movie review to classify its as Positive or Negative.")

#user input
input=st.text_area("Review")
if st.button("Classify"):
    sentiment,prediction=predic_sentiment(input)
    st.write(f"Sentiment:{sentiment}")
    st.write(f"prediction score:{prediction}")
else:
    st.write("Please enetr a movie review.")
