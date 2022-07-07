import streamlit as st
import pickle
import string
#from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

model = pickle.load(open('model.pkl','rb'))

st.title("Tweet Classifier")

input_tweet = st.text_area("Enter the Tweet")

input = [input_tweet]

if st.button('Predict'):

    result = model.predict(input)[0]
    result = result.argmax()

    classify={'angry': 0, 'disgust': 1, 'disgust|angry': 2, 'happy': 3, 'happy|sad': 4, 'happy|surprise': 5, 'not-relevant': 6, 'sad': 7, 'sad|angry': 8, 'sad|disgust': 9, 'sad|disgust|angry': 10, 'surprise': 11}

    new_dict = dict([(value, key) for key, value in classify.items()])

    st.header(new_dict[result])