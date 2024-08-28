import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))


def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


english_punctuations = string.punctuation
punctuations_list = english_punctuations


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)


def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', data)


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


tokenizer = RegexpTokenizer(r'\w+')


def tokenize_text(data):
    return tokenizer.tokenize(data)


porter_stemmer = nltk.PorterStemmer()  # Renamed variable


def stemming_on_text(data):
    return [porter_stemmer.stem(word) for word in data]


lemmatizer = nltk.WordNetLemmatizer()  # Renamed variable


def lemmatizer_on_text(data):
    return [lemmatizer.lemmatize(word) for word in data]


model = load_model('model.h5')

st.header('Twitter Sentiment Analysis')

label = st.selectbox('Label', [0, 1])
time = st.text_input('Time')
date = st.text_input('Date')
query = st.text_input('Query')
username = st.text_input('Username')
text = st.text_area('Text')

if st.button('Submit'):
    user_data = pd.DataFrame([[label, time, date, query, username, text]],
                             columns=["label", "time", "date", "query", "username", "text"])

    user_data['text'] = user_data['text'].str.lower()
    user_data['text'] = user_data['text'].apply(cleaning_stopwords)
    user_data['text'] = user_data['text'].apply(cleaning_punctuations)
    user_data['text'] = user_data['text'].apply(cleaning_repeating_char)
    user_data['text'] = user_data['text'].apply(cleaning_email)
    user_data['text'] = user_data['text'].apply(cleaning_URLs)
    user_data['text'] = user_data['text'].apply(cleaning_numbers)
    user_data['text'] = user_data['text'].apply(tokenize_text)
    user_data['text'] = user_data['text'].apply(stemming_on_text)
    user_data['text'] = user_data['text'].apply(lemmatizer_on_text)

    st.write('Preprocessed Text:', ' '.join(user_data['text'].iloc[0]))

    X = user_data['text'].apply(lambda x: ' '.join(x))
    max_len = 500
    tok = Tokenizer(num_words=2000)
    tok.fit_on_texts(X)
    sequences = tok.texts_to_sequences(X)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    prediction = model.predict(sequences_matrix)
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'

    st.write('Model Prediction:', sentiment)
