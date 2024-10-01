# Importing Necessary Libraries
import streamlit as st
import pickle
from preprocessor import text_transformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('punkt_tab')
encoder = LabelEncoder()
mnb = MultinomialNB()


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Classifier')

input_text = st.text_input('Enter Text to Classify')
if st.button('Classify'):
    # Preprocess
    transformed_text = text_transformer(input_text)
    transformed_text = [transformed_text]
    # Vectorize
    vector_input = tfidf.transform(transformed_text)
    # Predict
    result = model.predict(vector_input)
    # Display
    if result == 1:
        st.header('The text is Spam')
    else:
        st.header('The text is Not Spam')
