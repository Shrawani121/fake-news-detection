import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline and check if it's **FAKE** or **REAL**.")

user_input = st.text_area("Enter News Text")

# Predict
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned_input = clean_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.error("🚨 This is likely **FAKE NEWS**.")
        else:
            st.success("✅ This appears to be **REAL NEWS**.")
