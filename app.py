import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# Title
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline and check if it's **FAKE** or **REAL**.")

# Input box
user_input = st.text_area("Enter News Text")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.error("🚨 This is likely **FAKE NEWS**.")
        else:
            st.success("✅ This appears to be **REAL NEWS**.")
