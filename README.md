# 📰 Fake News Detection using NLP + ML

Detect whether news content is real or fake using TF-IDF and machine learning models.

## 📊 Accuracy
- Random Forest: 99.77% (Selected Model)
- Logistic Regression: 98.7%
- XGBoost: 99.71%

## 🧰 Tech Stack
- Python, Scikit-learn, NLTK, XGBoost, Streamlit

## 🚀 How it works
1. Input news content
2. Text is cleaned and vectorized using TF-IDF
3. Model predicts whether it's fake or real

## 🖥️ Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
