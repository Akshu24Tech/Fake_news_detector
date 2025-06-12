import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

nltk.download('punkt')
nltk.download('stopwords')

def simple_preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

@st.cache_resource(show_spinner=False)
def load_model_and_metrics():
    # Load Data
    true_news = pd.read_csv('True.csv')
    fake_news = pd.read_csv('Fake.csv')
    true_news['label'] = 0
    fake_news['label'] = 1
    df = pd.concat([true_news, fake_news], ignore_index=True)
    df['full_text'] = df['title'] + ' ' + df['text']
    df['clean_text'] = df['full_text'].apply(simple_preprocess)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, digits=4, output_dict=True)
    return model, acc, report

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")

st.write("""
Enter a news article (title and/or content) below to check if it's likely to be **Fake** or **Real**.
""")

model, acc, report = load_model_and_metrics()

user_input = st.text_area("Paste news title and content here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = simple_preprocess(user_input)
        pred = model.predict([clean])[0]
        proba = model.predict_proba([clean])[0]
        label = "Fake" if pred == 1 else "Real"
        st.markdown(f"### Prediction: {'🟥' if pred==1 else '🟩'} **{label} News**")
        st.write(f"Confidence: {np.max(proba)*100:.2f}%")

st.markdown("---")
st.subheader("Model Performance")
st.write(f"**Accuracy:** {acc:.4f}")
st.markdown("**Classification Report (sample):**")
st.json({k: report[k] for k in ['0', '1', 'accuracy']}) 