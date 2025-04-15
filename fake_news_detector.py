import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

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

# Load Data
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')
true_news['label'] = 0
fake_news['label'] = 1
df = pd.concat([true_news, fake_news], ignore_index=True)

# Combine and Preprocess
df['full_text'] = df['title'] + ' ' + df['text']
df['clean_text'] = df['full_text'].apply(simple_preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Build Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
    ('clf', MultinomialNB())  # Or use LogisticRegression(max_iter=200)
])

# Train & Evaluate
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, digits=4))