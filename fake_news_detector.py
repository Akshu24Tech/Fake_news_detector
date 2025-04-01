import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None  # Will be initialized in prepare_data
        self.classifier = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)

    def prepare_data(self, data_path):
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Print value counts of subjects to understand distribution
        print("\nSubject distribution:")
        print(df['subject'].value_counts())
        
        # Create binary labels based on subject
        # Assuming subjects like 'politics', 'left-news', 'Government News' are real (0)
        # and others like 'fake', 'conspiracy', etc. are fake (1)
        real_news_subjects = ['politicsNews', 'Government News', 'left-news', 'US_News', 'World News']
        df['label'] = df['subject'].apply(lambda x: 0 if x in real_news_subjects else 1)
        
        print("\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Combine title and text for better prediction
        df['full_text'] = df['title'] + ' ' + df['text']
        
        # Preprocess the combined text
        df['processed_text'] = df['full_text'].apply(self.preprocess_text)
        
        # Balance the dataset
        min_class_count = min(df['label'].value_counts())
        df_balanced = pd.concat([
            df[df['label'] == 0].sample(min_class_count, random_state=42),
            df[df['label'] == 1].sample(min_class_count, random_state=42)
        ])
        
        # Split the balanced data
        X_train, X_test, y_train, y_test = train_test_split(
            df_balanced['processed_text'], 
            df_balanced['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df_balanced['label']  # Ensure balanced split
        )
        
        # TF-IDF Vectorization with improved parameters
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        return X_train_vectorized, X_test_vectorized, y_train, y_test

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, digits=4)
        
        # Add more detailed evaluation
        print("\nPrediction distribution:")
        print(pd.Series(predictions).value_counts())
        
        return accuracy, report

    def predict(self, text, title):
        # Combine title and text for prediction
        full_text = f"{title} {text}"
        processed_text = self.preprocess_text(full_text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(vectorized_text)
        probability = self.classifier.predict_proba(vectorized_text)
        
        return prediction[0], probability[0]

def main():
    # Initialize the detector
    detector = FakeNewsDetector()
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = detector.prepare_data('Fake.csv')
    
    print("\nTraining model...")
    detector.train(X_train, y_train)
    
    print("\nEvaluating model...")
    accuracy, report = detector.evaluate(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Example predictions
    test_cases = [
        {
            "title": "Major Scientific Discovery Announced",
            "text": "Scientists at CERN have announced a breakthrough in particle physics research."
        },
        {
            "title": "Shocking Political Scandal Exposed",
            "text": "Anonymous sources reveal controversial documents about government officials."
        }
    ]
    
    print("\nTest Predictions:")
    for case in test_cases:
        prediction, probability = detector.predict(text=case['text'], title=case['title'])
        print(f"\nTitle: {case['title']}")
        print(f"Text: {case['text']}")
        print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
        print(f"Confidence: {max(probability):.4f}")

if __name__ == "__main__":
        main() 
