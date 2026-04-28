import os
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class SentimentModel:
    """Handles data processing and sentiment prediction."""
    
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, path):
        self.model = joblib.load(path)

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<br\s*/?>', ' ', text) 
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
        return text.strip()

    def predict(self, text):
        if not self.model:
            raise ValueError("Model not loaded.")
        
        cleaned = self.clean(text)
        prediction = self.model.predict([cleaned])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        if prediction == 0.5: sentiment = "Neutral"
        return sentiment

    def train(self, X, y, save_path=None):
        """Trains a new pipeline and optionally saves it."""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
            ('clf', LinearSVC())
        ])
        
        X_cleaned = [self.clean(t) for t in X]
        pipeline.fit(X_cleaned, y)
        self.model = pipeline
        
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Model saved to {save_path}")

# Standard Interface
def predict(text, model_path="movie_review_model.pkl"):
    engine = SentimentModel(model_path)
    return engine.predict(text)

def train(X, y, save_path="movie_review_model.pkl"):
    engine = SentimentModel()
    engine.train(X, y, save_path)
