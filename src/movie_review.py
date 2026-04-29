import os
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class SentimentModel:
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def clean(self, text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'<br\s*/?>', ' ', text) 
        text = re.sub(r'[^a-zA-Z\s]', '', text) 
        return text.strip()

    def predict(self, text):
        if not self.model: raise ValueError("Model not loaded.")
        cleaned = self.clean(text)
        prediction = self.model.predict([cleaned])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        if prediction == 0.5: sentiment = "Neutral"
        return sentiment

    @staticmethod
    def train(X, y, save_path=None):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),
            ('clf', LinearSVC())
        ])
        # Need to clean before training, so we instantiate locally
        dummy_model = SentimentModel()
        X_cleaned = [dummy_model.clean(t) for t in X]
        pipeline.fit(X_cleaned, y)
        
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Model saved to {save_path}")
        return pipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Simple defaults
    model_path = os.path.join(project_root, "models", "movie_review_model.pkl")
    data_path = os.path.join(project_root, "data", "movie_review", "cleaned_imdb_dataset.csv")

    print("--- Movie Review Tool ---")
    print("1. Train Model")
    print("2. Predict Sentiment")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        if os.path.exists(data_path):
            df = pd.read_csv(data_path).dropna()
            SentimentModel.train(df['review'].tolist(), df['sentiment'].tolist(), model_path)
        else:
            print(f"Error: Training data not found at {data_path}")

    elif choice == '2':
        try:
            model = SentimentModel(model_path)
            print("Type 'exit' to quit.")
            while True:
                text = input("\nEnter review text: ").strip()
                if text.lower() == 'exit': break
                print(f"Sentiment: {model.predict(text)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice.")
