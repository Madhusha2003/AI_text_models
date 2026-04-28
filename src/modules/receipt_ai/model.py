import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ReceiptClassifier:
    #Handles receipt text classification.#
    
    def __init__(self, model_path=None):
        self.model = None
        self.label_map = {0: 'PRICE', 1: 'TITLE', 2: 'QUANTITY', 3: 'JUNK'}
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict(self, text):
        if not self.model:
            raise ValueError("Model not loaded.")
        
        prediction = self.model.predict([text])[0]
        return self.label_map.get(prediction, 'UNKNOWN')

    def train(self, X, y, save_path=None):
        #Trains a new pipeline.#
        inv_label_map = {v: k for k, v in self.label_map.items()}
        y_encoded = [inv_label_map.get(label.upper(), 3) if isinstance(label, str) else label for label in y]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        pipeline.fit(X, y_encoded)
        self.model = pipeline
        
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Model saved to {save_path}")

# Standard Interface
def predict(text, model_path="receipt_classifier.pkl"):
    engine = ReceiptClassifier(model_path)
    return engine.predict(text)
