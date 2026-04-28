import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ReceiptItemCategorizer:
    """Handles receipt item categorization into specific departments."""
    
    categories = [
        'Electronics', 'Apparel and Fashion', 'Kitchen Appliances', 
        'Health and Beauty', 'Books and Media', 'Toys and Games', 
        'Sports and Outdoor Equipment', 'Furniture and Home Decor', 
        'Pet Supplies', 'Office Supplies'
    ]

    def __init__(self, model_path=None):
        self.model = None
        self.label_map = {i: cat for i, cat in enumerate(self.categories)}
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict(self, text):
        if not self.model:
            raise ValueError("Model not loaded.")
        
        prediction = self.model.predict([text])[0]
        return self.label_map.get(prediction, "Unknown")

    def train(self, X, y, save_path=None):
        """Trains a new pipeline and saves the model."""
        inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Encode labels, default to index 0 if not found
        y_encoded = []
        for label in y:
            if isinstance(label, str):
                y_encoded.append(inv_label_map.get(label, 0))
            else:
                y_encoded.append(label)

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        pipeline.fit(X, y_encoded)
        self.model = pipeline
        
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Category model saved to {save_path}")

# Standard Interface
def predict(text, model_path="receipt_category_model.pkl"):
    engine = ReceiptItemCategorizer(model_path)
    return engine.predict(text)

def train(X, y, save_path="receipt_category_model.pkl"):
    engine = ReceiptItemCategorizer()
    engine.train(X, y, save_path)