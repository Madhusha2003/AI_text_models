import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ReceiptClassifier:
    """Handles receipt text line classification (TITLE, PRICE, etc.)."""
    def __init__(self, model_path=None):
        self.model = None
        self.label_map = {0: 'PRICE', 1: 'TITLE', 2: 'QUANTITY', 3: 'JUNK'}
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict(self, text):
        if not self.model: raise ValueError("Model not loaded.")
        prediction = self.model.predict([text])[0]
        return self.label_map.get(prediction, 'UNKNOWN')

    @staticmethod
    def train(X, y, save_path=None):
        label_map = {0: 'PRICE', 1: 'TITLE', 2: 'QUANTITY', 3: 'JUNK'}
        inv_label_map = {v: k for k, v in label_map.items()}
        y_encoded = [inv_label_map.get(str(label).upper(), 3) for label in y]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        pipeline.fit(X, y_encoded)
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Model saved to {save_path}")
        return pipeline

class ReceiptItemCategorizer:
    """Handles receipt item categorization (Electronics, Apparel, etc.)."""
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def predict(self, text):
        if not self.model: raise ValueError("Model not loaded.")
        return self.model.predict([text])[0]

    @staticmethod
    def train(X, y, save_path=None):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        pipeline.fit(X, y) # LogisticRegression handles string labels natively
        if save_path:
            joblib.dump(pipeline, save_path)
            print(f"Category model saved to {save_path}")
        return pipeline

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    line_model = os.path.join(project_root, "models", "receipt_classifier.pkl")
    cat_model = os.path.join(project_root, "models", "receipt_category_model.pkl")
    data_dir = os.path.join(project_root, "data", "receipt_ai")

    print("--- Receipt AI Tool ---")
    print("1. Train Models")
    print("2. Predict (Line Classification)")
    print("3. Predict (Item Categorization)")
    choice = input("Select option (1/2/3): ").strip()

    if choice == '1':
        sub_choice = input("Select Model (1 for Line Classifier, 2 for Item Categorizer): ").strip()
        if sub_choice == '1':
            line_data = os.path.join(data_dir, "synthetic_receipts.csv")
            if not os.path.exists(line_data):
                line_data = os.path.join(data_dir, "data03.csv")
            if os.path.exists(line_data):
                df = pd.read_csv(line_data).dropna()
                ReceiptClassifier.train(df['text'].tolist(), df['label'].tolist(), line_model)
            else:
                print(f"Error: Line training data not found in {data_dir}")
        elif sub_choice == '2':
            cat_data = os.path.join(data_dir, "synthetic_products_full.csv")
            if not os.path.exists(cat_data):
                cat_data = os.path.join(data_dir, "data_set.csv")
            if os.path.exists(cat_data):
                df = pd.read_csv(cat_data).dropna()
                ReceiptItemCategorizer.train(df['Product Name'].tolist(), df['Category'].tolist(), cat_model)
            else:
                print(f"Error: Category training data not found in {data_dir}")

    elif choice == '2':
        try:
            model = ReceiptClassifier(line_model)
            print("Type 'exit' to quit.")
            while True:
                text = input("\nEnter receipt text: ").strip()
                if text.lower() == 'exit': break
                print(f"Classification: {model.predict(text)}")
        except Exception as e:
            print(f"Error loading model: {e}")

    elif choice == '3':
        try:
            model = ReceiptItemCategorizer(cat_model)
            print("Type 'exit' to quit.")
            while True:
                text = input("\nEnter item name: ").strip()
                if text.lower() == 'exit': break
                print(f"Category: {model.predict(text)}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Invalid choice.")
