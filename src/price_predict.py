import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

class PriceModel:
    def __init__(self, model_path=None, map_path=None):
        self.model = None
        self.commodity_map = {}
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        
        if map_path and os.path.exists(map_path):
            self._load_map(map_path)

    def _load_map(self, path):
        if not os.path.exists(path): return
        df = pd.read_csv(path)
        df['commodity'] = df['commodity'].str.lower()
        self.commodity_map = dict(zip(df['commodity'], df['commodity_id']))

    def predict(self, item_name, target_date):
        if not self.model: raise ValueError("Model not loaded.")
        item_id = self.commodity_map.get(item_name.lower())
        if item_id is None: raise ValueError(f"Commodity '{item_name}' not found in mapping.")

        input_df = pd.DataFrame({
            'year': [target_date.year],
            'month': [target_date.month],
            'day': [target_date.day],
            'commodity_id': [item_id]
        })
        return self.model.predict(input_df)[0]

    @staticmethod
    def train(df, save_path=None):
        X = df[['year', 'month', 'day', 'commodity_id']]
        y = df['price']
        
        model = RandomForestRegressor(n_estimators=100, random_state=1)
        model.fit(X, y)
        
        if save_path:
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")
        return model

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    # Simple defaults
    model_path = os.path.join(project_root, "models", "lanka_food_model.pkl")
    map_path = os.path.join(project_root, "data", "price_predict", "commodity_map.csv")
    data_path = os.path.join(project_root, "data", "price_predict", "wfp_food_prices_sri_lanka_cleaned.csv")

    print("--- Price Predictor Tool ---")
    print("1. Train Model")
    print("2. Predict Price")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df['year'], df['month'], df['day'] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day
            PriceModel.train(df, model_path)
        else:
            print(f"Error: Training data not found at {data_path}")

    elif choice == '2':
        try:
            model = PriceModel(model_path, map_path)
            print("Type 'exit' to quit.")
            while True:
                item = input("\nEnter commodity name: ").strip()
                if item.lower() == 'exit': break
                try:
                    price = model.predict(item, datetime.now())
                    print(f"Predicted Price: LKR {price:,.2f}")
                except Exception as e:
                    print(f"Error: {e}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Invalid choice.")
