import os
import pandas as pd
import joblib

class PriceModel:
    #Handles price data processing and prediction.#
    
    def __init__(self, model_path=None, map_path=None):
        self.model = None
        self.commodity_map = {}
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        
        if map_path and os.path.exists(map_path):
            self._load_map(map_path)

    def _load_map(self, path):
        df = pd.read_csv(path)
        df['commodity'] = df['commodity'].str.lower()
        self.commodity_map = dict(zip(df['commodity'], df['commodity_id']))

    def predict(self, item_name, target_date, market_id=1):
        if not self.model:
            raise ValueError("Model not loaded.")
        
        item_id = self.commodity_map.get(item_name.lower())
        if item_id is None:
            raise ValueError(f"Commodity '{item_name}' not found in mapping.")

        input_df = pd.DataFrame({
            'year': [target_date.year],
            'month': [target_date.month],
            'day': [target_date.day],
            'commodity_id': [item_id]
        })
        
        price = self.model.predict(input_df)[0]
        return price

# Standard Interface
def predict(item_name, target_date, model_path="lanka_food_model.pkl", map_path="commodity_map.csv"):
    engine = PriceModel(model_path, map_path)
    return engine.predict(item_name, target_date)
