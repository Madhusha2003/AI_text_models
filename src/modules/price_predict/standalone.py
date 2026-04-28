import os
from datetime import datetime
from model import predict

def main():
    model_path = "lanka_food_model.pkl"
    map_path = "commodity_map.csv"
    
    # Development fallbacks
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "..", "models", "lanka_food_model.pkl")
    if not os.path.exists(map_path):
        map_path = os.path.join("..", "..", "..", "data", "price_predict", "commodity_map.csv")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("--- Price Predictor Standalone Tool ---")
    item = input("Enter commodity name (e.g., 'rice (white)'): ").strip()
    
    try:
        price = predict(item, datetime.now(), model_path, map_path)
        print(f"Predicted Price: LKR {price:,.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
