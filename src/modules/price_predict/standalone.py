import os
import pandas as pd
from datetime import datetime
import model

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "lanka_food_model.pkl")
    map_path = os.path.join(base_dir, "commodity_map.csv")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))

    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "models", "lanka_food_model.pkl")
    if not os.path.exists(map_path):
        map_path = os.path.join(project_root, "data", "price_predict", "commodity_map.csv")

    print("--- Price Predictor Tool ---")
    print("1. Train Model")
    print("2. Predict Price")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        data_path = os.path.join(project_root, "data", "price_predict", "wfp_food_prices_sri_lanka_cleaned.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df['date'] = pd.to_datetime(df['date'])
            df['year'], df['month'], df['day'] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day
            model.train(df, model_path)
        else:
            print(f"Error: Training data not found at {data_path}")

    elif choice == '2':
        item = input("\nEnter commodity name: ").strip()
        try:
            price = model.predict(item, datetime.now(), model_path, map_path)
            print(f"Predicted Price: LKR {price:,.2f}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
