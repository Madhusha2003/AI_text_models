import os
import pandas as pd
import model

def main():
    # Smart path detection
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "movie_review_model.pkl")
    
    # Fallback to project models/ folder
    if not os.path.exists(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        model_path = os.path.join(project_root, "models", "movie_review_model.pkl")

    print("--- Movie Review Model Tool ---")
    print("1. Train Model")
    print("2. Predict Sentiment")
    choice = input("Select option (1/2): ").strip()

    if choice == '1':
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        data_path = os.path.join(project_root, "data", "movie_review", "cleaned_imdb_dataset.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path).dropna()
            model.train(df['review'].tolist(), df['sentiment'].tolist(), model_path)
        else:
            print(f"Error: Training data not found at {data_path}")

    elif choice == '2':
        text = input("\nEnter review text: ").strip()
        try:
            print(f"Sentiment: {model.predict(text, model_path)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
