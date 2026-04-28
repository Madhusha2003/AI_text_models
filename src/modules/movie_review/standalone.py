import os
from model import predict

def main():
    # To run this standalone, copy 'movie_review_model.pkl' to this folder
    model_path = "movie_review_model.pkl"
    
    # Fallback to main project models/ for development
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "..", "models", "movie_review_model.pkl")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("--- Movie Review Standalone Tool ---")
    while True:
        text = input("\nEnter review (or 'exit'): ").strip()
        if text.lower() == 'exit':
            break
        
        try:
            sentiment = predict(text, model_path=model_path)
            print(f"Sentiment: {sentiment}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
