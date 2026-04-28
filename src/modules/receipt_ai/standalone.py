import os
from model import predict

def main():
    model_path = "receipt_classifier.pkl"
    
    if not os.path.exists(model_path):
        model_path = os.path.join("..", "..", "..", "models", "receipt_classifier.pkl")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("--- Receipt Classifier Standalone Tool ---")
    text = input("Enter receipt line: ").strip()
    
    try:
        label = predict(text, model_path)
        print(f"Label: {label}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
