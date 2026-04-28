import os
import pandas as pd
import model
import model_category

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
    
    line_model = os.path.join(base_dir, "receipt_classifier.pkl")
    if not os.path.exists(line_model):
        line_model = os.path.join(project_root, "models", "receipt_classifier.pkl")
        
    cat_model = os.path.join(base_dir, "receipt_category_model.pkl")
    if not os.path.exists(cat_model):
        cat_model = os.path.join(project_root, "models", "receipt_category_model.pkl")

    print("--- Receipt AI Tool ---")
    print("1. Train Models")
    print("2. Predict (Line Classification)")
    print("3. Predict (Item Categorization)")
    choice = input("Select option (1/2/3): ").strip()

    data_dir = os.path.join(project_root, "data", "receipt_ai")

    if choice == '1':
        # Train Line Classifier
        line_data = os.path.join(data_dir, "synthetic_receipts.csv")
        if os.path.exists(line_data):
            df = pd.read_csv(line_data).dropna()
            model.train(df['text'].tolist(), df['label'].tolist(), line_model)
        
        # Train Item Categorizer
        cat_data = os.path.join(data_dir, "data_set.csv")
        if os.path.exists(cat_data):
            df = pd.read_csv(cat_data).dropna()
            model_category.train(df['Product Name'].tolist(), df['Category'].tolist(), cat_model)
        print("Training complete.")

    elif choice == '2':
        text = input("\nEnter receipt text: ").strip()
        try:
            print(f"Classification: {model.predict(text, line_model)}")
        except Exception as e:
            print(f"Error: {e}")

    elif choice == '3':
        text = input("\nEnter item name: ").strip()
        try:
            print(f"Category: {model_category.predict(text, cat_model)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
