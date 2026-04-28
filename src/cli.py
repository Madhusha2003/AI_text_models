import argparse
import sys
import os
from datetime import datetime

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simplified Imports from modules
import src.modules.price_predict.model as price_model
import src.modules.movie_review.model as movie_model
import src.modules.receipt_ai.model as receipt_model
from src.modules.data_scraper.scraper import WebScraper

from src.core.config import (
    PRICE_MODEL_PATH, PRICE_DATA_DIR, 
    MOVIE_MODEL_PATH, MOVIE_DATA_DIR,
    RECEIPT_DATA_DIR
)
from src.modules.receipt_ai.model import ReceiptClassifier # For training

# Local path for receipt model in config
RECEIPT_MODEL_PATH = os.path.join(os.path.dirname(PRICE_MODEL_PATH), 'receipt_classifier.pkl')

def handle_price(args):
    map_path = os.path.join(PRICE_DATA_DIR, 'commodity_map.csv')
    target_date = datetime.now()
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Using today.")

    try:
        price = price_model.predict(args.item, target_date, PRICE_MODEL_PATH, map_path)
        print(f"\n--- Price Prediction ---")
        print(f"Item: {args.item}")
        print(f"Date: {target_date.strftime('%Y-%m-%d')}")
        print(f"Predicted Price: LKR {price:,.2f}")
    except Exception as e:
        print(f"Error: {e}")

def handle_movie(args):
    if args.train:
        import pandas as pd
        data_path = os.path.join(MOVIE_DATA_DIR, 'cleaned_imdb_dataset.csv')
        if not os.path.exists(data_path):
            print(f"Training data not found at {data_path}")
            return
        df = pd.read_csv(data_path).dropna()
        engine = movie_model.SentimentModel()
        engine.train(df['review'].tolist(), df['sentiment'].tolist(), MOVIE_MODEL_PATH)
    else:
        try:
            sentiment = movie_model.predict(args.text, MOVIE_MODEL_PATH)
            print(f"\n--- Sentiment Analysis ---")
            print(f"Review: {args.text[:50]}...")
            print(f"Sentiment: {sentiment}")
        except Exception as e:
            print(f"Error: {e}")

def handle_receipt(args):
    if args.train:
        import pandas as pd
        data_path = os.path.join(RECEIPT_DATA_DIR, 'synthetic_receipts.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(RECEIPT_DATA_DIR, 'data03.csv')
            
        if not os.path.exists(data_path):
            print(f"Training data not found in {RECEIPT_DATA_DIR}")
            return
            
        df = pd.read_csv(data_path).dropna()
        engine = ReceiptClassifier()
        engine.train(df['text'].tolist(), df['label'].tolist(), RECEIPT_MODEL_PATH)
    else:
        try:
            label = receipt_model.predict(args.text, RECEIPT_MODEL_PATH)
            print(f"\n--- Receipt Classification ---")
            print(f"Text: {args.text}")
            print(f"Label: {label}")
        except Exception as e:
            print(f"Error: {e}")

def handle_scrape(args):
    scraper = WebScraper()
    print(f"Scraping {args.pages} pages...")
    data = scraper.scrape_books(num_pages=args.pages)
    path = scraper.save_raw(data)
    print(f"Saved {len(data)} books to {path}")

def main():
    parser = argparse.ArgumentParser(description="Unified AI Model Collection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available models/tools")

    # Price Predictor
    price_parser = subparsers.add_parser("price", help="Predict food prices")
    price_parser.add_argument("--item", required=True, help="Commodity name")
    price_parser.add_argument("--date", help="Target date (YYYY-MM-DD)")

    # Movie Review
    movie_parser = subparsers.add_parser("movie", help="Analyze movie review sentiment")
    movie_parser.add_argument("--text", help="Review text to analyze")
    movie_parser.add_argument("--train", action="store_true", help="Train the model")

    # Receipt AI
    receipt_parser = subparsers.add_parser("receipt", help="Classify receipt text")
    receipt_parser.add_argument("--text", help="Receipt text to classify")
    receipt_parser.add_argument("--train", action="store_true", help="Train the model")

    # Scraper
    scraper_parser = subparsers.add_parser("scrape", help="Scrape book data")
    scraper_parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")

    args = parser.parse_args()

    if args.command == "price":
        handle_price(args)
    elif args.command == "movie":
        handle_movie(args)
    elif args.command == "receipt":
        handle_receipt(args)
    elif args.command == "scrape":
        handle_scrape(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
