import argparse
import sys
import os
from datetime import datetime
import pandas as pd

# Add root to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.price_predict as price_predict
import src.movie_review as movie_review
import src.receipt_ai as receipt_ai
import src.data_scraper as data_scraper

from src.core.config import (
    PRICE_MODEL_PATH, PRICE_DATA_DIR, 
    MOVIE_MODEL_PATH, MOVIE_DATA_DIR,
    RECEIPT_DATA_DIR, RECEIPT_MODEL_PATH,
    RECEIPT_CATEGORY_MODEL_PATH
)

def handle_price(args):
    map_path = os.path.join(PRICE_DATA_DIR, 'commodity_map.csv')
    target_date = datetime.now()
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Using today.")

    try:
        model = price_predict.PriceModel(PRICE_MODEL_PATH, map_path)
        price = model.predict(args.item, target_date)
        print(f"\n--- Price Prediction ---")
        print(f"Item: {args.item}")
        print(f"Date: {target_date.strftime('%Y-%m-%d')}")
        print(f"Predicted Price: LKR {price:,.2f}")
    except Exception as e:
        print(f"Error: {e}")

def handle_movie(args):
    if args.train:
        data_path = os.path.join(MOVIE_DATA_DIR, 'cleaned_imdb_dataset.csv')
        if not os.path.exists(data_path):
            print(f"Training data not found at {data_path}")
            return
        df = pd.read_csv(data_path).dropna()
        movie_review.SentimentModel.train(df['review'].tolist(), df['sentiment'].tolist(), MOVIE_MODEL_PATH)
    else:
        try:
            model = movie_review.SentimentModel(MOVIE_MODEL_PATH)
            sentiment = model.predict(args.text)
            print(f"\n--- Sentiment Analysis ---")
            print(f"Review: {args.text[:50]}...")
            print(f"Sentiment: {sentiment}")
        except Exception as e:
            print(f"Error: {e}")

def handle_receipt(args):
    if args.train:
        data_path = os.path.join(RECEIPT_DATA_DIR, 'synthetic_receipts.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(RECEIPT_DATA_DIR, 'data03.csv')
            
        if not os.path.exists(data_path):
            print(f"Training data not found in {RECEIPT_DATA_DIR}")
            return
            
        df = pd.read_csv(data_path).dropna()
        receipt_ai.ReceiptClassifier.train(df['text'].tolist(), df['label'].tolist(), RECEIPT_MODEL_PATH)
    else:
        try:
            model = receipt_ai.ReceiptClassifier(RECEIPT_MODEL_PATH)
            label = model.predict(args.text)
            print(f"\n--- Receipt Classification ---")
            print(f"Text: {args.text}")
            print(f"Label: {label}")
        except Exception as e:
            print(f"Error: {e}")

def handle_receipt_category(args):
    if args.train:
        data_path = os.path.join(RECEIPT_DATA_DIR, 'synthetic_products_full.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(RECEIPT_DATA_DIR, 'data_set.csv')
            
        if not os.path.exists(data_path):
            print(f"Training data not found in {RECEIPT_DATA_DIR}")
            return
            
        df = pd.read_csv(data_path).dropna()
        receipt_ai.ReceiptItemCategorizer.train(df['Product Name'].tolist(), df['Category'].tolist(), RECEIPT_CATEGORY_MODEL_PATH)
    else:
        try:
            model = receipt_ai.ReceiptItemCategorizer(RECEIPT_CATEGORY_MODEL_PATH)
            cat = model.predict(args.text)
            print(f"\n--- Receipt Item Categorization ---")
            print(f"Text: {args.text}")
            print(f"Category: {cat}")
        except Exception as e:
            print(f"Error: {e}")

def handle_scrape(args):
    scraper = data_scraper.WebScraper()
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
    receipt_parser = subparsers.add_parser("receipt", help="Classify receipt text (TITLE, PRICE, etc.)")
    receipt_parser.add_argument("--text", help="Receipt text to classify")
    receipt_parser.add_argument("--train", action="store_true", help="Train the model")

    receipt_cat_parser = subparsers.add_parser("receipt_category", help="Categorize receipt items (Electronics, etc.)")
    receipt_cat_parser.add_argument("--text", help="Item text to categorize")
    receipt_cat_parser.add_argument("--train", action="store_true", help="Train the model")

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
    elif args.command == "receipt_category":
        handle_receipt_category(args)
    elif args.command == "scrape":
        handle_scrape(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
