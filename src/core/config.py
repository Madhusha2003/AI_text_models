import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model Paths
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, 'lanka_food_model.pkl')
MOVIE_MODEL_PATH = os.path.join(MODELS_DIR, 'movie_review_model.pkl') # We'll create this

# Data Paths
PRICE_DATA_DIR = os.path.join(DATA_DIR, 'price_predict')
MOVIE_DATA_DIR = os.path.join(DATA_DIR, 'movie_review')
RECEIPT_DATA_DIR = os.path.join(DATA_DIR, 'receipt_ai')
SCRAPER_DATA_DIR = os.path.join(DATA_DIR, 'scraper')
