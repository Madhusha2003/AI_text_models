import joblib
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import PRICE_MODEL_PATH

model = joblib.load(PRICE_MODEL_PATH)
if hasattr(model, 'feature_names_in_'):
    print(f"Expected features: {model.feature_names_in_}")
else:
    print("Model does not have feature_names_in_ attribute.")

if hasattr(model, 'n_features_in_'):
    print(f"Number of features: {model.n_features_in_}")
