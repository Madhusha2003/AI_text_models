# 🧠 Unified AI Model Collection

A modular collection of lightweight AI models for text processing, prediction, and web automation.
Designed with a unified CLI interface for easy integration and extensibility.

## ✨ Key Features
- **📊 Price Prediction Model** – Predict food prices in Sri Lanka using historical trends
- **🎬 Sentiment Analysis Model** – Classify movie reviews as Positive / Negative
- **🧾 Web Scraper Tool** – Extract structured book data from online sources
- **⚙️ Unified CLI Interface** – Access all models from a single command-line tool
- **🧩 Modular Architecture** – Each model is independently maintainable and reusable

## 🏗️ Architecture Overview

The project follows a layered design:
- `src/core/` → Shared utilities and base model logic
- `src/modules/` → Independent AI models (plug-and-play structure)
- `src/cli.py` → Unified interface to access all models
- `models/` → Stored trained model artifacts
- `data/` → Training datasets

## ⚙️ Installation
1. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate:**
   ```bash
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

All models are accessible via a unified CLI:

### 📊 Price Prediction
```bash
python src/cli.py price --item "rice (white)" --date "2024-05-01"
```

### 🎬 Sentiment Analysis
**Train model:**
```bash
python src/cli.py movie --train
```

**Predict:**
```bash
python src/cli.py movie --text "This movie was absolutely fantastic!"
```

### 🧾 Web Scraping
```bash
python src/cli.py scrape --pages 2
```

## 📁 Project Structure
```text
data/           # datasets
models/         # trained models
src/
 ├── core/      # shared logic (base classes, utils)
 ├── modules/   # independent AI models
 ├── cli.py     # unified command interface
```

## 🧩 Portability & Standalone Usage

Each model in this collection is designed to be **portable**. You can copy an individual model folder from `src/modules/` to any other project and use it independently.

### How to use a model standalone:
1. **Copy the module folder** (e.g., `src/modules/movie_review`).
2. **Copy the model file** (e.g., `models/movie_review_model.pkl`) into that folder.
3. **Run the `standalone.py`** script provided in the folder:
   ```bash
   cd movie_review
   python standalone.py
   ```

The code includes fallbacks to ensure that even if the shared `core` logic is missing, the model remains functional.

## 💡 Design Philosophy
This project is built with:
- **Modularity** → Each AI model is independent
- **Reusability** → Shared core logic across models
- **Extensibility** → New models can be added easily
- **Unified Interface** → Single CLI entry point for all tools

## 🔮 Future Improvements
- [ ] FastAPI backend for web integration
- [ ] Model versioning system
- [ ] Docker deployment support
- [ ] REST API endpoints for each model
- [ ] Mobile app integration (Android / Flutter)
