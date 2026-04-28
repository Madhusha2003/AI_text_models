import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from src.core.config import SCRAPER_DATA_DIR

class WebScraper:
    def __init__(self, base_url="http://books.toscrape.com/"):
        self.base_url = base_url

    def scrape_books(self, num_pages=1):
        #Scrapes book data from books.toscrape.com.#
        books = []
        for page in range(1, num_pages + 1):
            url = f"{self.base_url}catalogue/page-{page}.html"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            
            for article in soup.find_all("article", class_="product_pod"):
                title = article.h3.a["title"]
                price = article.find("p", class_="price_color").text
                rating = article.p["class"][1]
                availability = article.find("p", class_="instock availability").text.strip()
                
                books.append({
                    "title": title,
                    "price": price,
                    "rating": rating,
                    "availability": availability
                })
        return books

    def save_raw(self, data, filename="books_raw.csv"):
        df = pd.DataFrame(data)
        path = os.path.join(SCRAPER_DATA_DIR, filename)
        df.to_csv(path, index=False)
        return path
