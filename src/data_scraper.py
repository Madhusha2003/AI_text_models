import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse

class WebScraper:
    def __init__(self, base_url="http://books.toscrape.com/"):
        self.base_url = base_url

    def scrape_books(self, num_pages=1):
        """Scrapes book data from books.toscrape.com."""
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)
        data_dir = os.path.join(project_root, "data", "scraper")
        
        # Fallback to local directory if data_dir doesn't exist
        if not os.path.exists(data_dir):
            data_dir = base_dir
            
        path = os.path.join(data_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web Scraper Tool")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")
    args = parser.parse_args()

    scraper = WebScraper()
    print(f"Scraping {args.pages} pages...")
    data = scraper.scrape_books(num_pages=args.pages)
    path = scraper.save_raw(data)
    print(f"Saved {len(data)} books to {path}")
