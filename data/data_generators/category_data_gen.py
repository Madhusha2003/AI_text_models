import random
import pandas as pd

# -----------------------------
# Expanded product dictionary
# -----------------------------
data_map = {
    "Electronics": [
        "iPhone", "Samsung Galaxy", "MacBook", "Dell Laptop", "HP Pavilion",
        "Sony Headphones", "Bluetooth Speaker", "Smartwatch", "iPad",
        "Gaming Console", "Wireless Earbuds", "Power Bank", "Router"
    ],
    "Apparel And Fashion": [
        "Nike T-Shirt", "Adidas Hoodie", "Levi's Jeans", "Polo Shirt",
        "Leather Jacket", "Summer Dress", "Sneakers", "Cap", "Shorts"
    ],
    "Kitchen Appliances": [
        "Blender", "Coffee Maker", "Air Fryer", "Microwave Oven",
        "Rice Cooker", "Toaster", "Juicer", "Stand Mixer"
    ],
    "Health And Beauty": [
        "Shampoo", "Conditioner", "Body Lotion", "Face Wash",
        "Toothpaste", "Perfume", "Deodorant", "Hair Oil", "Moisturizer"
    ],
    "Food And Grocery": [
        "Milk", "Bread", "Butter", "Cheese", "Rice",
        "Sugar", "Tea Pack", "Coffee Powder", "Chips", "Chocolate"
    ],
    "Household Items": [
        "Dish Soap", "Laundry Detergent", "Floor Cleaner",
        "Mop", "Broom", "Trash Bags", "Air Freshener"
    ],
    "Personal Care": [
        "Soap Bar", "Toothbrush", "Razor", "Shaving Cream",
        "Hand Sanitizer", "Talcum Powder"
    ],
    "Tools And Hardware": [
        "Hammer", "Screwdriver Set", "Electric Drill",
        "Wrench", "Nails Pack", "Pliers", "Measuring Tape"
    ],
    "Automotive": [
        "Engine Oil", "Brake Fluid", "Car Battery",
        "Windshield Wipers", "Car Wax", "Fuel Filter"
    ],
    "Education And Stationery": [
        "Notebook", "Pen Set", "Pencil Box",
        "Eraser", "Highlighter", "Backpack", "Stapler"
    ],
    "Baby Products": [
        "Baby Diapers", "Baby Lotion", "Baby Shampoo",
        "Feeding Bottle", "Baby Powder", "Baby Wipes"
    ]
}

categories = list(data_map.keys())

# -----------------------------
# Settings
# -----------------------------
target_size = 1200
rows = []

target_per_category = target_size // len(categories)

# -----------------------------
# Generate dataset
# -----------------------------
for category in categories:
    used = set()

    while len(used) < target_per_category:
        product = random.choice(data_map[category])

        # realistic variations
        variant = random.choice([
            "", "Pro", "Plus", "Max", "Lite", "2025 Edition"
        ])

        size = random.choice([
            "", "128GB", "256GB", "1L", "500ml", "XL", "Size M"
        ])

        product_name = f"{product} {variant} {size}".strip()

        if product_name in used:
            continue

        used.add(product_name)
        rows.append([product_name, category])

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame(rows, columns=["Product Name", "Category"])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save file
df.to_csv("data\\receipt_ai\\synthetic_products_full.csv", index=False)

# Preview
print(df.head())
print("Total rows:", len(df))
print(df['Category'].value_counts())