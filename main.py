from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sqlite3
import os
import requests
from pydantic import BaseModel
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DB_PATH = os.path.join(BASE_DIR, "clothing_db.sqlite")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5122"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

class Item(BaseModel):
    type: str
    color: str
    brand: str
    material: str
    style: str
    state: str

def get_price_from_db(item: Item):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = '''SELECT price FROM clothing_items 
               WHERE type=? AND color=? AND brand=? AND material=? AND style=? AND state=?'''
    cursor.execute(query, (item.type.lower(), item.color.lower(), item.brand.lower(), 
                           item.material.lower(), item.style.lower(), item.state.lower()))
    result = cursor.fetchall()
    conn.close()
    if result:
        prices = [row['price'] for row in result]
        return sum(prices) / len(prices)
    return None

def get_lowest_price_link(query: str):
    url = "https://app.zenserp.com/api/v2/search"
    params = {
        "q": query,
        "location": "United States",
        "search_engine": "google.com",
        "tbm": "shop",
        "num": 5,
        "apikey": "2376c9b0-1dd4-11f0-a27c-4da665a28829"
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json()
            if "shopping_results" in results:
                sorted_results = sorted(results["shopping_results"], key=lambda x: float(x.get("price", "0").replace("$", "")))
                return sorted_results[0]["link"] if sorted_results else None
    except:
        return None
    return None

BRAND_STORES = {
    "nike": "https://www.nike.com/w?q={query}",
    "adidas": "https://www.adidas.com/us/search?q={query}",
    "zara": "https://www.zara.com/us/en/search?searchTerm={query}",
    "hm": "https://www2.hm.com/en_us/search-results.html?q={query}",
    "uniqlo": "https://www.uniqlo.com/us/en/search?q={query}",
    "shein": "https://www.shein.com/search?q={query}",
    "puma": "https://us.puma.com/us/en/search?q={query}",
    "under armour": "https://www.underarmour.com/en-us/search?q={query}"
}

@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)
    search_query = f"{item.type} {item.color} {item.brand} {item.style} {item.material}".strip()
    query_encoded = search_query.replace(" ", "+")
    state = item.state.lower()
    brand_key = item.brand.lower()

    amazon_link = f"https://www.amazon.com/s?k={query_encoded}"
    shein_link = f"https://www.shein.com/search?q={query_encoded}"
    ebay_link = f"https://www.ebay.com/sch/i.html?_nkw={query_encoded}"

    product_urls = {
        "lowest_price_link": get_lowest_price_link(search_query),
        "amazon_search_link": amazon_link,
    }

    if state == "new":
        product_urls["shein_search_link"] = f"https://www.google.com/search?q=site:shein.com+{query_encoded}"
    elif state == "used":
        product_urls["ebay_search_link"] = ebay_link

    if brand_key in BRAND_STORES:
        product_urls["brand_store_link"] = BRAND_STORES[brand_key].format(query=query_encoded)
    else:
        product_urls["brand_store_link"] = f"https://www.google.com/search?q=site:{brand_key}.com+{query_encoded}"

    if dataset_price is not None:
        return {
            "predicted_price": dataset_price,
            "source": "database",
            "product_urls": product_urls
        }

    try:
        input_data = [
            encoders['type'].transform([item.type.lower()])[0],
            encoders['color'].transform([item.color.lower()])[0],
            encoders['brand'].transform([item.brand.lower()])[0],
            encoders['material'].transform([item.material.lower()])[0],
            encoders['style'].transform([item.style.lower()])[0],
            encoders['state'].transform([item.state.lower()])[0]
        ]
    except Exception as e:
        return {"error": f"Invalid value: {str(e)}"}

    predicted_price = model.predict([input_data])[0]

    return {
        "predicted_price": predicted_price,
        "source": "model",
        "product_urls": product_urls
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
