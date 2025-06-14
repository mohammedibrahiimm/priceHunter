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
    allow_origins=["*"],
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
        print("Zenserp response:", response.text)
        if response.status_code == 200:
            results = response.json()
            if "shopping_results" in results:
                valid_results = []
                for item in results["shopping_results"]:
                    price = None
                    if "price_parsed" in item and "value" in item["price_parsed"]:
                        try:
                            price = float(item["price_parsed"]["value"])
                        except:
                            pass
                    elif "price" in item:
                        try:
                            price_str = item["price"].replace("$", "").split("$")[0]
                            price = float(price_str)
                        except:
                            pass
                    if price is not None:
                        valid_results.append((price, item["link"]))
                if valid_results:
                    valid_results.sort(key=lambda x: x[0])
                    return valid_results[0][1]
    except Exception as e:
        print("Error:", e)
        return None
    return None

official_store_search_links = {
    "adidas": "https://www.adidas.com/us/search?q=",
    "nike": "https://www.nike.com/w?q=",
    "h&m": "https://www2.hm.com/en_us/search-results.html?q=",
    "zara": "https://www.zara.com/us/en/search?searchTerm=",
    "shein": "https://www.shein.com/search?q=",
    "gucci": "https://www.gucci.com/us/en/search?searchTerm=",
    "dior": "https://www.dior.com/en_us/fashion/search/",
    "forever21": "https://www.forever21.com/us/shop/search.html?q=",
    "pull&bear": "https://www.pullandbear.com/us/search?term=",
    "armani": "https://www.armani.com/en-us/search?q=",
    "lacoste": "https://www.lacoste.com/us/search/?q=",
    "gap": "https://www.gap.com/browse/search.do?searchText=",
    "levis": "https://www.levi.com/US/en_US/search?q="
}

@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)

    search_query = f"{item.type} {item.color} {item.brand} {item.style} {item.material}".strip()
    query_encoded = search_query.replace(" ", "+")
    state = item.state.lower()
    brand_lower = item.brand.lower()

    amazon_link = f"https://www.amazon.com/s?k={query_encoded}"
    shein_link = f"https://www.shein.com/search?q={query_encoded}"
    ebay_link = f"https://www.ebay.com/sch/i.html?_nkw={query_encoded}"
    official_store_link = None

    if brand_lower in official_store_search_links:
        official_store_link = official_store_search_links[brand_lower] + query_encoded

    product_urls = {
        "lowest_price_link": get_lowest_price_link(search_query),
    }

    if state == "new":
        product_urls["amazon"] = amazon_link
        if official_store_link:
            product_urls["official_store"] = official_store_link
    elif state == "used":
        product_urls["ebay"] = ebay_link
        product_urls["shein"] = shein_link

    if dataset_price is not None:
        return {
            "predicted_price": round(dataset_price, 2),
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
        "predicted_price": round(predicted_price, 2),
        "source": "model",
        "product_urls": product_urls
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
