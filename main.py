from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sqlite3
import os
import requests
from pydantic import BaseModel
import uvicorn

# ⛓️ إعدادات المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DB_PATH = os.path.join(BASE_DIR, "clothing_db.sqlite")

# ✅ تحميل الموديل والمشفرات
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# ✅ إعداد FastAPI و CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ قاعدة البيانات
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ✅ نموذج البيانات
class Item(BaseModel):
    type: str
    color: str
    brand: str
    material: str
    style: str
    state: str

# ✅ البحث في قاعدة البيانات
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

# ✅ دالة للبحث في مواقع متعددة وجلب النتائج
def search_multiple_products(query: str, sites: list):
    all_results = []

    for site in sites:
        try:
            print(f"🔍 Searching for: {query} on {site}")
            url = "https://app.zenserp.com/api/v2/search"
            params = {
                "q": query,
                "location": "United States",
                "search_engine": "google.com",
                "tbm": "shop",
                "num": 5,
                "domain": site,
                "apikey": "3c0ce450-1c63-11f0-b37b-9f198730fcec"
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            results = response.json()
            if "shopping_results" in results:
                filtered = [
                    r for r in results["shopping_results"]
                    if "goinggoinggone" not in r.get("link", "").lower()
                ]
                all_results.extend(filtered)
        except Exception as e:
            print(f"❌ Error while searching {site}: {e}")

    try:
        sorted_results = sorted(
            all_results,
            key=lambda x: float(x.get("price", "0").replace("$", "").replace(",", ""))
        )
        return sorted_results
    except Exception as e:
        print(f"❌ Error while sorting results: {e}")
        return []

# ✅ Endpoint رئيسي
@app.post("/predict_price/")
async def predict_price(item: Item):
    try:
        dataset_price = get_price_from_db(item)
        search_query = f"{item.brand} {item.color} {item.material} {item.style} {item.type}"
        state = item.state.lower()

        cheapest_link = None
        extra_links = {}

        if state == "new":
            results = search_multiple_products(search_query, ["amazon.com", "shein.com"])
            if results:
                cheapest_link = results[0].get("link")
                for r in results:
                    link = r.get("link", "")
                    if "amazon" in link and "amazon_url" not in extra_links:
                        extra_links["amazon_url"] = link
                    elif "shein" in link and "shein_url" not in extra_links:
                        extra_links["shein_url"] = link

        elif state == "used":
            results = search_multiple_products(search_query, ["ebay.com"])
            if results:
                cheapest_link = results[0].get("link")
                for r in results:
                    link = r.get("link", "")
                    if "ebay" in link and "ebay_url" not in extra_links:
                        extra_links["ebay_url"] = link

        if dataset_price is not None:
            return {
                "predicted_price": dataset_price,
                "source": "database",
                "product_url": cheapest_link,
                **extra_links
            }

        # Model prediction
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
            print(f"❌ Encoding Error: {e}")
            return {"error": f"Invalid value: {str(e)}"}

        predicted_price = model.predict([input_data])[0]

        return {
            "predicted_price": predicted_price,
            "source": "model",
            "product_url": cheapest_link,
            **extra_links
        }

    except Exception as e:
        print(f"❌ General Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}

# ✅ لتشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
