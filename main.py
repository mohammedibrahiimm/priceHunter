from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sqlite3
import os
import requests
from pydantic import BaseModel
import uvicorn

# إعدادات المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DB_PATH = os.path.join(BASE_DIR, "clothing_db.sqlite")

# تحميل الموديل والمشفرات
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# إعداد FastAPI و CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# قاعدة البيانات
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# نموذج البيانات
class Item(BaseModel):
    type: str
    color: str
    brand: str
    material: str
    style: str
    state: str

# البحث في قاعدة البيانات
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

# Zenserp: جلب أقل سعر
def get_lowest_price_link(query: str):
    url = "https://app.zenserp.com/api/v2/search"
    params = {
        "q": query,
        "location": "United States",
        "search_engine": "google.com",
        "tbm": "shop",
        "num": 5,
        "apikey": "3c0ce450-1c63-11f0-b37b-9f198730fcec"
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

# Endpoint رئيسي
@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)

    # 🔍 تجهيز بيانات البحث
    search_query = f"{item.type} {item.color} {item.brand} {item.style} {item.material}".strip()
    query_encoded = search_query.replace(" ", "+")
    state = item.state.lower()

    # 🔗 تكوين روابط البحث اليدوية
    amazon_link = f"https://www.amazon.com/s?k={query_encoded}"
    shein_link = f"https://www.shein.com/search?q={query_encoded}"
    ebay_link = f"https://www.ebay.com/sch/i.html?_nkw={query_encoded}"

    # 🧠 روابط المنتجات
    product_urls = {
        "lowest_price_link": get_lowest_price_link(search_query),
        "amazon_search_link": amazon_link,
    }

    if state == "new":
        product_urls["shein_search_link"] = shein_link
    elif state == "used":
        product_urls["ebay_search_link"] = ebay_link

    if dataset_price is not None:
        return {
            "predicted_price": dataset_price,
            "source": "database",
            "product_urls": product_urls
        }

    # تنبؤ من الموديل
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

# تشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
