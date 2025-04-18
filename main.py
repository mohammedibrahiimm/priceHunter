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

# ✅ دالة جلب رابط منتج من Zenserp
def search_product_zenserp(query: str, site: str = None):
    try:
        url = "https://app.zenserp.com/api/v2/search"
        params = {
            "q": query,
            "location": "United States",
            "search_engine": "google.com",
            "tbm": "shop",
            "num": 5,
            "apikey": "3c0ce450-1c63-11f0-b37b-9f198730fcec"
        }
        if site:
            params["domain"] = site

        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        if "shopping_results" in results:
            filtered_results = [
                r for r in results["shopping_results"]
                if "goinggoinggone" not in r.get("link", "").lower()
            ]
            sorted_results = sorted(filtered_results, key=lambda x: float(x.get("price", "0").replace("$", "")))
            return sorted_results[0] if sorted_results else None
    except Exception as e:
        return None
    return None

# ✅ Endpoint رئيسي
@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)
    search_query = f"{item.brand} {item.color} {item.material} {item.style} {item.type}"
    state = item.state.lower()

    cheapest_product = None
    amazon_product = None
    shein_product = None
    ebay_product = None

    if state == "new":
        amazon_product = search_product_zenserp(search_query, site="amazon.com")
        shein_product = search_product_zenserp(search_query, site="shein.com")
        candidates = [p for p in [amazon_product, shein_product] if p is not None]
    elif state == "used":
        ebay_product = search_product_zenserp(search_query, site="ebay.com")
        candidates = [ebay_product] if ebay_product else []

    # اختيار أرخص منتج
    if candidates:
        cheapest_product = min(
            candidates,
            key=lambda x: float(x.get("price", "0").replace("$", ""))
        )

    result = {
        "predicted_price": dataset_price if dataset_price else None,
        "source": "database" if dataset_price else "model",
        "product_url": cheapest_product
    }

    # إضافة روابط إضافية حسب الحالة
    if state == "new":
        if amazon_product:
            result["amazon_url"] = amazon_product.get("link")
        if shein_product:
            result["shein_url"] = shein_product.get("link")
    elif state == "used":
        if ebay_product:
            result["ebay_url"] = ebay_product.get("link")

    # تنبؤ بالسعر في حالة عدم وجوده في قاعدة البيانات
    if not dataset_price:
        try:
            input_data = [
                encoders['type'].transform([item.type.lower()])[0],
                encoders['color'].transform([item.color.lower()])[0],
                encoders['brand'].transform([item.brand.lower()])[0],
                encoders['material'].transform([item.material.lower()])[0],
                encoders['style'].transform([item.style.lower()])[0],
                encoders['state'].transform([item.state.lower()])[0]
            ]
            predicted_price = model.predict([input_data])[0]
            result["predicted_price"] = predicted_price
        except Exception as e:
            result["error"] = f"Invalid value: {str(e)}"

    return result

# ✅ لتشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
