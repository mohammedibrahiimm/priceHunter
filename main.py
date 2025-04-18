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
def search_product_zenserp(query: str, site: str = "amazon.com"):
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
    if response.status_code == 200:
        results = response.json()
        if "shopping_results" in results:
            sorted_results = sorted(results["shopping_results"], key=lambda x: float(x.get("price", "0").replace("$", "")))
            return sorted_results[0] if sorted_results else None
    return None

# ✅ دالة لتحليل السعر في الرابط
def extract_price_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        # هنا يمكننا استخراج السعر من الرابط عن طريق تحليل الصفحة (مثال: باستخدام BeautifulSoup)
        # ده مجرد placeholder: هنفترض إننا استخرجنا السعر بنجاح من الصفحة
        price = 0.0  # سعر افتراضي
        # تحليل الصفحة لاستخراج السعر الحقيقي سيحتاج مكتبة مثل BeautifulSoup
        return price
    return None

# ✅ Endpoint رئيسي
@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)

    search_query = f"{item.brand} {item.color} {item.material} {item.style} {item.type}"
    state = item.state.lower()
    product_url = None

    if state == "new":
        # بحث في أمازون وشين
        amazon_product = search_product_zenserp(search_query, site="amazon.com")
        shein_product = search_product_zenserp(search_query, site="shein.com")
        product_url = amazon_product or shein_product
    elif state == "used":
        # بحث في eBay
        product_url = search_product_zenserp(search_query, site="ebay.com")

    # إذا كانت البيانات موجودة في قاعدة البيانات
    if dataset_price is not None:
        return {
            "predicted_price": dataset_price,
            "source": "database",
            "product_url": product_url
        }

    # التنبؤ بالسعر من الموديل
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

    # إذا وجدنا رابط من Zenserp، قارن السعر المتوقع مع السعر الذي وجدناه
    if product_url:
        price_from_url = extract_price_from_url(product_url)
        if price_from_url:
            price_diff = abs(predicted_price - price_from_url)
            if price_diff < 10:  # فرق السعر مسموح به (مثلاً أقل من 10 دولارات)
                return {
                    "predicted_price": predicted_price,
                    "source": "model",
                    "product_url": product_url,
                    "price_diff": price_diff
                }
    
    return {
        "predicted_price": predicted_price,
        "source": "model",
        "product_url": product_url
    }

# ✅ لتشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
