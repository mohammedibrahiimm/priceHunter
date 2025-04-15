from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sqlite3
import os
from pydantic import BaseModel
import uvicorn

# Getting file paths in a way compatible with Railway
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DB_PATH = os.path.join(BASE_DIR, "clothing_db.sqlite")

# Loading the model and encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Creating the API
app = FastAPI()

# CORS settings to work with React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your link here if working in a different environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connecting to the database
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# Data model
class Item(BaseModel):
    type: str
    color: str
    brand: str
    material: str
    style: str
    state: str

# Function to search for price in the database
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

# Endpoint for price prediction with Amazon search link
@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)

    # ⛓️ Creating Amazon search link
    search_query = f"{item.brand} {item.color} {item.material} {item.style} {item.type}"
    amazon_search_url = f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"

    if dataset_price is not None:
        return {
            "predicted_price": dataset_price,
            "source": "database",
            "product_search_url": amazon_search_url
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
        "product_search_url": amazon_search_url
    }

# Running the application on Railway
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))