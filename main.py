from fastapi import FastAPI
from routes import router
import joblib
import sqlite3
import os
from pydantic import BaseModel
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
DB_PATH = os.path.join(BASE_DIR, "clothing_db.sqlite")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

app = FastAPI()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)  
    conn.row_factory = sqlite3.Row
    return conn  
app.include_router(router) #routes.py
 
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

@app.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)
    
    if dataset_price is not None:
        return {"predicted_price": dataset_price, "source": "database"}
    
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
    
    return {"predicted_price": predicted_price, "source": "model"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
