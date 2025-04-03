from fastapi import APIRouter
from models import Item
from database import get_price_from_db, predict_price_model

router = APIRouter()

@router.post("/predict_price/")
async def predict_price(item: Item):
    dataset_price = get_price_from_db(item)
    predicted_price = predict_price_model(item, dataset_price)
    return {"predicted_price": predicted_price}
