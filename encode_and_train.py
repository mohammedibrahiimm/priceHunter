import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

#Load the datasets
dataset_path_1 = "E:/Projects/data-science-final-project-itm-main/clothing_items.csv"
dataset_path_2 = "E:/Projects/data-science-final-project-itm-main/clothing_items_new_500.csv"
dataset_path_3 = "E:/Projects/data-science-final-project-itm-main/clothing_items_new2_1000.csv"

df1 = pd.read_csv(dataset_path_1)
df2 = pd.read_csv(dataset_path_2)
df3 = pd.read_csv(dataset_path_3)

df = pd.concat([df1, df2, df3], ignore_index=True)  # Merge all files into one dataset
# Encode categorical columns
categorical_columns = ["type", "color", "brand", "material", "style", "state"]

# إنشاء LabelEncoders لكل عمود
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # convert values to numerical
    encoders[col] = le

# save the encoders
encoders_path = "E:/Projects/data-science-final-project-itm-main/encoders.pkl"
joblib.dump(encoders, encoders_path)
print(f"✅ تم حفظ LabelEncoders في: {encoders_path}")

# training model
X = df[categorical_columns]
y = df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# save the model
model_path = "E:/Projects/data-science-final-project-itm-main/model.pkl"
joblib.dump(model, model_path)
print(f"Model saved in:  {model_path}")
