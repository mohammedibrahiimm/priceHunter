import sqlite3

DB_PATH = "E:/Projects/data-science-final-project-itm-main/clothing_db.sqlite"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM clothing_items")  # يحسب عدد الصفوف في الجدول
count = cursor.fetchone()[0]

print(f"✅ عدد الصفوف في قاعدة البيانات: {count}")

conn.close()
