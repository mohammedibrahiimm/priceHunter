import sqlite3

DB_PATH = "E:/Projects/data-science-final-project-itm-main/clothing_db.sqlite"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# جلب أول 10 صفوف من الجدول
cursor.execute("SELECT * FROM clothing_items LIMIT 10")
rows = cursor.fetchall()

if rows:
    print("📊 أول 10 صفوف من قاعدة البيانات:")
    for row in rows:
        print(row)
else:
    print("⚠️ لا توجد بيانات في جدول clothing_items!")

conn.close()
