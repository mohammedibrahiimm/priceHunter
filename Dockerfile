# استخدام صورة رسمية لـ Python
FROM python:3.10

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات المشروع إلى الحاوية
COPY . .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# تشغيل التطبيق
CMD ["python", "main.py"]
