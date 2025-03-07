# Базовый образ с Python 3.11
FROM python:3.11-slim

# Установка системных зависимостей для Tesseract и OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование проекта
COPY . .

# Установка Python-зависимостей из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Запуск бота
CMD ["python", "bot.py"]
