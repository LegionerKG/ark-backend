FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование проекта
WORKDIR /app
COPY . .

# Установка Python-зависимостей
RUN pip install -r requirements.txt

# Запуск бота
CMD ["python", "bot.py"]
