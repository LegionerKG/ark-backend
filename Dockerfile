# Используем официальный образ Python как основу
FROM python:3.11-slim

# Устанавливаем системные зависимости, включая tesseract-ocr и зависимости OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код проекта
COPY . .

# Указываем команду для запуска бота
CMD ["python", "bot.py"]
