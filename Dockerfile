# Используем официальный образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости, включая tesseract-ocr
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# Копируем и устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем код приложения
COPY . .

# Указываем команду для запуска бота
CMD ["python", "bot.py"]
