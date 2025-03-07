import telebot
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
from telebot import types
import io

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TOKEN)

# Хранение временных данных пользователей
user_states = {}
user_templates = {}

# Структура данных для шаблонов
TEMPLATES_DIR = "user_templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    user_states[user_id] = {}
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton('/learn'), types.KeyboardButton('/create'))
    bot.reply_to(message, "Выберите действие:", reply_markup=markup)

@bot.message_handler(commands=['learn'])
def learn_template(message):
    user_id = message.from_user.id
    user_states[user_id]['action'] = 'learn'
    bot.reply_to(message, "Введите название шаблона:")
    bot.register_next_step_handler(message, get_template_name)

def get_template_name(message):
    user_id = message.from_user.id
    template_name = message.text.strip()
    
    if not template_name:
        bot.reply_to(message, "Название не может быть пустым")
        return
    
    user_states[user_id]['template_name'] = template_name
    bot.reply_to(message, "Отправьте пример поста для обучения")
    bot.register_next_step_handler(message, process_template_image)

def process_template_image(message):
    user_id = message.from_user.id
    
    if not message.photo:
        bot.reply_to(message, "Пожалуйста, отправьте изображение")
        return
        
    try:
        # Сохраняем изображение
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_path = f"{TEMPLATES_DIR}/{user_id}_temp.jpg"
        
        with open(temp_path, 'wb') as f:
            f.write(downloaded_file)
        
        # Анализируем шаблон
        template = analyze_template(temp_path)
        template['name'] = user_states[user_id]['template_name']
        
        # Сохраняем шаблон
        user_dir = f"{TEMPLATES_DIR}/{user_id}"
        os.makedirs(user_dir, exist_ok=True)
        template_path = f"{user_dir}/{template['name']}.json"
        
        with open(template_path, 'w') as f:
            json.dump(template, f)
            
        # Генерируем подтверждение
        preview = generate_preview(template, temp_path)
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("Подтвердить", callback_data=f"confirm_{template['name']}"),
            types.InlineKeyboardButton("Отменить", callback_data=f"cancel_{template['name']}")
        )
        
        bot.send_photo(user_id, preview, "Подтвердите шаблон:", reply_markup=markup)
        os.remove(temp_path)
        
    except Exception as e:
        bot.reply_to(message, f"Ошибка: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_template(image_path):
    """Улучшенный анализ шаблона"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Анализ текстовых блоков
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 5000:  # Минимальный размер области
            # Анализ цвета
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(image, mask=mask)[:3]
            
            # Анализ шрифта (эмпирически)
            font_size = int(h * 0.8)
            
            text_areas.append({
                'position': (x, y, w, h),
                'color': [int(c) for c in mean_color],
                'font_size': font_size,
                'font_family': 'arial.ttf'  # Здесь можно добавить анализ шрифта
            })
    
    # Анализ логотипа
    logo_coords = find_logo(gray)
    
    return {
        'text_areas': text_areas,
        'logo_position': logo_coords,
        'background_color': [int(c) for c in image[10, 10]],
        'color_scheme': get_color_scheme(image)
    }

def find_logo(gray_image):
    """Улучшенный поиск логотипа"""
    # Используем детектор границ
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ищем самый большой контур
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        return (x, y, w, h)
    
    return (50, 50, 200, 100)  # Значение по умолчанию

def get_color_scheme(image):
    """Анализ цветовой схемы"""
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    
    dominant = palette[np.argmax(counts)]
    return [int(c) for c in dominant]

def generate_preview(template, image_path):
    """Генерация предпросмотра шаблона"""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    for area in template['text_areas']:
        font = ImageFont.truetype(area['font_family'], area['font_size'])
        text = "Пример текста"
        x, y, w, h = area['position']
        text_color = tuple(area['color'])
        
        # Автоматическая подгонка текста
        lines = []
        words = text.split()
        while words:
            line = []
            while words and font.getsize(' '.join(line + words[:1]))[0] <= w:
                line.append(words.pop(0))
            lines.append(' '.join(line))
        
        y_text = y
        for line in lines:
            draw.text((x, y_text), line, font=font, fill=text_color)
            y_text += font.getsize(line)[1]
    
    # Добавляем рамку для логотипа
    if template.get('logo_position'):
        x, y, w, h = template['logo_position']
        draw.rectangle([x, y, x+w, y+h], outline=(255,0,0), width=3)
    
    preview = io.BytesIO()
    img.save(preview, format='PNG')
    preview.seek(0)
    return preview

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    user_id = call.from_user.id
    action, template_name = call.data.split('_', 1)
    
    if action == 'confirm':
        bot.answer_callback_query(call.id, "Шаблон сохранен!")
        bot.send_message(user_id, f"Шаблон '{template_name}' успешно сохранен!")
    elif action == 'cancel':
        # Удаляем шаблон
        template_path = f"{TEMPLATES_DIR}/{user_id}/{template_name}.json"
        if os.path.exists(template_path):
            os.remove(template_path)
        bot.answer_callback_query(call.id, "Шаблон удален")
        bot.send_message(user_id, f"Шаблон '{template_name}' удален")

@bot.message_handler(commands=['create'])
def create_post(message):
    user_id = message.from_user.id
    user_dir = f"{TEMPLATES_DIR}/{user_id}"
    
    if not os.path.exists(user_dir) or not os.listdir(user_dir):
        bot.reply_to(message, "У вас нет сохраненных шаблонов. Создайте их через /learn")
        return
    
    # Создаем клавиатуру с шаблонами
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    templates = [f.stem for f in os.scandir(user_dir)]
    markup.add(*[types.KeyboardButton(t) for t in templates])
    markup.add(types.KeyboardButton("Отмена"))
    
    user_states[user_id]['action'] = 'select_template'
    bot.reply_to(message, "Выберите шаблон:", reply_markup=markup)
    bot.register_next_step_handler(message, process_template_selection)

def process_template_selection(message):
    user_id = message.from_user.id
    template_name = message.text
    
    if template_name == "Отмена":
        return start(message)
    
    template_path = f"{TEMPLATES_DIR}/{user_id}/{template_name}.json"
    
    if not os.path.exists(template_path):
        bot.reply_to(message, "Шаблон не найден")
        return
    
    with open(template_path, 'r') as f:
        template = json.load(f)
    
    user_states[user_id]['current_template'] = template
    bot.reply_to(message, "Отправьте текст и изображение для поста (можно архивом)")
    bot.register_next_step_handler(message, generate_final_post)

def generate_final_post(message):
    user_id = message.from_user.id
    template = user_states[user_id].get('current_template')
    
    if not template:
        bot.reply_to(message, "Ошибка: шаблон не выбран")
        return
    
    try:
        # Получаем текст и изображение
        text = message.caption or "Текст поста"
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file))
        
        # Создаем пост по шаблону
        draw = ImageDraw.Draw(img)
        
        for area in template['text_areas']:
            font = ImageFont.truetype(area['font_family'], area['font_size'])
            x, y, w, h = area['position']
            text_color = tuple(area['color'])
            
            # Автоматическая подгонка текста
            lines = []
            words = text.split()
            while words:
                line = []
                while words and font.getsize(' '.join(line + words[:1]))[0] <= w:
                    line.append(words.pop(0))
                lines.append(' '.join(line))
            
            y_text = y
            for line in lines:
                draw.text((x, y_text), line, font=font, fill=text_color)
                y_text += font.getsize(line)[1]
        
        # Добавляем логотип
        if 'logo_position' in template:
            logo = Image.open("default_logo.png")  # Путь к вашему логотипу
            img.paste(logo, template['logo_position'][:2], logo)
        
        # Сохраняем результат
        result = io.BytesIO()
        img.save(result, format='PNG')
        result.seek(0)
        
        bot.send_photo(user_id, result, "Ваш пост готов!")
        
    except Exception as e:
        bot.reply_to(message, f"Ошибка генерации: {str(e)}")

if __name__ == "__main__":
    bot.polling()
