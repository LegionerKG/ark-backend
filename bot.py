import telebot
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import json
import os
import io
import logging
from telebot import types
import easyocr
from ultralytics import YOLO

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Получение токена из переменной окружения
TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    logger.error("TELEGRAM_TOKEN не задан в переменных окружения!")
    raise ValueError("Необходимо указать TELEGRAM_TOKEN в переменных окружения")

bot = telebot.TeleBot(TOKEN)

# Хранение временных данных пользователей
user_states = {}

# Директория для сохранения JSON-файлов и временных изображений
TEMPLATES_DIR = "user_templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Инициализация модели YOLOv8 и EasyOCR
model = YOLO('yolov8n.pt')  # Предобученная модель, замените на 'best.pt' после дообучения
reader = easyocr.Reader(['en'])  # Инициализация EasyOCR для текста

@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    user_states[user_id] = {}
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(types.KeyboardButton('/learn'), types.KeyboardButton('/create'))
    bot.reply_to(message, "Выберите действие:", reply_markup=markup)

@bot.message_handler(commands=['learn'])
def learn_template(message):
    """Начало процесса «обучения» шаблона (запрос названия)."""
    user_id = message.from_user.id
    user_states[user_id]['action'] = 'learn'
    bot.reply_to(message, "Введите название шаблона:")
    bot.register_next_step_handler(message, get_template_name)

def get_template_name(message):
    """Получаем от пользователя название шаблона и ждём изображение."""
    user_id = message.from_user.id
    template_name = message.text.strip()
    
    if not template_name:
        bot.reply_to(message, "Название не может быть пустым")
        return
    
    user_states[user_id]['template_name'] = template_name
    bot.reply_to(message, "Отправьте пример поста (изображение) для обучения.")
    bot.register_next_step_handler(message, process_template_image)

def process_template_image(message):
    """Обработка загруженного изображения-шаблона, его анализ и сохранение."""
    user_id = message.from_user.id
    
    if not message.photo:
        bot.reply_to(message, "Пожалуйста, отправьте изображение.")
        return
        
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_path = os.path.join(TEMPLATES_DIR, f"{user_id}_temp.jpg")
        
        with open(temp_path, 'wb') as f:
            f.write(downloaded_file)
        
        template = analyze_template(temp_path)
        template['name'] = user_states[user_id]['template_name']
        
        user_dir = os.path.join(TEMPLATES_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        template_path = os.path.join(user_dir, f"{template['name']}.json")
        image_save_path = os.path.join(user_dir, f"{template['name']}.jpg")  # Сохраняем оригинал
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        with open(image_save_path, 'wb') as f:
            f.write(downloaded_file)  # Сохраняем фото
        
        preview = generate_preview(template, temp_path)
        markup = types.InlineKeyboardMarkup()
        markup.add(
            types.InlineKeyboardButton("Подтвердить", callback_data=f"confirm_{template['name']}"),
            types.InlineKeyboardButton("Отменить", callback_data=f"cancel_{template['name']}")
        )
        
        bot.send_photo(user_id, preview, "Подтвердите шаблон:", reply_markup=markup)
        os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Ошибка в process_template_image: {str(e)}")
        bot.reply_to(message, f"Ошибка: {str(e)}")
        temp_path = os.path.join(TEMPLATES_DIR, f"{user_id}_temp.jpg")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_template(image_path):
    """Анализ шаблона с использованием предобученной YOLOv8 и EasyOCR."""
    # Загружаем изображение
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.open(image_path).convert("RGBA")
    
    # Используем EasyOCR для детекции текста
    ocr_results = reader.readtext(img_rgb)
    text_areas = []
    for (bbox, text, prob) in ocr_results:
        if prob < 0.5:
            continue
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x1, y1 = int(top_left[0]), int(top_left[1])
        x2, y2 = int(bottom_right[0]), int(bottom_right[1])
        w, h = x2 - x1, y2 - y1
        
        # Определяем цвет текста
        cx, cy = x1 + w//2, y1 + h//2
        if cx < img_rgb.shape[1] and cy < img_rgb.shape[0]:
            color = [int(img_rgb[cy, cx][0]), int(img_rgb[cy, cx][1]), int(img_rgb[cy, cx][2])]
        else:
            color = [255, 255, 255]
        
        font_size = int(h * 0.8)
        if font_size < 10:
            font_size = 10
        
        # Проверяем яркость фона под текстом (нужна ли заливка)
        region = np.array(img_pil.crop((x1, y1, x2, y2)).convert("L"))
        brightness = np.mean(region)
        needs_overlay = brightness > 128  # Если фон слишком яркий, добавляем затемнение
        
        text_area = {
            'position': (x1, y1, w, h),
            'color': color,
            'font_size': font_size,
            'font_family': 'arial.ttf',
            'needs_overlay': needs_overlay
        }
        text_areas.append(text_area)
    
    # Используем YOLOv8 для поиска логотипа (если модель дообучена)
    logo_position = None
    results = model(img_rgb)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label = result.names[int(box.cls)]
            if label == 'logo':  # Предполагаем, что YOLO дообучена на класс "logo"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                logo_position = (x1, y1, w, h)
                break
    
    # Если логотип не найден, задаём дефолтное положение
    if not logo_position:
        img_h, img_w = image.shape[:2]
        logo_position = (int(img_w * 0.8), int(img_h * 0.8), 100, 100)
    
    # Определяем зоны затемнения на основе яркости фона под текстом
    overlay_areas = []
    for area in text_areas:
        if area['needs_overlay']:
            x, y, w, h = area['position']
            overlay_areas.append((x, y, w, h))
    
    # Определяем цветовую схему
    background_color = [int(c) for c in img_rgb[10, 10]]
    color_scheme = get_color_scheme(image)
    
    return {
        'text_areas': text_areas,
        'logo_position': logo_position,
        'overlay_areas': overlay_areas,
        'background_color': background_color,
        'color_scheme': color_scheme
    }

def get_color_scheme(image):
    """Анализ цветовой схемы через k-means."""
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return [int(c) for c in dominant]

def generate_preview(template, image_path):
    """Генерация предпросмотра шаблона."""
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # Добавляем затемнение
    for overlay_area in template.get('overlay_areas', []):
        x, y, w, h = overlay_area
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0, 128))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
    
    for area in template.get('text_areas', []):
        font_path = area['font_family'] or "arial.ttf"
        font_size = area['font_size']
        text_color = tuple(area['color'])
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        x, y, w, h = area['position']
        sample_text = "Пример текста"
        lines = wrap_text(sample_text, font, w)
        
        total_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
        current_y = y + (h - total_height) // 2
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            line_x = x + (w - line_width) // 2
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            current_y += line_height
    
    if 'logo_position' in template:
        x, y, w, h = template['logo_position']
        draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)
    
    preview = io.BytesIO()
    img.save(preview, format='PNG')
    preview.seek(0)
    return preview

def wrap_text(text, font, max_width):
    """Перенос текста по ширине."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = current_line + [word]
        test_text = ' '.join(test_line)
        bbox = font.getbbox(test_text)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    return lines

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """Обработка подтверждения/отмены шаблона."""
    user_id = call.from_user.id
    action, template_name = call.data.split('_', 1)
    
    if action == 'confirm':
        bot.answer_callback_query(call.id, "Шаблон сохранен!")
        bot.send_message(user_id, f"Шаблон '{template_name}' успешно сохранен!")
    elif action == 'cancel':
        template_path = os.path.join(TEMPLATES_DIR, str(user_id), f"{template_name}.json")
        image_path = os.path.join(TEMPLATES_DIR, str(user_id), f"{template_name}.jpg")
        if os.path.exists(template_path):
            os.remove(template_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        bot.answer_callback_query(call.id, "Шаблон удалён")
        bot.send_message(user_id, f"Шаблон '{template_name}' удален")

@bot.message_handler(commands=['create'])
def create_post(message):
    """Выбор шаблона для создания поста."""
    user_id = message.from_user.id
    user_dir = os.path.join(TEMPLATES_DIR, str(user_id))
    
    if not os.path.exists(user_dir) or not os.listdir(user_dir):
        bot.reply_to(message, "У вас нет сохраненных шаблонов. Создайте их через /learn.")
        return
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    templates = [os.path.splitext(entry.name)[0] for entry in os.scandir(user_dir) if entry.is_file() and entry.name.endswith('.json')]
    
    if not templates:
        bot.reply_to(message, "Нет файлов-шаблонов. Создайте через /learn.")
        return
    
    for t in templates:
        markup.add(types.KeyboardButton(t))
    markup.add(types.KeyboardButton("Отмена"))
    
    user_states[user_id]['action'] = 'select_template'
    bot.reply_to(message, "Выберите шаблон:", reply_markup=markup)
    bot.register_next_step_handler(message, process_template_selection)

def process_template_selection(message):
    """Обработка выбора шаблона и запрос фото."""
    user_id = message.from_user.id
    template_name = message.text
    
    if template_name == "Отмена":
        return start(message)
    
    template_path = os.path.join(TEMPLATES_DIR, str(user_id), f"{template_name}.json")
    
    if not os.path.exists(template_path):
        bot.reply_to(message, "Шаблон не найден.")
        return
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    user_states[user_id]['current_template'] = template
    bot.reply_to(message, "Отправьте изображение (можно с подписью) для финального поста.")
    bot.register_next_step_handler(message, generate_final_post)

def generate_final_post(message):
    """Генерация финального поста с текстом и логотипом."""
    user_id = message.from_user.id
    template = user_states[user_id].get('current_template')
    
    if not template:
        bot.reply_to(message, "Ошибка: шаблон не выбран.")
        return
    
    if not message.photo:
        bot.reply_to(message, "Пожалуйста, отправьте именно фото.")
        return
    
    try:
        text = message.caption or "Текст поста"
        lines = text.split('\n')  # Разделяем текст по строкам
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file)).convert("RGBA")
        
        draw = ImageDraw.Draw(img)
        text_areas = template.get('text_areas', [])
        
        # Добавляем затемнение
        for overlay_area in template.get('overlay_areas', []):
            x, y, w, h = overlay_area
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0, 128))
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)
        
        # Сортируем зоны по Y-координате
        text_areas.sort(key=lambda a: a['position'][1])
        
        # Распределяем текст по зонам
        for i, area in enumerate(text_areas):
            x, y, w, h = area['position']
            text_color = tuple(area['color'])
            font_path = area.get('font_family', "arial.ttf")
            font_size = area.get('font_size', 24)
            
            # Добавляем затемнение, если нужно
            if area.get('needs_overlay', False):
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x, y, x+w, y+h], fill=(0, 0, 0, 128))
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)
            
            # Если это первая зона (например, "BREAKING"), используем первую строку
            # Если вторая зона (основной текст), используем оставшиеся строки
            if i == 0 and len(lines) > 1:
                area_text = lines[0]
            else:
                area_text = ' '.join(lines[i:]) if i < len(lines) else ""
            
            final_font = fit_text_in_area(area_text, font_path, font_size, w, h)
            area_lines = wrap_text_multi(area_text, final_font, w, h)
            
            total_height = sum(final_font.getbbox(line)[3] - final_font.getbbox(line)[1] for line in area_lines)
            current_y = y + (h - total_height) // 2
            
            for line in area_lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                line_x = x + (w - line_width) // 2
                draw.text((line_x, current_y), line, font=final_font, fill=text_color)
                current_y += line_height
        
        if 'logo_position' in template:
            lx, ly, lw, lh = template['logo_position']
            if os.path.exists("default_logo.png"):
                logo = Image.open("default_logo.png").convert("RGBA")
                logo.thumbnail((lw, lh), Image.Resampling.LANCZOS)
                img.paste(logo, (lx, ly), logo)
        
        result = io.BytesIO()
        img.save(result, format='PNG')
        result.seek(0)
        
        bot.send_photo(user_id, result, "Ваш пост готов!")
        
    except Exception as e:
        logger.error(f"Ошибка в generate_final_post: {str(e)}")
        bot.reply_to(message, f"Ошибка: {str(e)}")

def fit_text_in_area(text, font_path, initial_size, max_width, max_height):
    """Подбор размера шрифта для области."""
    size = initial_size
    while size > 10:
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = ImageFont.load_default()
        
        lines = wrap_text_multi(text, font, max_width, max_height)
        total_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)
        
        if total_height <= max_height:
            return font
        
        size -= 2
    
    return ImageFont.load_default()

def wrap_text_multi(text, font, max_width, max_height):
    """Перенос текста с учётом ширины и высоты."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = current_line + [word]
        test_text = ' '.join(test_line)
        bbox = font.getbbox(test_text)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def polling_with_error_handling():
    """Запуск бота с обработкой ошибок."""
    try:
        logger.info("Starting bot with polling...")
        bot.polling(none_stop=True, timeout=15)
    except telebot.apihelper.ApiTelegramException as e:
        if "Timed out" in str(e):
            logger.warning(f"Timed out error: {str(e)}. Retrying...")
            polling_with_error_handling()
        elif "Conflict" in str(e):
            logger.error(f"Conflict error: {str(e)}. Bot instance already running elsewhere. Exiting...")
            raise
        else:
            logger.error(f"Unexpected Telegram API error: {str(e)}. Retrying in 5 seconds...")
            import time
            time.sleep(5)
            polling_with_error_handling()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}. Retrying in 5 seconds...")
        import time
        time.sleep(5)
        polling_with_error_handling()

if __name__ == "__main__":
    polling_with_error_handling()
