import telebot
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import os
import io
import textwrap
import pytesseract
import shutil

from telebot import types

TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TOKEN)

# Хранение временных данных пользователей
user_states = {}
# Дополнительный словарь не всегда нужен,
# поскольку шаблоны сохраняются в файл.
# user_templates = {}

# Директория для сохранения JSON-файлов и временных изображений
TEMPLATES_DIR = "user_templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Проверяем, установлен ли Tesseract в системе (для OCR)
TESSERACT_AVAILABLE = bool(shutil.which("tesseract"))

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
        # Сохраняем изображение во временный файл
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        temp_path = os.path.join(TEMPLATES_DIR, f"{user_id}_temp.jpg")
        
        with open(temp_path, 'wb') as f:
            f.write(downloaded_file)
        
        # Анализируем шаблон
        template = analyze_template(temp_path)
        template['name'] = user_states[user_id]['template_name']
        
        # Сохраняем шаблон в JSON
        user_dir = os.path.join(TEMPLATES_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        template_path = os.path.join(user_dir, f"{template['name']}.json")
        
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
            
        # Генерируем предпросмотр
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
        # при ошибке удаляем временный файл, чтобы не копить мусор
        temp_path = os.path.join(TEMPLATES_DIR, f"{user_id}_temp.jpg")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def analyze_template(image_path):
    """
    Улучшенный анализ шаблона:
    1) Если tesseract установлен, пытается найти текстовые зоны через OCR.
    2) В противном случае – fallback: threshold + contours.
    3) Анализируем логотип.
    4) Анализируем цветовую схему (dominant color).
    """
    # Загружаем в OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if TESSERACT_AVAILABLE:
        text_areas = find_text_zones_ocr(image)
        if not text_areas:
            # OCR не нашёл или не смог. fallback
            text_areas = find_text_zones_contours(gray, image)
    else:
        text_areas = find_text_zones_contours(gray, image)
    
    # Логотип
    logo_coords = find_logo(gray, image.shape)
    
    # Цветовая схема
    background_color = [int(c) for c in image[10, 10]]
    color_scheme = get_color_scheme(image)
    
    return {
        'text_areas': text_areas,
        'logo_position': logo_coords,
        'background_color': background_color,
        'color_scheme': color_scheme
    }

def find_text_zones_ocr(bgr_image):
    """
    Используем Tesseract, чтобы найти фрагменты текста.
    Возвращаем список словарей вида:
    [
      {
        'position': (x, y, w, h),
        'color': [r, g, b],
        'font_size': int,
        'font_family': 'arial.ttf'
      }, ...
    ]
    Примечание: Tesseract может находить много маленьких боксов.
    Чтобы их объединять в более крупные блоки, можно дописать логику группировки.
    Для упрощения здесь возвращаем bounding box для каждого «непустого» текста.
    """
    import pytesseract
    from pytesseract import Output
    
    # OCR работает лучше в RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    data = pytesseract.image_to_data(rgb_image, output_type=Output.DICT)
    n_boxes = len(data['level'])
    result = []
    
    # Порог площади, чтобы отсекать микроскопические фрагменты
    MIN_AREA = 2000
    
    for i in range(n_boxes):
        text_str = data['text'][i].strip()
        if not text_str:
            continue
        
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        area = w * h
        if area < MIN_AREA:
            continue
        
        # Цвет примерно берём в центре
        cx, cy = x + w//2, y + h//2
        if cx < bgr_image.shape[1] and cy < bgr_image.shape[0]:
            color_bgr = bgr_image[cy, cx]
            color = [int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])]
        else:
            color = [255, 255, 255]
        
        # Эвристически считаем, что высота определяет шрифт
        font_size = int(h * 0.8)
        if font_size < 10:
            font_size = 10
        
        result.append({
            'position': (x, y, w, h),
            'color': color,
            'font_size': font_size,
            'font_family': 'arial.ttf'
        })
    
    return result

def find_text_zones_contours(gray_image, bgr_image):
    """
    Fallback для случая, если OCR недоступен или дал нулевой результат.
    Ищем контуры на бинаризованном изображении. Для каждого крупного контура
    считаем это потенциальной текстовой зоной.
    """
    _, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 5000:  # Минимальный размер области
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(bgr_image, mask=mask)[:3]
            color = [int(mean_color[2]), int(mean_color[1]), int(mean_color[0])]
            font_size = int(h * 0.8)
            if font_size < 10:
                font_size = 10
            
            text_areas.append({
                'position': (x, y, w, h),
                'color': color,
                'font_size': font_size,
                'font_family': 'arial.ttf'
            })
    return text_areas

def find_logo(gray_image, img_shape):
    """
    Улучшенный поиск логотипа:
    1) Canny для ребёр
    2) Находим самый большой контур, НЕ превышающий 40% площади всего изображения
    3) Если нет подходящего, возвращаем дефолт (50,50,200,100)
    """
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    best_cnt = None
    img_h, img_w = img_shape[:2]
    total_area = img_w * img_h
    AREA_THRESHOLD = 0.4 * total_area  # 40% от всей картинки
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Игнорируем совсем гигантские области (фон)
        if area > max_area and area < AREA_THRESHOLD:
            max_area = area
            best_cnt = cnt
    
    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        return (x, y, w, h)
    else:
        return (50, 50, 200, 100)  # Значение по умолчанию

def get_color_scheme(image):
    """
    Анализ цветовой схемы (k=5).
    Возвращаем доминирующий цвет как [R,G,B].
    Можно дополнительно вернуть весь список кластеров/палитру.
    """
    pixels = np.float32(image.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    
    dominant = palette[np.argmax(counts)]
    return [int(c) for c in dominant]

def generate_preview(template, image_path):
    """
    Генерация предпросмотра шаблона:
    - рисуем поверх текста "Пример текста" в зонах,
    - отображаем рамку логотипа.
    """
    img = Image.open(image_path).convert("RGBA")
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
        # Пытаемся адаптировать текст под ширину w
        lines = wrap_text(sample_text, font, w)
        
        # Начальная позиция: вертикально по центру области
        total_height = sum(font.getsize(line)[1] for line in lines)
        current_y = y + (h - total_height)//2
        for line in lines:
            line_width, line_height = font.getsize(line)
            line_x = x + (w - line_width)//2
            draw.text((line_x, current_y), line, font=font, fill=text_color)
            current_y += line_height
    
    # Рамка для логотипа
    if 'logo_position' in template:
        x, y, w, h = template['logo_position']
        draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)
    
    preview = io.BytesIO()
    img.save(preview, format='PNG')
    preview.seek(0)
    return preview

def wrap_text(text, font, max_width):
    """
    Примитивная функция переноса слов, чтобы не вылезали за max_width.
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = current_line + [word]
        w, _ = font.getsize(' '.join(test_line))
        if w <= max_width:
            current_line = test_line
        else:
            # перенос
            lines.append(' '.join(current_line))
            current_line = [word]
    
    # добавляем последнюю строку
    if current_line:
        lines.append(' '.join(current_line))
    return lines

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """
    Обработка подтверждения/отмены шаблона.
    """
    user_id = call.from_user.id
    action, template_name = call.data.split('_', 1)
    
    if action == 'confirm':
        bot.answer_callback_query(call.id, "Шаблон сохранен!")
        bot.send_message(user_id, f"Шаблон '{template_name}' успешно сохранен!")
    elif action == 'cancel':
        # Удаляем шаблон
        template_path = os.path.join(TEMPLATES_DIR, str(user_id), f"{template_name}.json")
        if os.path.exists(template_path):
            os.remove(template_path)
        bot.answer_callback_query(call.id, "Шаблон удалён")
        bot.send_message(user_id, f"Шаблон '{template_name}' удален")

@bot.message_handler(commands=['create'])
def create_post(message):
    """
    Позволяем выбрать один из сохранённых шаблонов и создать пост.
    """
    user_id = message.from_user.id
    user_dir = os.path.join(TEMPLATES_DIR, str(user_id))
    
    if not os.path.exists(user_dir) or not os.listdir(user_dir):
        bot.reply_to(message, "У вас нет сохраненных шаблонов. Создайте их через /learn.")
        return
    
    # Создаем клавиатуру с шаблонами
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    templates = []
    # Сканируем директорию user_dir, собираем имена файлов .json
    for entry in os.scandir(user_dir):
        if entry.is_file() and entry.name.endswith('.json'):
            templates.append(os.path.splitext(entry.name)[0])
    
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
    """Пользователь выбирает шаблон из списка, затем присылает фото + подпись."""
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
    """
    Получаем новое изображение + подпись (caption), заполняем по шаблону.
    Выводим результат обратно.
    """
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
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file)).convert("RGBA")
        
        # Накладываем текстовые блоки из template['text_areas']
        draw = ImageDraw.Draw(img)
        
        for area in template.get('text_areas', []):
            x, y, w, h = area['position']
            text_color = tuple(area['color'])
            font_path = area.get('font_family', "arial.ttf")
            font_size = area.get('font_size', 24)
            
            # Автоматически уменьшаем шрифт, чтобы текст влез.
            final_font = fit_text_in_area(text, font_path, font_size, w, h)
            
            lines = wrap_text_multi(text, final_font, w, h)
            
            # Подсчитаем общую высоту
            total_height = sum(final_font.getsize(line)[1] for line in lines)
            current_y = y + (h - total_height)//2
            
            for line in lines:
                line_width, line_height = final_font.getsize(line)
                line_x = x + (w - line_width)//2
                draw.text((line_x, current_y), line, font=final_font, fill=text_color)
                current_y += line_height
        
        # Добавляем логотип
        if 'logo_position' in template:
            lx, ly, lw, lh = template['logo_position']
            if os.path.exists("default_logo.png"):
                logo = Image.open("default_logo.png").convert("RGBA")
                # При необходимости уменьшим логотип под размер (lw, lh)
                logo.thumbnail((lw, lh), Image.ANTIALIAS)
                # Вставляем логотип с учётом прозрачности
                img.paste(logo, (lx, ly), logo)
            # Иначе можно вставить заглушку или пропустить
        
        # Сохраняем результат в буфер
        result = io.BytesIO()
        img.save(result, format='PNG')
        result.seek(0)
        
        bot.send_photo(user_id, result, "Ваш пост готов!")
        
    except Exception as e:
        bot.reply_to(message, f"Ошибка генерации: {str(e)}")

def fit_text_in_area(text, font_path, initial_size, max_width, max_height):
    """
    Итеративно уменьшаем шрифт, пока не гарантируем, что текст влезет (по высоте).
    Предполагаем, что ужимаем строки в wrap_text_multi.
    """
    size = initial_size
    while size > 10:
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = ImageFont.load_default()
        
        lines = wrap_text_multi(text, font, max_width, max_height)
        # Считаем общую высоту
        total_height = sum(font.getsize(line)[1] for line in lines)
        
        if total_height <= max_height:
            return font
        
        size -= 2
    
    return ImageFont.load_default()

def wrap_text_multi(text, font, max_width, max_height):
    """
    Перенос слов, который прерывает цикл, если текст вышел за пределы max_height.
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = current_line + [word]
        w, h = font.getsize(' '.join(test_line))
        if w <= max_width:
            current_line = test_line
        else:
            # перенос строки
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Дополнительно можно обрезать, если строк слишком много (чтобы не вылезть за max_height)
    # Однако здесь мы просто вернём все строки, а выше при fit_text_in_area() уменьшаем шрифт.
    return lines

if __name__ == "__main__":
    bot.polling()
