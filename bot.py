import os
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Update
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

user_templates = {}
user_logos = {}

def analyze_template(image_path):
    logger.info(f"Analyzing template: {image_path}")
    img = cv2.imread(image_path)
    text_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    for i, conf in enumerate(text_data["conf"]):
        if int(conf) > 60:
            text_pos = {"x": text_data["left"][i], "y": text_data["top"][i]}
            break
    else:
        text_pos = {"x": 50, "y": 50}
    return {
        "text_position": text_pos,
        "logo_position": {"x": img.shape[1] - 100, "y": img.shape[0] - 100},
        "background": "gradient"
    }

def create_post(template, photo_path, text, logo_path):
    logger.info("Creating post...")
    base_img = Image.open(photo_path).convert("RGBA")
    width, height = base_img.size
    gradient = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for y in range(height):
        r = int(255 * (y / height))
        g = int(100 * (1 - y / height))
        b = 150
        draw.line([(0, y), (width, y)], fill=(r, g, b, 200))
    result = Image.alpha_composite(base_img, gradient)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    text_pos = template["text_position"]
    draw.text((text_pos["x"], text_pos["y"]), text, font=font, fill=(255, 255, 255, 255))
    logo = Image.open(logo_path).convert("RGBA").resize((100, 100))
    logo_pos = template["logo_position"]
    result.paste(logo, (logo_pos["x"], logo_pos["y"]), logo)
    result_path = "result.png"
    result.save(result_path, "PNG")
    return result_path

def handle_template(update: Update, context):
    logger.info("Handling /template command")
    if not update.message.photo:
        update.message.reply_text("Отправьте фото с шаблоном вместе с /template!")
        return
    photo_file = update.message.photo[-1].get_file()
    photo_path = f"template_{update.message.from_user.id}.jpg"
    photo_file.download(photo_path)
    template = analyze_template(photo_path)
    user_id = update.message.from_user.id
    user_templates[user_id] = template
    update.message.reply_text("Шаблон сохранен! Теперь отправьте логотип, затем фото и текст.")

def handle_logo(update: Update, context):
    user_id = update.message.from_user.id
    if user_id not in user_templates:
        update.message.reply_text("Сначала отправьте шаблон с /template!")
        return
    photo_file = update.message.photo[-1].get_file()
    logo_path = f"logo_{user_id}.png"
    photo_file.download(logo_path)
    user_logos[user_id] = logo_path
    update.message.reply_text("Логотип сохранен! Отправьте фото и текст для поста.")

def handle_content(update: Update, context):
    user_id = update.message.from_user.id
    if user_id not in user_templates:
        update.message.reply_text("Сначала отправьте шаблон с /template!")
        return
    if user_id not in user_logos:
        update.message.reply_text("Сначала отправьте логотип!")
        return
    if update.message.photo and update.message.caption:
        photo_file = update.message.photo[-1].get_file()
        photo_path = f"photo_{user_id}.jpg"
        photo_file.download(photo_path)
        text = update.message.caption
        logo_path = user_logos[user_id]
        template = user_templates[user_id]
        result_path = create_post(template, photo_path, text, logo_path)
        with open(result_path, "rb") as result_file:
            update.message.reply_photo(result_file)
        os.remove(photo_path)
        os.remove(result_path)
    else:
        update.message.reply_text("Отправьте фото и текст в одном сообщении!")

def start(update: Update, context):
    update.message.reply_text(
        "Привет! Я бот для создания постов.\n"
        "1. Отправьте шаблон с /template\n"
        "2. Отправьте логотип отдельно\n"
        "3. Отправьте фото и текст для создания поста"
    )

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        logger.error("TELEGRAM_TOKEN не задан!")
        raise ValueError("TELEGRAM_TOKEN не задан в переменных окружения")
    logger.info("Starting bot with polling...")
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("template", handle_template))
    dp.add_handler(MessageHandler(Filters.photo & ~Filters.caption, handle_logo))
    dp.add_handler(MessageHandler(Filters.photo & Filters.caption, handle_content))
    updater.start_polling()
    logger.info("Bot is running...")
    updater.idle()

if __name__ == "__main__":
    main()
