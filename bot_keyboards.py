from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton


menu_keyboard = InlineKeyboardMarkup(row_width=1)
recognize_text = InlineKeyboardButton(text='Найти текст на изображении', callback_data='recognize')

menu_keyboard.add(recognize_text)

menu = ReplyKeyboardMarkup(resize_keyboard=True).add(KeyboardButton(text="Найти текст на изображении"))