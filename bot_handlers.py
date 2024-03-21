from config import dp, bot, model
from aiogram import types
from bot_keyboards import menu_keyboard, menu
from aiogram.dispatcher import FSMContext
from bot_states import GetImageState
import time as ti
import os
from rec_text import img_to_str

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    await bot.send_message(chat_id=message.from_user.id, text=f"""Привет! {message.from_user.username}\nя бот для распознования текста на изображении!""", parse_mode=types.ParseMode.MARKDOWN, reply_markup=menu)


@dp.message_handler(text='Найти текст на изображении')
async def recognize_handler(message: types.Message):
    await bot.send_message(chat_id=message.chat.id, 
                           text='Отправь мне изображение с текстом', 
                           parse_mode=types.ParseMode.MARKDOWN,
                           reply_markup=types.ReplyKeyboardRemove())
    await GetImageState.image.set()


@dp.message_handler(state=GetImageState.image, content_types=types.ContentType.DOCUMENT)
async def set_image(message: types.Message, state: FSMContext):
    if not os.path.exists(str(message.chat.id)):
        os.mkdir(str(message.chat.id))
    time = ti.time()
    file = f"{message.chat.id}/{message.from_user.id}-{time}.jpg"
    await message.document.download(destination_file=file)
    rec_text = img_to_str(model, file)
    await state.update_data(image=message.document)
    await state.finish()
    await bot.send_message(chat_id=message.chat.id, text="Изображение успешно сохраненно!")
    if rec_text != 'Some error!':
        await bot.send_photo(chat_id=message.chat.id, 
                            photo=open('output.jpg', 'rb'),
                            caption=f"Распознанный текст на изображении:  <b>{rec_text}</b>", 
                            parse_mode=types.ParseMode.HTML,
                            reply_markup=menu)
    else:
        await bot.send_message(chat_id=message.chat.id, text="<b>Some Error</b>", parse_mode=types.ParseMode.HTML)
    await state.finish()


    