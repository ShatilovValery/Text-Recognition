from dotenv import load_dotenv
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram import Bot, Dispatcher
import os
import keras


load_dotenv()

BOT_CONFIG = str(os.getenv('BOT_TOKEN'))
storage = MemoryStorage()
bot = Bot(token=BOT_CONFIG)
dp = Dispatcher(bot, storage=storage)
model = keras.models.load_model('models/test_model.h5')