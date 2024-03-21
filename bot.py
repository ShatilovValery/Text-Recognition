import config as cfg
from aiogram import executor


async def on_start_up(_):
    print("BOT HAS BEEN STARTED")

if __name__ == '__main__':
    import bot_handlers
    executor.start_polling(cfg.dp, skip_updates=False, on_startup=on_start_up)