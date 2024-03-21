from aiogram.dispatcher.filters.state import StatesGroup, State

class GetImageState(StatesGroup):
    image = State()