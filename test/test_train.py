import emnist
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from tensorflow import keras


x, y = emnist.extract_training_samples('byclass')
model = Sequential()

#Создаем 2 слоя двумерной свертки
#Creating 2 layers of two-dimensional convolution
model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))

#Добавляем фильтер maxPolling
#Adding the max Polling filter
model.add(MaxPooling2D(pool_size=(2, 2)))

#добавляем слой который решает проблему при переобучении
#we add a layer that solves the problem of retraining
model.add(Dropout(0.25))
model.add(Flatten())

#добавляем слой который получает иформацию с других слоев
#adding a layer that receives information from other layers
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

#добавляем слой который решает проблему при переобчении с повышенным рэйтом
#we add a layer that solves the problem of retraining with an increased rate
model.add(Dropout(0.5))

#добавляем слой который получает иформацию с других слоев с другой функцией активации
#adding a layer that receives information from other layers with a different activation function
model.add(Dense(62, activation='softmax'))

#добавляем метрики указваем функции потерь и оптимизаторы
#we add metrics, specify loss functions and optimizers
model.compile(optimizer = Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#создаем callback который будет оптимизировать процесс обучения
#creating a callback that will optimize the learning process
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
															patience=3, 
															verbose=1, 
															factor=0.5, 
															min_lr=0.00001)

#обучаем модель на ранее загруженных данных
#training the model on previously uploaded data
model.fit(x, y, epochs=30)
model.save('test_3_layers.h5')