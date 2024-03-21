from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
import numpy as np
import keras
import numpy as np
import idx2numpy



def make_model():
    #Указываем путь до датасета на котором будем проводить обучение модели
    emnist_path = 'data/'
    #Подгрузка заголовков
    emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 
                     69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
                     83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 
                     103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 
                     115, 116, 117, 118, 119, 120, 121, 122]
    #Загрузка данных из датасета
    

    X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')
    X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

    #Предварительная обработка данных
    X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))
    k = 10
    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    #Приводим все числа в массивах к формату 0 - 1
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    X_test = X_test.astype(np.float32)
    X_test /= 255.0
    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    #Создаем модель сверточной нейронной сети
    model = Sequential()

    #Создаем 2 слоя двумерной свертки
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))

    #Добавляем фильтер maxPolling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #добавляем слой который решает проблему при переобчении
    model.add(Dropout(0.25))
    model.add(Flatten())

    #добавляем слой который получает иформацию с других слоев
    model.add(Dense(512, activation='relu'))

    #добавляем слой который решает проблему при переобчении с повышенным рэйтом
    model.add(Dropout(0.5))

    #добавляем слой который получает иформацию с других слоев с другой функцией активации
    model.add(Dense(len(emnist_labels), activation='softmax'))

    #добавляем метрики указваем функции потерь и оптимизаторы
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.compile(optimizer = Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    #создаем callback который будет оптимизировать процесс обучения
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                                patience=3, 
                                                                verbose=1, 
                                                                factor=0.5, 
                                                                min_lr=0.00001)
    
    #обучаем модель на ранее загруженных данных
    history = model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), 
              callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

    print("Точность проверки равна :", history.history['val_accuracy'])
    print("Точность обучения равна :", history.history['accuracy'])
    print("Потеря проверки равна :", history.history['val_loss'])
    print("Потеря при обучении равна :", history.history['loss'])
    #сохраняем модель 
    model.save('model_epochs_30_test.h5')
# make_model()
