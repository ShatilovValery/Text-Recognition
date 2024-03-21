import cv2
import numpy as np
import keras
import os
import csv


def letter_finder(file):
    if not os.path.exists('letter'):
        os.mkdir('letter')
    #сохраняем путь к файлу в переменную
    image_file = file
    #загружаем изображение из файла
    img = cv2.imread(image_file)
    print(file)
    #преобразовывем изображение в ЧБ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #сегментация изображения
    blur = cv2.blur(gray, (5,5))
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    print(file)
    #накладываем эффект рамытия 
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    #используя функцию findContours находим контуры букв на обработанном изображении
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #делаем копию изображения для отрисовки контуров букв
    output = img.copy()
    #создаем список для вырезанных букв из озибражения
    letters = []
    #проходимся циклом по всем контурам

    for idx, contour in enumerate(contours):
        #получаем координаты прямоугольника на изображении
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            #вырезаем буквы из картинки 
            letter_crop = gray[y:y + h, x:x + w]
            t,letter_crop_tresh = cv2.threshold(letter_crop, 100, 255, cv2.THRESH_BINARY)
            letter_crop = cv2.resize(letter_crop_tresh, (28, 28), interpolation=cv2.INTER_AREA)
            ret, thresh = cv2.threshold(letter_crop, 100, 255, cv2.THRESH_BINARY)
            #накладываем эффект рамытия 
            img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

            #добавляем в список letters координату х ширины и изображение буквы
            if not w * h < 200:
                #рисуем прямоугольник на изображении
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # t,img = cv2.threshold(letter_crop, 100, 255, cv2.THRESH_BINARY)
                letters.append((x, w, img_erode))
    #сортируем список letter по координате х
    letters.sort(key=lambda x: x[0], reverse=False)
    #сохраняем копию изображения с выделенными биквами
    cv2.imwrite(f'output.jpg', output)
    #возвращаем список letters из функции
    return letters


def predict_img(model, img):
    emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
    #расширяем форму массива
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    #переворачиваем изображение на 90 градусов тк в датасете они повернуты 
    # img_arr[0] = np.rot90(img_arr[0], 3)
    # img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    #закидываем в обученную модель изображение
    predict = model.predict(img_arr)
    #выбираем элемент с максимальной вероятностью
    result = np.argmax(predict, axis=1)
    #возвращаем буквук
    return chr(emnist_labels[result[0]])


def img_to_str(model, image_file):
    #получаем список букв
    letters = letter_finder(image_file)
    #объявляем переменную result в нее мы будем сохранять распознанный текст
    result = ""
    #проходимся по этому списку
    for i in range(len(letters)):
        #передаем в функцию predict_img только изображение (оно хранится под индексом 2) и добавляем к строке result
        result += predict_img(model, letters[i][2])
    #возращаем строку result
    return result


# model = keras.models.load_model('test_model.h5')
# file = 'source/Vbc.jpg'
# result = img_to_str(model, file)
# print(result)