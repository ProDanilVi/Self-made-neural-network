#Импортируются необходимые библиотеки: NumPy для работы с массивами и Matplotlib для визуализации результатов
import numpy as np
import matplotlib.pyplot as plt
#Импортируется пользовательский модуль utils, вероятно, содержащий функцию load_dataset(), которая загружает набор данных MNIST, состоящий из изображений рукописных цифр и соответствующих им меток
import utils

#Строка вызывает функцию load_dataset() из модуля utils, чтобы загрузить набор данных изображений и их меток
images, labels = utils.load_dataset()

#Инициализируются веса (w_i_h и w_h_o) и смещения (b_i_h и b_h_o) для каждого слоя сети случайными значениями
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

#Задаются параметры обучения: скорость обучения (learn_rate), количество эпох (epochs), переменные для отслеживания правильных ответов (nr_correct) и общей потери (loss)
learn_rate = 0.01
nr_correct = 0
loss = 0
epochs = 3
#Основной внешний цикл for epoch in range(epochs) проходится по заданному количеству эпох обучения (epochs)
for epoch in range(epochs):
    #Во внутреннем цикле for img, l in zip(images, labels), каждое изображение img и его соответствующая метка l из набора данных загружаются последовательно. zip() используется для итерации по изображениям и меткам одновременно
    for img, l in zip(images, labels):
        #Каждое изображение преобразуется в одномерный массив с помощью np.reshape(img, (-1, 1)), аналогично и метка преобразуется в одномерный массив
        img = np.reshape(img, (-1, 1))
        l = np.reshape(l, (-1, 1))

        #Ввод прямого распространения -> скрытый
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        #Скрытое прямое распространение -> вывод
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        #Расчет затрат/ошибок
        loss += 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        #Вывод обратного распространения -> скрытый (производная функции затрат)
        delta_o = o - l
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o
        #Скрытое обратное распространение -> ввод (производная функции активации)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    #Покажите точность для этой эпохи
    print(f"Потеря: {round((loss[0] / images.shape[0]) * 100, 2)}%")
    print(f"Точность: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0
    loss = 0

exit(0)