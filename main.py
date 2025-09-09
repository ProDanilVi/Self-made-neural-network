#Импортирует библиотеку NumPy, которая предоставляет функциональность для работы с многомерными массивами, а также инструменты для работы с линейной алгеброй, преобразованием Фурье и многое другое
import numpy as np
#Импортирует модуль pyplot из библиотеки Matplotlib, который предоставляет функции для создании графиков и визуализации данных
import matplotlib.pyplot as plt
#Импортирует модуль utils
import utils

#Строка вызывает функцию load_dataset() из модуля utils, чтобы загрузить набор данных изображений и их меток
images, labels = utils.load_dataset()

#Создается матрица весов weights_input_to_hidden размером (20, 784) с помощью функции numpy.random.uniform(), которая заполняет ее случайными значениями из равномерного распределения в диапазоне от -0.5 до 0.5. Эти веса представляют собой веса между входным слоем и скрытым слоем нейронов
weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (20, 784))
#Аналогично создается матрица весов weights_hidden_to_output размером (10, 20) для связей между скрытым слоем и выходным слоем
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (10, 20))
#Создается вектор смещений для скрытого слоя bias_input_to_hidden размером (20, 1) и заполняем нулями
bias_input_to_hidden = np.zeros((20, 1))
#Создается вектор смещений для выходного слоя bias_hidden_to_output размером (10, 1) и также заполняется нулями
bias_hidden_to_output = np.zeros((10, 1))

#Устанавливается количество эпох обучения (в данном случае, 3 эпохи)
epochs = 3
#Инициализируются переменные e_loss и e_correct для отслеживания общей потери и количества правильно предсказанных образцов в каждой эпохе соответственно
e_loss = 0
e_correct = 0
#Устанавливается коэффициент скорости обучения, который определяет размер шага обновления весов в процессе градиентного спуска
learning_rate = 0.01

#Внешний цикл проходится по заданному количеству эпох обучения epochs. В каждой эпохе происходит проход по всем изображениям и их меткам
for epoch in range(epochs):
    print(f"Эпоха №{epoch}")

    #Внутренний цикл перебирает каждое изображение и соответствующую ему метку
    for image, label in zip(images, labels):
        #Преобразуют изображение и метку в формат столбцовых векторов
        image = np.reshape(image, (-1, 1))
        label = np.reshape(label, (-1, 1))

        #Прямое распространение (на скрытый слой)
        hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
        hidden = 1 / (1 + np.exp(-hidden_raw)) #сигмовидная

        #Прямое распространение (на выходной уровень)
        output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
        output = 1 / (1 + np.exp(-output_raw))

        #Расчет потерь/ошибок
        e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
        e_correct += int(np.argmax(output) == np.argmax(label))

        #Обратное распространение (выходной уровень)
        delta_output = output - label
        weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
        bias_hidden_to_output += -learning_rate * delta_output

        #Обратное распространение (скрытый слой)
        delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * (hidden * (1 - hidden))
        weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
        bias_input_to_hidden += -learning_rate * delta_hidden

    #Выведение отладочную информацию между эпохами
    print(f"Потеря: {round((e_loss[0] / images.shape[0]) * 100, 3)}%")
    print(f"Точность: {round((e_correct / images.shape[0]) * 100, 3)}%")
    e_loss = 0
    e_correct = 0

test_image = plt.imread("3.jpg", format="jpg")

#Оттенки серого + Единица измерения RGB + обратные цвета
gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 
test_image = 1 - (gray(test_image).astype("float32") / 255)

#Преобразует изображение test_image из двумерного массива в одномерный массив
test_image = np.reshape(test_image, (test_image.shape[0] * test_image.shape[1]))

#Переменная image преобразуется обратно в двумерный массив, где каждый пиксель изображения представлен в отдельной строке
image = np.reshape(test_image, (-1, 1))

#Прямое распространение (на скрытый слой)
hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
hidden = 1 / (1 + np.exp(-hidden_raw)) #сигмовидная
#Прямое распространение (на выходной уровень)
output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
output = 1 / (1 + np.exp(-output_raw))

#Исходное изображение test_image отображается с помощью функции plt.imshow
plt.imshow(test_image.reshape(28, 28), cmap="Greys")
#На графике отображается заголовок, который указывает результат предсказания модели
plt.title(f"ИИ предполагает, что пользовательский номер равен: {output.argmax()}")
#График отображается с помощью функции plt.show()
plt.show()