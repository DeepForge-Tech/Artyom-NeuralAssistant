# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import random
# # import os 
# # import json
# # from sklearn.feature_extraction.text import CountVectorizer
# # from tqdm import tqdm,trange
# # from time import sleep

# # np.random.seed(0)

# # BATCH_SIZE = 50
# # EPOCHS = 2500
# # LOSS = 0
# # ALPHA = 0.1
# # CATEGORIES = {
# #     '1':'communication',
# #     '2':'weather',
# #     '3':'youtube',
# #     '4':'webbrowser',
# #     '5':'music',
# #     '6':'news',
# #     '7':'todo',
# #     '8':'calendar',
# #     '9':'joikes'
# # }
# # # CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']
# # learning_rate = 0.1
# # LossArray = []
# # train_data = {}
# # test_data = {}


# # # Read data and setup maps for integer encoding and decoding.
# # ProjectDir = os.getcwd()
# # file = open('Datasets/MarcusDataset.json','r',encoding='utf-8')
# # DataFile = json.load(file)
# # train_data = DataFile['train_dataset']
# # test_data = DataFile['test_dataset']

# # vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
# # vocab_size = len(vocab)
 
# # print('%d unique words found' % vocab_size)
# # # Assign indices to each word.
# # word_to_idx = { w: i for i, w in enumerate(vocab) }
# # idx_to_word = { i: w for i, w in enumerate(vocab) }

# # from sklearn import datasets
# # iris = datasets.load_iris()
# # dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

# # def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))

# # def deriv_sigmoid(y):
# #     return y * (1 - y)

# # def tanh(x):
# #     return np.tanh(x)

# # def softmax(xs):
# #     # Applies the Softmax Function to the input array.
# #     return np.exp(xs) / sum(np.exp(xs))

# # def deriv_tanh(y):
# #     return 1 - y * y

# # def cross_entropy(PredictedValue,Target):
# #     return -np.log(PredictedValue[0, Target])

# # def MSE(PredictedValue,TargetValue):
# #     Loss = ((TargetValue - PredictedValue) ** 2).mean()
# #     return Loss

# # class NeuralNetwork():
# #     def __init__(self,LENGHT_DATA,*args,**kwargs):
# #         self.Loss = 0
# #         self.LossArray = []
# #         self.Accuracy = 0
# #         self.AccuracyArray = []

# #         self.LENGHT_DATA = LENGHT_DATA
# #         self.INPUT_LAYERS = self.LENGHT_DATA
# #         self.HIDDEN_LAYERS = 128
# #         self.OUTPUT_LAYERS = 10
# #         # Weights
# #         self.whh = np.random.randn(self.HIDDEN_LAYERS, self.HIDDEN_LAYERS) / 1000
# #         self.wxh = np.random.randn(self.HIDDEN_LAYERS, self.INPUT_LAYERS) / 1000
# #         self.why = np.random.randn(self.OUTPUT_LAYERS, self.HIDDEN_LAYERS) / 1000

# #         # Biases
# #         self.bh = np.zeros((self.HIDDEN_LAYERS, 1))
# #         self.by = np.zeros((self.OUTPUT_LAYERS, 1))

# #     def PreprocessingText(self,text):
# #         '''
# #         Возвращает массив унитарных векторов
# #         которые представляют слова в введенной строке текста
# #         - текст является строкой string
# #         - унитарный вектор имеет форму (vocab_size, 1)
# #         '''
        
# #         Input = []
# #         for w in text.split(' '):
# #             v = np.zeros((vocab_size, 1))
# #             v[word_to_idx[w]] = 1
# #             Input.append(v)
# #         return Input

# #     def FeedForward(self,Input):
# #         '''
# #         Выполнение фазы прямого распространения нейронной сети с
# #         использованием введенных данных.
# #         Возврат итоговой выдачи и скрытого состояния.
# #         - Входные данные в массиве однозначного вектора с формой (input_size, 1).
# #         '''
# #         h = np.zeros((self.whh.shape[0], 1))
 
# #         self.last_inputs = Input
# #         self.last_hs = { 0: h }
 
# #         # Выполнение каждого шага нейронной сети RNN
# #         for i, x in enumerate(Input):
# #             h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
# #             self.last_hs[i + 1] = h
 
# #         # Подсчет вывода
# #         y = self.why @ h + self.by
 
# #         return y, h

# #     def BackwardPropagation(self,d_y):
# #         '''
# #         Выполнение фазы обратного распространения RNN.
# #         - d_y (dL/dy) имеет форму (output_size, 1).
# #         - learn_rate является вещественным числом float.
# #         '''
# #         n = len(self.last_inputs)
 
# #         # Вычисление dL/dWhy и dL/dby.
# #         d_why = d_y @ self.last_hs[n].T
# #         d_by = d_y
 
# #         # Инициализация dL/dWhh, dL/dWxh, и dL/dbh к нулю.
# #         d_whh = np.zeros(self.whh.shape)
# #         d_wxh = np.zeros(self.wxh.shape)
# #         d_bh = np.zeros(self.bh.shape)
 
# #         # Вычисление dL/dh для последнего h.
# #         d_h = self.why.T @ d_y
 
# #         # Обратное распространение во времени.
# #         for t in reversed(range(n)):
# #             # Среднее значение: dL/dh * (1 - h^2)
# #             temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
 
# #             # dL/db = dL/dh * (1 - h^2)
# #             d_bh += temp
 
# #             # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
# #             d_whh += temp @ self.last_hs[t].T
 
# #             # dL/dWxh = dL/dh * (1 - h^2) * x
# #             d_wxh += temp @ self.last_inputs[t].T
 
# #             # Далее dL/dh = dL/dh * (1 - h^2) * Whh
# #             d_h = self.whh @ temp
 
# #         # Отсекаем, чтобы предотвратить разрыв градиентов.
# #         for d in [d_wxh, d_whh, d_why, d_bh, d_by]:
# #             np.clip(d, -1, 1, out=d)
 
# #         # Обновляем вес и смещение с использованием градиентного спуска.
# #         self.whh -= learning_rate * d_whh
# #         self.wxh -= learning_rate * d_wxh
# #         self.why -= learning_rate * d_why
# #         self.bh -= learning_rate * d_bh
# #         self.by -= learning_rate * d_by

# #     def train(self,data,BackPropagation):
# #         for epoch in range(EPOCHS):
# #             for i in range(len(data) // 50):

# #                 batch_x, batch_y = zip(*data[i*50 : i*50+50])
# #                 x = np.concatenate(batch_x, axis=0)
# #                 y = np.array(batch_y)
# #                 PredictedValue = self.FeedForward(x)
# #                 STG = PredictedValue
# #                 STG[y] -= 1
# #                 # Backward Propagation
# #                 self.BackwardPropagation(STG)
# #         # bar = trange(EPOCHS,leave=True)
# #         # iteration = 0
# #         # for epoch in bar:
# #         #     items = list(data.items())
# #         #     # random.shuffle(items)

# #         #     loss = 0
# #         #     num_correct = 0

# #         #     for x, y in items:
# #         #         Input = self.PreprocessingText(x)
# #         #         Target = int(y)

# #         #         # Forward
# #         #         Output, OutputHiddenLayer = self.FeedForward(Input)
# #         #         PredictedValue = softmax(Output)

# #         #         # Calculate loss / accuracy
# #         #         loss = MSE(Output,Target)
# #         #         num_correct += int(np.argmax(PredictedValue) == Target)
# #         #         if BackPropagation:
# #         #           # Build dL/dy
# #         #           STG = PredictedValue
# #         #           STG[Target] -= 1
# #         #           # Backward Propagation
# #         #           self.BackwardPropagation(STG)
# #         #         train_loss, train_acc = loss / len(data), num_correct / len(data)
# #         #         bar.set_description(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
# #         #         # if iteration % 100 ==  0:
# #         #         #     print(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
# #         #         # iteration += 1
# #         #     LossArray.append(loss)
# #         # print(LossArray)
# #         # plt.plot(EPOCHS,np.array(LossArray))
# #         # plt.show()
# #     def predict(self,data:str):
# #         Input = self.PreprocessingText(str(data))
# #         print('Input')
# #         print(Input)
# #         # Forward
# #         Output, OutputHiddenLayer = self.FeedForward(Input)
# #         PredictedValue = softmax(Output)
# #         print(PredictedValue)
# #         print('Argmax:' + str(np.argmax(Output)))
# #         print(f'{CATEGORIES[np.argmax(Output)] + 1}')


# # network = NeuralNetwork(vocab_size)
# # network.train(dataset,True)


# # import numpy as np
# # import random
# # import matplotlib.pyplot as plt
# # from tqdm import tqdm,trange
# # from time import sleep
# # import os
# # import json
# # import re
# # import nltk
# # from nltk.corpus import stopwords
# # from nltk.tokenize import word_tokenize
# # import string
# # from sklearn.feature_extraction.text import TfidfVectorizer

# # # Параметры обучения
# # EPOCHS = 2500
# # BATCH_SIZE = 50
# # learning_rate = 0.0002
# # tokenize_text = []
# # train_dataset = {}
# # test_dataset = {}

# # def tanh(x):
# #     return np.tanh(x)

# # def softmax(xs):
# #     # Applies the Softmax Function to the input array.
# #     return np.exp(xs) / sum(np.exp(xs))

# # def deriv_tanh(y):
# #     return 1 - tanh(y) * tanh(y)

# # def cross_entropy(PredictedValue,Target):
# #     return -np.log(PredictedValue[0, Target])

# # def MSE(PredictedValue,TargetValue):
# #     Loss = ((TargetValue - PredictedValue) ** 2).mean()
# #     return Loss

# # def CountAccuracy(PredictedValue,Target,LENGHT_DATA):
# #     CorrectPredictions = 0
# #     PredictedValue = np.argmax(PredictedValue)
# #     if PredictedValue == Target:
# #         CorrectPredictions += 1

# #     Accuracy = CorrectPredictions / LENGHT_DATA
# #     return Accuracy

# # class NeuralNetwork:
# #     def __init__(self,LENGHT_DATA):

# #         # Размер датасета
# #         self.LENGHT_DATA = LENGHT_DATA

# #         # Определние входных,скрытых,выходных слоёв
# #         self.INPUT_LAYERS = self.LENGHT_DATA
# #         self.HIDDEN_LAYERS = 128
# #         self.OUTPUT_LAYERS = 10

# #         # Инициализция гиперпараметров
# #         self.ht = 0
# #         self.hPrevious = 0
# #         self.Output = 0
# #         self.Loss = 0
# #         self.LossArray = []
# #         self.Accuracy = 0
# #         self.AccuracyArray = []

# #         # Инициализация весов
# #         self.whh = np.random.randn(self.HIDDEN_LAYERS, self.HIDDEN_LAYERS) / 1000
# #         self.wxh = np.random.randn(self.HIDDEN_LAYERS, self.INPUT_LAYERS) / 1000
# #         self.why = np.random.randn(self.OUTPUT_LAYERS, self.HIDDEN_LAYERS) / 1000

# #         # Инициализация смещений
# #         self.bh = np.zeros((self.HIDDEN_LAYERS, 1))
# #         self.by = np.zeros((self.OUTPUT_LAYERS, 1))

# #     def PreprocessingText(self,text:str):
# #         Input = []
# #         for w in text.split(' '):
# #             v = np.zeros((vocab_size, 1))
# #             v[word_to_idx[w]] = 1
# #             Input.append(v)
# #         return np.squeeze(np.array(Input))
# #         # text = text.lower()
# #         # text = re.sub(r'\d+', '', text)
# #         # translator = str.maketrans('', '', string.punctuation)
# #         # text.translate(translator)
# #         # text = text
# #         # vectorizer = TfidfVectorizer()
# #         # vectorized_text = vectorizer.fit_transform(text.split('\n'))
# #         # vectorized_text = vectorizer.transform(text.split('\n'))
# #         # vectorized_text = vectorized_text.toarray()
# #         # return np.squeeze(np.array(vectorized_text))

# #     def ForwardPropagation(self,Input):
# #         self.Input = Input
# #         self.ht = np.zeros((self.whh.shape[0], 1))
# #         # self.yt = np.zeros((self.why.shape[0], 1))
# #         for x in Input:
# #             print(x)
# #             self.ht = np.tanh(np.dot(self.wxh,x) + np.dot(self.whh,self.hPrevious) + self.bh)
# #             self.hPrevious = self.ht
# #         self.yt = np.dot(self.why,self.ht) + self.by
# #         self.Output = self.yt
# #         return self.Output

# #     def BackwardPropagation(self,PredictedValue,Target):
# #         d_why = np.zeros_like(self.why)
# #         d_whh = np.zeros_like(self.whh)
# #         d_wxh = np.zeros_like(self.wxh)

# #         Error = PredictedValue - Target
# #         print(deriv_tanh(PredictedValue))
# #         OutputGradient = Error * deriv_tanh(PredictedValue)
# #         d_why = OutputGradient * self.ht.T
# #         HiddenGradient = OutputGradient * self.why.T * deriv_tanh(self.ht.T)
# #         d_whh = HiddenGradient * self.ht.T
# #         InputSum = HiddenGradient * self.whh
# #         InputGradient = InputSum * deriv_tanh(self.Input)
# #         d_wxh = InputGradient * self.Input
# #         self.why = self.why - learning_rate * d_why
# #         self.whh = self.whh - learning_rate * d_whh
# #         self.wxh = self.wxh - learning_rate * d_wxh
# #     def train(self,dataset):
# #         progressbar = trange(EPOCHS,leave=True)
# #         for epoch in progressbar:
# #             items = list(dataset.items())
# #             random.shuffle(items)

# #             self.Loss = 0
# #             self.Accuracy = 0

# #             for x, y in items:
# #                 Input = self.PreprocessingText(x)
# #                 Target = int(y)
# #                 PredictedValue = self.ForwardPropagation(Input)
# #                 self.BackwardPropagation(np.argmax(PredictedValue),Target)
# #                 self.Loss = MSE(PredictedValue,Target)
# #                 self.Accuracy = CountAccuracy(PredictedValue,Target,len(items))
# #                 self.LossArray.append(self.Loss)
# #                 self.AccuracyArray.append(self.Accuracy)
    
# #     def predict(self,data:str):
# #         Input = self.Preprocessing(data)
# #         PredictedValue = self.ForwardPropagation(Input)
# #         return np.argmax(PredictedValue)

# # ProjectDir = os.getcwd()
# # file = open('Datasets/MarcusDataset.json','r',encoding='utf-8')
# # DataFile = json.load(file)
# # train_dataset = DataFile['train_dataset']
# # test_dataset = DataFile['test_dataset']

# # vocab = list(set([w for text in train_dataset.keys() for w in text.split(' ')]))
# # vocab_size = len(vocab)
 
# # print('%d unique words found' % vocab_size)
# # # Assign indices to each word.
# # word_to_idx = { w: i for i, w in enumerate(vocab) }
# # idx_to_word = { i: w for i, w in enumerate(vocab) }
# # network = NeuralNetwork(vocab_size)
# # network.train(train_dataset)

# import math
# text = 'абвгдежзийклмнопрстуфхцчшщъыьэюя.,-_'
# n = math.ceil((math.sqrt(len(text)))) # получение размера квадратной матрицы
# text = iter(text)
# data = [[next(text) for _ in range(6)] for i in range(n)]
# print(data)
# for i in range(n):
#     data.append([])
#     for char in text[i * n: (i + 1) * n]:
#         data[-1].append(char)
# 
import random
d = {'a':1, 'b':2, 'c':3, 'd':4}
l = list(d.items())
random.shuffle(l)
d = dict(l)
print(d)