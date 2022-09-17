from itertools import tee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os 
import json
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm,trange
from time import sleep

np.random.seed(42)

BATCH_SIZE = 50
EPOCHS = 25000
LOSS = 0

CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']
learning_rate = 0.0002
LossArray = []
train_data = {}
test_data = {}


# Read data and setup maps for integer encoding and decoding.
ProjectDir = os.getcwd()
file = open('Datasets/MarcusDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def softmax(xs):
    # Applies the Softmax Function to the input array.
    return np.exp(xs) / sum(np.exp(xs))

def deriv_tanh(y):
    return 1 - y * y

def cross_entropy(PredictedValue,Target):
    return -np.log(PredictedValue[0, Target])

def MSE(PredictedValue,TargetValue):
    Loss = ((TargetValue - PredictedValue) ** 2).mean()
    return Loss

class NeuralNetwork():
    def __init__(self,LENGHT_DATA:int,*args,**kwargs):
        self.Loss = 0
        self.LossArray = []
        self.Accuracy = 0
        self.AccuracyArray = []
        self.LocalLoss = 0.5
        self.LocalAccuracy = 1.0

        self.LENGHT_DATA = LENGHT_DATA
        self.INPUT_LAYERS = self.LENGHT_DATA
        self.HIDDEN_LAYERS = 128
        self.OUTPUT_LAYERS = 10
        # Weights
        self.whh = np.random.randn(self.HIDDEN_LAYERS, self.HIDDEN_LAYERS) / 1000
        self.wxh = np.random.randn(self.HIDDEN_LAYERS, self.INPUT_LAYERS) / 1000
        self.why = np.random.randn(self.OUTPUT_LAYERS, self.HIDDEN_LAYERS) / 1000

        # Biases
        self.bh = np.zeros((self.HIDDEN_LAYERS, 1))
        self.by = np.zeros((self.OUTPUT_LAYERS, 1))

    def PreprocessingText(self,text):
        '''
        Возвращает массив унитарных векторов
        которые представляют слова в введенной строке текста
        - текст является строкой string
        - унитарный вектор имеет форму (vocab_size, 1)
        '''
        
        Input = []
        for w in text.split(' '):
            v = np.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            Input.append(v)
        return Input

    def FeedForward(self,Input):
        '''
        Выполнение фазы прямого распространения нейронной сети с
        использованием введенных данных.
        Возврат итоговой выдачи и скрытого состояния.
        - Входные данные в массиве однозначного вектора с формой (input_size, 1).
        '''
        h = np.zeros((self.whh.shape[0], 1))
 
        self.last_inputs = Input
        self.last_hs = { 0: h }
 
        # Выполнение каждого шага нейронной сети RNN
        for i, x in enumerate(Input):
            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            self.last_hs[i + 1] = h
 
        # Подсчет вывода
        y = self.why @ h + self.by
 
        return y, h

    def BackwardPropagation(self,d_y):
        '''
        Выполнение фазы обратного распространения RNN.
        - d_y (dL/dy) имеет форму (output_size, 1).
        - learn_rate является вещественным числом float.
        '''
        n = len(self.last_inputs)
 
        # Вычисление dL/dWhy и dL/dby.
        d_why = d_y @ self.last_hs[n].T
        d_by = d_y
 
        # Инициализация dL/dWhh, dL/dWxh, и dL/dbh к нулю.
        d_whh = np.zeros(self.whh.shape)
        d_wxh = np.zeros(self.wxh.shape)
        d_bh = np.zeros(self.bh.shape)
 
        # Вычисление dL/dh для последнего h.
        d_h = self.why.T @ d_y
 
        # Обратное распространение во времени.
        for t in reversed(range(n)):
            # Среднее значение: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
 
            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp
 
            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_whh += temp @ self.last_hs[t].T
 
            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_wxh += temp @ self.last_inputs[t].T
 
            # Далее dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.whh @ temp
 
        # Отсекаем, чтобы предотвратить разрыв градиентов.
        for d in [d_wxh, d_whh, d_why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)
 
        # Обновляем вес и смещение с использованием градиентного спуска.
        self.whh -= learning_rate * d_whh
        self.wxh -= learning_rate * d_wxh
        self.why -= learning_rate * d_why
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

    def train(self,data:dict,BackPropagation):
        bar = trange(EPOCHS,leave=True)
        iteration = 0
        for epoch in bar:
            items = list(data.items())
            random.shuffle(items)

            loss = 0
            num_correct = 0

            for x, y in items:
                Input = self.PreprocessingText(x)
                Target = int(y)

                # Forward
                Output, OutputHiddenLayer = self.FeedForward(Input)
                PredictedValue = softmax(Output)

                # Calculate loss / accuracy
                # loss = MSE(Output,Target)
                num_correct += int(np.argmax(PredictedValue) == Target)
                if BackPropagation:
                  # Build dL/dy
                  STG = PredictedValue
                  STG[Target] -= 1
                  # Backward Propagation
                  self.BackwardPropagation(STG)
                train_loss, train_acc = loss / len(data), num_correct / len(data)
                
                if train_loss <= self.LocalLoss and train_acc == self.LocalAccuracy:
                    self.LocalLoss = train_loss
                    self.LocalAccuracy = train_acc
                    # print('Best model')
                    self.save()
                bar.set_description(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
                # if iteration % 100 ==  0:
                #     print(f'Epoch: {epoch}/{EPOCHS}; Loss: {train_loss}; Accuracy: {train_acc}')
                # iteration += 1
                LossArray.append(train_loss)
        plt.plot(LossArray)
        plt.show()
    def predict(self,data:str):
        Input = self.PreprocessingText(data)
        # Forward
        Output, OutputHiddenLayer = self.FeedForward(Input)
        PredictedValue = softmax(Output)
        print(PredictedValue)
        print('Argmax:' + str(np.argmax(Output)))
        print(f'{CATEGORIES[np.argmax(Output)]}')

    def save(self):
        np.savez_compressed('Artyom_NeuralAssistant', self.whh,self.wxh,self.why,self.bh,self.by)

    def load(self,PathParametrs = 'Artyom_NeuralAssistant.npz'):
        ParametrsFile = np.load(PathParametrs)
        self.whh = ParametrsFile['arr_0']
        self.wxh = ParametrsFile['arr_1']
        self.why = ParametrsFile['arr_2']
        self.bh = ParametrsFile['arr_3']
        self.by = ParametrsFile['arr_4']

network = NeuralNetwork(vocab_size)
network.train(train_data,True)
network.train(train_data,False)

vocab = list(set([w for text in test_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

network = NeuralNetwork(vocab_size)
network.train(test_data,True)
network.train(test_data,False)

# Тест нейросети на скорость
CorrectPredictions = 0
items = list(train_data.items())
random.shuffle(items)
for x, y in items:
    PredictedValue = network.predict(x)
    if PredictedValue == int(y):
        CorrectPredictions += 1
print(CorrectPredictions)
while True:
    InputData = input('Input data:')
    if InputData != 'stop':
        vocab = list(set([w for w in InputData.split(' ')]))
        vocab_size = len(vocab)
        print('%d unique words found' % vocab_size)
        # Assign indices to each word.
        word_to_idx = { w: i for i, w in enumerate(vocab) }
        print(word_to_idx)
        idx_to_word = { i: w for i, w in enumerate(vocab) }
        network = NeuralNetwork(vocab_size)
        PredictedValue = network.predict(InputData)
        print(PredictedValue)
    else:
        break
    
