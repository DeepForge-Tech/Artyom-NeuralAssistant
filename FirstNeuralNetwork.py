from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas
import os
import json

INPUT_DIM = 28
HIDDEN_DIM = 128
OUTPUT_DIM = 10

# Состояние выхода сети
Output = 0
#Состояние памяти 
StateMemory = 0
# Веса входа сети для входного гейта
InputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса выхода сети для входного гейта
IGOutputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса входа сети для гейта памяти
IMInputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса входа сети для гейта памяти
IFGMemoryWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса выхода сети для гейта памяти
FGOutputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса состояние памяти сети для гейта памяти
FGMemoryWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса входа сети для изменения памяти
IEMemoryWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса выхода сети для изменения памяти
EMoutputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса входа сети для выходного гейта
OGInputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса выхода сети для выходного гейта
OGOutputWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Веса состояние памяти сети для выходного гейта
OGMemoryWeights = np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)
# Порог вхождения
learning_rate = 0.1
ALPHA = 0.0002
EPOCHS = 10000
BATCH_SIZE = 100

LossArray = []

# Функция активации значения входного гейта
def sigmoid(array):
  # Наша функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-array))

def softmax(array):
    out = np.exp(array)
    return out / np.sum(out)

def cross_entropy(Prediction,Label):
    entropy = Label * np.log(Prediction + 1e-6) # to prevent log value overflow
    return -np.sum(entropy, axis=1, keepdims=True)

# Функция активации
def tanh(array):
    return np.tanh(array)

class Neuron():
    def __init__(self,*args,**kwargs):
        self.learning_rate = learning_rate

    def CountInputGate(self,Input):
        InputGate = sigmoid(InputWeights * Input + IGOutputWeights * Output + IMInputWeights * StateMemory + learning_rate)
        return InputGate

    def CountForgetGate(self,Input,Output):
        ForgetGate = sigmoid(IFGMemoryWeights * Input + FGOutputWeights * Output + FGMemoryWeights * StateMemory + self.learning_rate)
        return ForgetGate

    def CountEditMemory(self,Input):
        EditMemory = tanh(IEMemoryWeights * Input + EMoutputWeights * Output + self.learning_rate)
        return EditMemory

    def UpdateStateMemory(self,InputGate,ForgetGate,EditMemory,StateMemory):
        StateMemory = EditMemory * InputGate + StateMemory * ForgetGate
        return StateMemory

    def CountOutputGate(self,Input,Output,StateMemory):
        OutputGate = sigmoid(OGInputWeights * Input + OGOutputWeights * Output + OGMemoryWeights * StateMemory + self.learning_rate)
        return OutputGate

    def CountOutput(self,StateMemory,OutputGate):
        Output = tanh(StateMemory) * OutputGate
        return Output

    def FeedForward(self,Input):
        global StateMemory
        global Output
        InputGate = self.CountInputGate(Input)
        ForgetGate = self.CountForgetGate(Input,Output)
        EditMemory = self.CountEditMemory(Input)
        StateMemory = self.UpdateStateMemory(InputGate,ForgetGate,EditMemory,StateMemory)
        OutputGate = self.CountOutputGate(Input,Output,StateMemory)
        Output = self.CountOutput(StateMemory,OutputGate)
        return np.argmax(Output)

# Класс нейросети
class NeuralNetwork():
    def __init__(self,*args,**kwargs):
        self.bias = learning_rate
        self.h1 = Neuron(self.bias)
        self.h2 = Neuron(self.bias)
        self.h3 = Neuron(self.bias)
        self.o1 = Neuron(self.bias)

    def FeedForward(self,Input):
        output_h1 = self.h1.FeedForward(Input)
        output_h2 = self.h2.FeedForward(Input)
        output_h3 = self.h3.FeedForward(Input)
        output_o1 = self.o1.FeedForward(np.array([output_h1,output_h2,output_h3]))

        return output_o1

    def predict(self,text):
        pass
    
    def train(self,dataset):
        for epoch in range(EPOCHS):
            Prediction = 1
            Label = 1
            Loss = np.sum(cross_entropy(Prediction,Label))
            LossArray.append(Loss)


newsgroups_train = fetch_20newsgroups(subset='train')
network = Neuron(0.0002)
print(network.FeedForward(np.random.randn(INPUT_DIM, HIDDEN_DIM) / np.sqrt(INPUT_DIM / 2)))
