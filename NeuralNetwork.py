import numpy as np
import matplotlib.pyplot as plt
from PreprocessingText import PreprocessingDataset
from tqdm import trange
import os
import json
from rich.progress import track,Progress

EPOCHS = 10000
learning_rate = 0.0002

# Read data and setup maps for integer encoding and decoding.
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']
Preprocessing = PreprocessingDataset()
TrainInput,TrainTarget = Preprocessing.Start(train_data,'train')
TestInput,TestTarget = Preprocessing.Start(test_data,'test')
TrainInput = np.squeeze(TrainInput)
TrainTarget = np.array(TrainTarget)
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']

class NeuralNetwork:
    def __init__(self,LENGHT_DATA):
        self.LENGHT_DATA = LENGHT_DATA
        print(self.LENGHT_DATA)
        self.INPUT_LAYERS = self.LENGHT_DATA
        self.HIDDEN_LAYERS = self.LENGHT_DATA
        self.OUTPUT_LAYERS = 9
        self.w1 = (np.random.rand(self.INPUT_LAYERS, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.random.normal(0.0, pow(self.INPUT_LAYERS, -0.5), (self.HIDDEN_LAYERS, self.INPUT_LAYERS))
        self.w2 = (np.random.rand(self.HIDDEN_LAYERS, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.random.normal(0.0, pow(self.HIDDEN_LAYERS, -0.5), (self.OUTPUT_LAYERS, self.HIDDEN_LAYERS))
        self.b1 = (np.random.rand(1, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.zeros((self.HIDDEN_LAYERS, 1))
        self.b2 = (np.random.rand(1, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.zeros((self.OUTPUT_LAYERS, 1))
        self.LossArray = []
        self.AccuracyArray = []
        self.Loss = 0
        self.Accuracy= 0
        self.LocalLoss = 0.5
        self.LocalAccuracy = 1.0

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(self,y):
        return y * (1 - y)
    
    def relu(self,x):
        return x * (x > 0)

    def deriv_relu(self,x):
        return (x >= 0).astype(float)

    def Loss(self,y,output):
        return 1/2 * (y - output) ** 2
    
    def Loss_deriv(self,y,output):
        return y - output

    def MSE(self,PredictedValue,TargetValue):
        Loss = ((TargetValue - PredictedValue) ** 2).mean()
        return Loss
    
    def CrossEntropy(self,PredictedValue,Target):
        return -np.log(PredictedValue[0, Target])

    def FeedForwardPropagation(self,Input,Target):
        self.InputLayer = self.sigmoid(np.dot(Input,self.w1) + self.b1)
        self.OutputLayer = self.sigmoid(np.dot(self.InputLayer,self.w2) + self.b2)
        self.Output = self.OutputLayer
        self.Error = self.CrossEntropy(self.Output,Target)
        if np.argmax(self.Output) == Target:
            self.Accuracy += 1
        return self.Output

    def BackwardPropagation(self,Input,Target):
        d1_w2 = learning_rate *\
                self.Loss_deriv(Target,self.Output) * \
                self.deriv_sigmoid(self.Output)
        d2_w2 = d1_w2 * self.InputLayer.reshape(-1,1)
        d1_w1 = learning_rate * \
                self.Loss_deriv(Target,self.Output) * \
                self.deriv_sigmoid(self.Output) @ \
                self.w2.T * \
                self.deriv_sigmoid(self.InputLayer)
        d2_w1 = np.matrix(d1_w1).T @ np.matrix(Input)
        self.w1 += d2_w1
        self.w2 += d2_w2
    
    def train(self,TrainInput,TrainTarget):
        # bar = trange(EPOCHS,leave=True)
        for epoch in  track(range(EPOCHS), description='[green]Processing data',style='bar.back',refresh_per_second=False):
            for Input,Target in zip(TrainInput,TrainTarget):
                # print('Target')
                # print(Target)
                OutputValue = self.FeedForwardPropagation(Input,Target)
                PredictedValue = np.argmax(OutputValue)
                self.BackwardPropagation(Input,Target)

                # bar.set_description(f'Epoch: {epoch}/{EPOCHS}; Loss: {self.Error}')
            self.LossArray.append(self.Error)
            self.AccuracyArray.append(self.Accuracy)
            # rich.do_step(epoch)
        # График ошибок
        plt.title('Train Loss')
        plt.plot(self.LossArray)
        plt.show()
        # График правильных предсказаний
        plt.title('Train Accuracy')
        plt.plot(self.AccuracyArray)
        plt.show()
    def save(self,PathParametrs = 'Artyom_NeuralAssistant.npz'):
        np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2)

    def open(self,PathParametrs = 'Artyom_NeuralAssistant.npz'):
        ParametrsFile = np.load(PathParametrs)
        self.w1 = ParametrsFile['arr_0']
        self.w2 = ParametrsFile['arr_1']
        self.b1 = ParametrsFile['arr_2']
        self.b2 = ParametrsFile['arr_3']

network = NeuralNetwork(len(TrainInput[0]))
network.train(TrainInput,TrainTarget)

network = NeuralNetwork(len(TestInput[0]))
network.train(TestInput,TestTarget)