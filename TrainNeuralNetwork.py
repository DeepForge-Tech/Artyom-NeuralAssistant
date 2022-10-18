from NeuralNetwork import NeuralNetwork
import os
import json
import numpy
from PreprocessingText import PreprocessingDataset
import numpy as np

# Initialization parametrs
# Read data and setup maps for integer encoding and decoding.
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']
Preprocessing = PreprocessingDataset()
TrainInput,TrainTarget = Preprocessing.Start(Dictionary = train_data,mode = 'train')
TestInput,TestTarget = Preprocessing.Start(Dictionary = test_data,mode = 'test')
TrainInput = np.squeeze(TrainInput)
TrainTarget = np.array(TrainTarget)
TestInput = np.squeeze(TestInput)
TestTarget = np.array(TestTarget)

def Train():
    network = NeuralNetwork(len(TrainInput[0]))
    network.train(TrainInput,TrainTarget)
    network = NeuralNetwork(len(TestInput[0]))
    network.train(TestInput,TestTarget)

if __name__ == '__main__':
    Train()