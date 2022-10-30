from NeuralNetwork import NeuralNetwork
import os
import json
from PreprocessingText import PreprocessingDataset

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
TrainInput = Preprocessing.ToMatrix(TrainInput)
TrainTarget = Preprocessing.ToNumpyArray(TrainTarget)
TestInput = Preprocessing.ToMatrix(TestInput)
TestTarget = Preprocessing.ToNumpyArray(TestTarget)

def Train():
    network = NeuralNetwork(len(TrainInput[0]))
    network.train(TrainInput,TrainTarget)
    network = NeuralNetwork(len(TestInput[0]))
    network.train(TestInput,TestTarget)

if __name__ == '__main__':
    Train()