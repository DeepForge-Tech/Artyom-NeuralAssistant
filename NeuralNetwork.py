import numpy as np
import matplotlib.pyplot as plt
import os
from PreprocessingText import PreprocessingDataset
from rich.progress import track
import mplcyberpunk

plt.style.use("cyberpunk")
EPOCHS = 200000
learning_rate = 0.001
ProjectDir = os.getcwd()
Preprocessing = PreprocessingDataset()
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']

class NeuralNetwork:
    def __init__(self,LENGHT_DATA):
        self.LENGHT_DATA = LENGHT_DATA
        self.INPUT_LAYERS = self.LENGHT_DATA
        self.HIDDEN_LAYERS = self.LENGHT_DATA
        self.OUTPUT_LAYERS = len(CATEGORIES)
        self.w1 = np.random.randn(self.INPUT_LAYERS,self.HIDDEN_LAYERS) / 1000#(np.random.rand(self.INPUT_LAYERS, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.random.normal(0.0, pow(self.INPUT_LAYERS, -0.5), (self.HIDDEN_LAYERS, self.INPUT_LAYERS))
        self.w2 = np.random.randn(self.HIDDEN_LAYERS,self.OUTPUT_LAYERS) / 1000#(np.random.rand(self.HIDDEN_LAYERS, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.random.normal(0.0, pow(self.HIDDEN_LAYERS, -0.5), (self.OUTPUT_LAYERS, self.HIDDEN_LAYERS))
        self.b1 = (np.random.rand(1, self.HIDDEN_LAYERS) - 0.5) * 2 * np.sqrt(1/self.INPUT_LAYERS)#np.zeros((self.HIDDEN_LAYERS, 1))
        self.b2 = (np.random.rand(1, self.OUTPUT_LAYERS) - 0.5) * 2 * np.sqrt(1/self.HIDDEN_LAYERS)#np.zeros((self.OUTPUT_LAYERS, 1))
        self.LossArray = []
        self.Loss = 0
        self.LocalLoss = 0.5

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

    def softmax(self,xs):
        # Applies the Softmax Function to the input array.
        return np.exp(xs) / sum(np.exp(xs))

    def FeedForwardPropagation(self,Input):
        self.InputLayer = self.sigmoid(np.dot(Input,self.w1) + self.b1)
        self.OutputLayer = self.sigmoid(np.dot(self.InputLayer,self.w2) + self.b2)
        self.Output = self.OutputLayer
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
        for epoch in  track(range(EPOCHS), description='[green]Training model'):
            self.Error = 0
            for Input,Target in zip(TrainInput,TrainTarget):
                OutputValue = self.FeedForwardPropagation(Input)
                self.BackwardPropagation(Input,Target)
                self.Error = self.MSE(self.Output,Target)
                if float(self.Error) <= self.LocalLoss and np.argmax(self.Output) == Target:
                    self.LocalLoss = self.Error
                    # print('Best model')
                    self.save()
            self.LossArray.append(self.Error)
        # График ошибок
        plt.title('Train Loss')
        plt.plot(self.LossArray)
        plt.savefig(os.path.join(ProjectDir,'Graphics','Loss.png'))
        # plt.show()
    
    def predict(self,Input):
        OutputValue = self.FeedForwardPropagation(Input)
        PredictedValue = np.argmax(OutputValue)
        print(PredictedValue)
        return PredictedValue

    def save(self,PathParametrs = os.path.join(ProjectDir,'Models','Artyom_NeuralAssistant.npz')):
        np.savez_compressed(PathParametrs, self.w1,self.w2,self.b1,self.b2)

    def open(self,PathParametrs = os.path.join(ProjectDir,'Models','Artyom_NeuralAssistant.npz')):
        ParametrsFile = np.load(PathParametrs)
        for n in range(int(self.HIDDEN_LAYERS)):
            for i in range(self.HIDDEN_LAYERS):
                if (0 <= n) and (n < len(self.w1)):
                    if (0 <= i) and (i < len(self.w1[n])):
                        self.w1[n][i] = ParametrsFile['arr_0'][n][i]
                if (0 <= n) and (n < len(self.w2)):
                    if (0 <= i) and (i < len(self.w2[n])):
                        self.w2[n][i] = ParametrsFile['arr_1'][n][i]
                if (0 <= n) and (n < len(self.b1)):
                    if (0 <= i) and (i < len(self.b1[n])):
                        self.b1[n][i] = ParametrsFile['arr_2'][n][i]
                if (0 <= n) and (n < len(self.b2)):
                    if (0 <= i) and (i < len(self.b2[n])):
                        self.b2[n][i] = ParametrsFile['arr_3'][n][i]
        print('W1')
        print(self.w1)
        print('Parametrs W1')
        print(ParametrsFile['arr_0'])
    
    

def TestPredict():
    while True:
        command = input('>>>')
        if command == 'exit':
            break
        else:
            Test = [command]
            Test = Preprocessing.Start(PredictArray=Test,mode = 'predict')
            Test = Preprocessing.ToMatrix(Test)
            network = NeuralNetwork(len(Test))
            network.open()
            network.predict(Test)

if __name__ == '__main__':
    TestPredict()
