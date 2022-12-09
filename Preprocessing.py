# Импортирование необходмых библиотек в целях подготовки датасета для нейросети
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import random
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from rich.progress import track

# Подготовка датасета
ProjectDir = os.getcwd()
if os.path.exists(os.path.join(ProjectDir,'Datasets/ArtyomDataset.json')):
    file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
    DataFile = json.load(file)
    dataset = DataFile['dataset']
    file.close()
else:
    raise RuntimeError

if os.path.exists(os.path.join(ProjectDir,'NeuralNetworkSettings/Settings.json')):
    file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/Settings.json'),'r',encoding='utf-8')
    DataFile = json.load(file)
    CATEGORIES = DataFile['CATEGORIES']
    CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
    file.close()
else:
    raise RuntimeError


class PreprocessingDataset:
    def __init__(self):
        self.Dictionary = {}
        self.TrainInput = []
        self.TrainTarget = []
        self.TestInput = []
        self.TestTarget = []
        self.PredictInput = []
        self.PredictArray = []
        self.Mode = 'train'
        self.x = []
        self.y = []

    def ToMatrix(self,array):
        return np.squeeze(array)

    def ToNumpyArray(self,array):
        return np.array(array)

    def PreprocessingAudio(self,PathAudio:str,mode:str = 'train'):
        self.Mode = mode
        self.PathAudio = PathAudio
        if self.Mode == 'train' or self.Mode == 'test':
            self.DatasetFiles = list(os.walk(self.PathAudio))
            for (root,dirs,files) in track(os.walk(self.PathAudio,topdown=True),description='[green]Preprocessing'):
                for file in files[:10]:
                    if file.endswith('.wav'):
                        self.AudioFile = os.path.join(root,file)
                        audio,sample_rate = librosa.load(self.AudioFile,res_type='kaiser_fast')
                        mfccs = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
                        mfccs = np.mean(mfccs.T,axis=0)
                        self.x.append(mfccs)
                    elif file.endswith('.txt'):
                        file = open(os.path.join(root,file),'r+',encoding="utf-8")
                        DataFile = file.read()
                        self.y.append(DataFile)
                        file.close()
                # print (root)
                # print("\n" + "\n" + "\n")
                # print (dirs)
                # print("\n" + "\n" + "\n")
                # print (files)
                # print("\n" + "\n" + "\n")
                # print ('--------------------------------')
            InputDatasetFile = open("Datasets/SpeechInputDataset.json", "w", encoding ='utf-8')
            json.dump(self.y, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
            labelencoder=LabelEncoder()
            labelencoder = labelencoder.fit_transform(self.y)
            self.TrainTarget = self.ToNumpyArray(labelencoder)
            self.TrainInput = self.ToMatrix(self.x)
            print(len(self.y[0]))
            return self.TrainInput,self.TrainTarget
            
        elif self.Mode == 'predcit':
            InputDatasetFile = open("Datasets/SpeechInputDataset.json", "r", encoding ='utf8')
            DataFile = json.load(InputDatasetFile)
            InputDatasetFile.close()
            labelencoder=LabelEncoder()
            labelencoder = labelencoder.fit_transform(DataFile)
            self.AudioFile = self.PathAudio
            audio,sample_rate = librosa.load(self.AudioFile,res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
            mfccs = np.mean(mfccs.T,axis=0)
            self.PredictInput = self.ToMatrix(labelencoder.transform(mfccs))
            return self.PredictInput
    def PreprocessingText(self,PredictArray:list = [],Dictionary:dict = {},mode = 'train'):
        self.Mode = mode
        if self.Mode == 'train' or self.Mode == 'test':
            self.Dictionary = list(Dictionary.items())
            random.shuffle(self.Dictionary)
            self.Dictionary = dict(self.Dictionary)
            for intent in track(self.Dictionary,description='[green]Preprocessing'):
                for questions in Dictionary[intent]['questions']:
                    self.x.append(questions)
                    self.y.append(intent)
            if self.Mode == 'train':
                for target in self.y:
                    self.TrainTarget.append(CATEGORIES[target])
            elif self.Mode == 'test':
                for target in self.y:
                    self.TestTarget.append(CATEGORIES[target])
            vectorizer = TfidfVectorizer()
            vectorizer = vectorizer.fit_transform(self.x)
            VectorizedData = vectorizer.toarray()
            InputDatasetFile = open("Datasets/InputDataset.json", "w", encoding ='utf8')
            json.dump(self.x, InputDatasetFile,ensure_ascii=False,sort_keys=True, indent=2)
            InputDatasetFile.close()
            if self.Mode == 'train':
                self.TrainInput = self.ToMatrix(VectorizedData)
                return self.TrainInput,self.TrainTarget
            elif self.Mode == 'test':
                self.TestInput = VectorizedData
                return self.TestInput,self.TestTarget

        elif self.Mode == 'predict':
            self.PredictArray = PredictArray
            InputDatasetFile = open("Datasets/InputDataset.json", "r", encoding ='utf8')
            DataFile = json.load(InputDatasetFile)
            InputDatasetFile.close()
            vectorizer = TfidfVectorizer()
            vectorizer.fit_transform(DataFile)
            self.PredictInput = self.ToMatrix(vectorizer.transform(self.PredictArray).toarray())
            return self.PredictInput

# PreprocessingDataset().PreprocessingAudio(PathAudio="C:/Users/Blackflame576/Documents/Blackflame576/DigitalBit/Artyom-NeuralAssistant/Datasets/SpeechDataset/")