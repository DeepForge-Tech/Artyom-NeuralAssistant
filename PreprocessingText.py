import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import os
import json
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('stopwords')

test = {
    "Привет Артём, как дела, что делаешь?":0,
    "Музыка":"4",
    "Музыку":"4",
    "Прочти новости":"5",
    "Новости":"5"
}

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

    def ToMatrix(self,array):
        return np.squeeze(array)

    def ToNumpyArray(self,array):
        return np.array(array)

    def Start(self,PredictArray:list = [],Dictionary:dict = {},mode = 'train'):
        self.Mode = mode
        if self.Mode == 'train' or self.Mode == 'test':
            LocalDictionary = list(Dictionary.items())
            random.shuffle(LocalDictionary)
            self.Dictionary = dict(LocalDictionary)
            self.Dictionary = list(self.Dictionary.items())
            suggestions = []
            for Input, Target in self.Dictionary:
                Input = Input.lower()
                Input = remove_stopwords(Input)
                print(Input)
                Input = re.sub(r'\d+', '', Input)
                translator = str.maketrans('', '', string.punctuation)
                Input = Input.translate(translator)
                print(Input)
                suggestions.append(Input)
                if self.Mode == 'train':
                    self.TrainTarget.append(int(Target))
                elif self.Mode == 'test':
                    self.TestTarget.append(int(Target))
        elif self.Mode == 'predict':
            self.PredictArray = PredictArray
            suggestions = []
            for Input in self.PredictArray:
                Input = Input.lower()
                Input = re.sub(r'\d+', '', Input)
                translator = str.maketrans('', '', string.punctuation)
                Input = Input.translate(translator)
                suggestions.append(Input)
        vectorizer = TfidfVectorizer(max_features=1500, min_df=0, max_df=9)
        vectorizer = vectorizer.fit_transform(suggestions)
        VectorizedData = vectorizer.toarray()
        if self.Mode == 'train':
            self.TrainInput.append(VectorizedData)
        elif self.Mode == 'test':
            self.TestInput.append(VectorizedData)
        elif self.Mode == 'predict':
            self.PredictInput.append(VectorizedData)
        if self.Mode == 'train':
            return self.TrainInput,self.TrainTarget
        elif self.Mode == 'test':
            return self.TestInput,self.TestTarget
        elif self.Mode == 'predict':
            return self.PredictInput

# Read data and setup maps for integer encoding and decoding.
# ProjectDir = os.getcwd()
# file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
# DataFile = json.load(file)
# train_data = DataFile['train_dataset']
# test_data = DataFile['test_dataset']
# Preprocessing = PreprocessingDataset()
# TrainInput,TrainTarget = Preprocessing.Start(train_data,'train')
# TestInput,TestTarget = Preprocessing.Start(test_data,'test')
# clf = LogisticRegression(random_state=0).fit(TrainInput, TrainTarget)
# clf.predict(X[:2, :])
# preprocessing = PreProcessingDataset()
# preprocessing.Start(test)