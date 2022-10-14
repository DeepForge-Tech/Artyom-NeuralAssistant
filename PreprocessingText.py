import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import os
import json
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

test = {
    "Привет Артём, как дела, что делаешь?":0,
    "Музыка":"4",
    "Музыку":"4",
    "Прочти новости":"5",
    "Новости":"5"
}

class PreprocessingDataset:
    def __init__(self,):
        self.Dictionary = {}
        self.TrainInput = []
        self.TrainTarget = []
        self.TestInput = []
        self.TestTarget = []
        self.Mode = 'train'

    def Start(self,Dictionary:dict,mode = 'train'):
        self.Dictionary = Dictionary
        self.Mode = mode
        self.Dictionary = list(self.Dictionary.items())
        suggestions = []
        for Input, Target in self.Dictionary:
            Input = Input.lower()
            Input = re.sub(r'\d+', '', Input)
            translator = str.maketrans('', '', string.punctuation)
            Input = Input.translate(translator)
            # Input = " ".join(Input.split())
            # stop_words = set(stopwords.words("russian"))
            # word_tokens = word_tokenize(Input)
            # Input = [word for word in word_tokens if word not in stop_words]
            suggestions.append(Input)
            if self.Mode == 'train':
                self.TrainTarget.append(int(Target))
            elif self.Mode == 'test':
                self.TestTarget.append(int(Target))
        # print(suggestions)
        vectorizer = TfidfVectorizer(max_features=1500, min_df=0, max_df=2)
        vectorizer = vectorizer.fit_transform(suggestions)
        # vectorizer.transform(Input)
        VectorizedData = vectorizer.toarray()
        if self.Mode == 'train':
            self.TrainInput.append(VectorizedData)
        elif self.Mode == 'test':
            self.TestInput.append(VectorizedData)
        # print(vectorizer.toarray())
        if self.Mode == 'train':
            return self.TrainInput,self.TrainTarget
        elif self.Mode == 'test':
            return self.TestInput,self.TestTarget

# Read data and setup maps for integer encoding and decoding.
# ProjectDir = os.getcwd()
# file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
# DataFile = json.load(file)
# train_data = DataFile['train_dataset']
# test_data = DataFile['test_dataset']
# Preprocessing = PreProcessingDataset()
# TrainInput,TrainTarget = Preprocessing.Start(train_data,'train')
# TestInput,TestTarget = Preprocessing.Start(test_data,'test')
# clf = LogisticRegression(random_state=0).fit(TrainInput, TrainTarget)
# clf.predict(X[:2, :])
# preprocessing = PreProcessingDataset()
# preprocessing.Start(test)