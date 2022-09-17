import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
import nltk
import numpy as np

# Загрузка данных для удаления стоп-слов и пунктуации
nltk.download('stopwords')
nltk.download('punkt')

stemmer = PorterStemmer()

class Preprocessing():
    def __init__(self,*args,**kwargs):
        self.Dataset = []
    
    def PreprocessingText(self,Array):
        self.Dataset = Array
        for i in range(len(self.Dataset)):
            self.Dataset[i] = self.Dataset[i].lower()
            self.Dataset[i] = re.sub(r'\d+', '', self.Dataset[i])
            translator = str.maketrans('', '', string.punctuation)
            self.Dataset[i] = self.Dataset[i].translate(translator)
            self.Dataset[i] = " ".join(self.Dataset[i].split())
            stop_words = set(stopwords.words("russian"))
            word_tokens = word_tokenize(self.Dataset[i])
            # self.Dataset[i] = [word for word in word_tokens if word not in stop_words]
            # word_tokens = word_tokenize(self.Dataset[i])
            self.Dataset[i] = [stemmer.stem(word) for word in word_tokens]
            questions = self.Dataset[i]

            ### Universal list of colors
            total_questions = self.Dataset[i]

            ### map each color to an integer
            mapping = {}
            for x in range(len(total_questions)):
                mapping[total_questions[x]] = x

            one_hot_encode = []

            for c in questions:
                arr = list(np.zeros(len(total_questions), dtype = int))
                arr[mapping[c]] = 1
                one_hot_encode.append(arr)
            self.Dataset[i] = one_hot_encode
        return np.array(list(self.Dataset))

# print(Preprocessing().PreprocessingText(['наука о данных использует научные методы, алгоритмы и многие типы процессов']))