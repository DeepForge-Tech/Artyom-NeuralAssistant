from NeuralNetwork import *
import os
import json
import random

# Переменные
EPOCHS = 50000
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']
train_data = {}
test_data = {}

# Read data and setup maps for integer encoding and decoding.
ProjectDir = os.getcwd()
file = open('Datasets/ArtyomDataset.json','r',encoding='utf-8')
DataFile = json.load(file)
file.close()
train_data = DataFile['train_dataset']
test_data = DataFile['test_dataset']

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

network = NeuralNetwork(vocab_size,len(CATEGORIES),CATEGORIES,word_to_idx,idx_to_word)
network.train(EPOCHS,train_data,True)
network.train(EPOCHS,train_data,False)

vocab = list(set([w for text in test_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
 
print('%d unique words found' % vocab_size)
# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

network = NeuralNetwork(vocab_size,len(CATEGORIES),CATEGORIES,word_to_idx,idx_to_word)
network.train(EPOCHS,test_data,True)
network.train(EPOCHS,test_data,False)

# file = open('Datasets/ArtyomNameDataset.json','r',encoding='utf-8')
# DataFile = json.load(file)
# train_data = DataFile['train_dataset']
# test_data = DataFile['test_dataset']



# vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
# vocab_size = len(vocab)
 
# print('%d unique words found' % vocab_size)
# # Assign indices to each word.
# word_to_idx = { w: i for i, w in enumerate(vocab) }
# idx_to_word = { i: w for i, w in enumerate(vocab) }

# network = NeuralNetwork(vocab_size,len(TARGET_NAME),TARGET_NAME,word_to_idx,idx_to_word)
# network.train(EPOCHS,train_data,True)
# network.train(EPOCHS,train_data,False)

# vocab = list(set([w for text in test_data.keys() for w in text.split(' ')]))
# vocab_size = len(vocab)
 
# print('%d unique words found' % vocab_size)
# # Assign indices to each word.
# word_to_idx = { w: i for i, w in enumerate(vocab) }
# idx_to_word = { i: w for i, w in enumerate(vocab) }

# network = NeuralNetwork(vocab_size,len(TARGET_NAME),TARGET_NAME,word_to_idx,idx_to_word)
# network.train(train_data,True)
# network.train(train_data,False)