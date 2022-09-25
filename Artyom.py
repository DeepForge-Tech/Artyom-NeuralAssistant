import time
import random
import os
import json
import fuzzywuzzy
import torch
import webbrowser
import sounddevice as sd
import sys
import pyaudio
from vosk import Model, KaldiRecognizer
import NeuralNetwork

language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu

CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes']

class ArtyomAssistant:
    def __init__(self):
        self.RecognitionModel = Model('model')
        self.Recognition = KaldiRecognizer(self.RecognitionModel,16000)
        self.RecognitionAudio = pyaudio.PyAudio()
        self.stream = self.RecognitionAudio.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=8000)
        self.stream.start_stream()

        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
        self.model.to(device)
    
    def Tell(self,text: str):
        audio = self.model.apply_tts(text=text+"..",
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)

        sd.play(audio, sample_rate * 1.05)
        time.sleep((len(audio) / sample_rate) + 0.5)
        sd.stop()

    def SpeechRecognition(self):
        while True:
            data = self.stream.read(8000,exception_on_overflow=False)
            if (self.Recognition.AcceptWaveform(data)) and (len(data) > 0):
                answer = json.loads(self.Recognition.Result())
                if answer['text']:
                    yield answer['text']

    def start(self):
        for text in self.SpeechRecognition():
            print(text)
            vocab = list(set([w for w in text.split(' ')]))
            vocab_size = len(vocab)
            word_to_idx = { w: i for i, w in enumerate(vocab) }
            idx_to_word = { i: w for i, w in enumerate(vocab) }
            network = NeuralNetwork.NeuralNetwork(vocab_size,len(CATEGORIES),CATEGORIES,word_to_idx,idx_to_word)
            network.load()
            PredictedValue = network.predict(text)
            print(PredictedValue)

if __name__ == '__main__':
    Artyom = ArtyomAssistant()
    Artyom.start()