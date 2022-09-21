from NeuralNetwork import*
import time
import random
import os
import json
import fuzzywuzzy
import torch
import webbrowser
import sounddevice as sd
import vosk
import sys
import queue

language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu


class Artyom:
    def __init__(self):
        self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)
        self.model.to(device)
    
    def tell(self,text: str):
        audio = self.model.apply_tts(text=text+"..",
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)

        sd.play(audio, sample_rate * 1.05)
        time.sleep((len(audio) / sample_rate) + 0.5)
        sd.stop()

    def start(self):
        while True:
            pass

if __name__ == '__main__':
    Artyom = Artyom()
    Artyom.start()