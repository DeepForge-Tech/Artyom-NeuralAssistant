# Импортирование необходмых модулей
import time
import random
import os
import json
import torch
import webbrowser
import sounddevice as sd
import sys
import pyaudio
from vosk import Model, KaldiRecognizer
from PreprocessingText import PreprocessingDataset
from NeuralNetwork import NeuralNetwork
import geocoder
from pyowm import OWM
from num2words import num2words
import threading
from MusicManager import MusicManager

# Инициализация параметров
language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
MusicManager = MusicManager()
Preprocessing = PreprocessingDataset() 
owm = OWM('2221d769ed67828e858caaa3803161ea')
ProjectDir = os.path.dirname(os.path.realpath(__file__))
NAMES = ['Артём','Артемий','Артёша','Артемьюшка','Артя','Артюня','Артюха','Артюша','Артёмка','Артёмчик','Тёма']
CATEGORIES = ['communication','weather','youtube','webbrowser','music','news','todo','calendar','joikes','exit','time','gratitude','stopwatch','off-stopwatch','pause-stopwatch','unpause-stopwatch','off-music','timer','off-timer','pause-timer','unpause-timer','turn-up-music','turn-down-music','pause-music','unpause-music','shutdown','reboot','hibernation']
file = open('ArtyomAnswers.json','r',encoding='utf-8')
ANSWERS  = json.load(file)
file.close()

class ArtyomAssistant:
    def __init__(self):
        self.Functions = {
            'communication':self.CommunicationCommand,'weather':self.WeatherCommand,
            'time':self.TimeCommand,"music":self.MusicCommand,
            'youtube':self.YoutubeCommand,'webbrowser':self.WebbrowserCommand
        }
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
    
    def CommunicationCommand(self):
        self.Tell(random.choice(ANSWERS['communication']))
    
    def WeatherCommand(self,WithInternet:bool=False):
        geolocation = geocoder.ip('me')
        coordinates = geolocation.latlng
        location = str(location.address.split(',')[4]).lower()
        mgr = owm.weather_manager()
        one_call = mgr.one_call(lat=coordinates[0], lon=coordinates[1])
        temp = one_call.current.temperature('celsius')['temp']  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}
        print(temp)
        self.Tell('Сейчас {} градусов по цельсию'.format(num2words(int(temp), lang='ru')))
    
    def TimeCommand(self):
        hours = num2words(int(time.strftime('%H')), lang='ru')
        minutes = num2words(int(time.strftime('%M')), lang='ru')
        self.Tell(f'Сейчас {hours} {minutes}')

    def MusicCommand(self,command):
        if command == 'music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicThread = threading.Thread(target = MusicManager.PlayMusic)
                MusicThread.start()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                self.Tell(random.choice(ANSWERS['play-music']))
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == False:
                MusicManager.UnpauseMusic()

        elif command == 'off-music':
            if MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                MusicManager.StopMusic()
            elif MusicManager.PlayingMusic == False and MusicManager.StoppedMusic == True:
                self.Tell(random.choice(ANSWERS['off-music']))

        elif command == 'pause-music':
            if MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True and MusicManager.StoppedMusic == False:
                MusicManager.PauseMusic()
            elif MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == False:
                self.Tell(random.choice(ANSWERS['pause-music']))
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == False  and MusicManager.StoppedMusic == True:
                self.Tell('Музыка выключена.')
                # self.Tell('Включить её?')

        elif command == 'unpause-music':
            if MusicManager.PausedMusic == True and MusicManager.PlayingMusic == False:
                MusicManager.UnpauseMusic()
            elif MusicManager.PausedMusic == False and MusicManager.PlayingMusic == True:
                self.Tell(random.choice(ANSWERS['unpause-music']))

    def YoutubeCommand(self):
        self.Tell(random.choice(ANSWERS['youtube']))
        webbrowser.open_new_tab('https://youtube.com')
    
    def WebbrowserCommand(self):
        self.Tell(random.choice(ANSWERS['webbrowser']))
        webbrowser.open_new_tab('https://google.com')

    def CommandManager(self,PredictedValue):
        operation = CATEGORIES[PredictedValue]
        if operation == 'music' or operation == 'off-music' or operation == 'pause-music' or operation == 'unpause-music':
            self.Functions[operation](operation)
        else:
            self.Functions[operation]()




    def Start(self):
        for text in self.SpeechRecognition():
            print(text)
            for name in NAMES:
                if name.lower() in text and len(text.split()) > 1:
                    print('hello')
                    Input = text.replace(name.lower(),"")
                    # Input = [text]
                    # Input = Preprocessing.Start(PredictArray=Input,mode = 'predict')
                    # Input = Preprocessing.ToMatrix(Input)
                    # network = NeuralNetwork(len(Input))
                    # network.open()
                    # PredictedValue = network.predict(Input)
                    self.CommandManager(3)
                    break
                elif name.lower() in text and len(text.split()) == 1:
                    self.Tell('Чем могу помочь?')
                    break
                
if __name__ == '__main__':
    Artyom = ArtyomAssistant()
    Artyom.Start()