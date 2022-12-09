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
from Preprocessing import PreprocessingDataset
from NeuralNetwork import NeuralNetwork
import geocoder
from pyowm import OWM
from num2words import num2words
import threading
from MusicManager import MusicManager
import platform
from loguru import logger
from datetime import date
from win10toast import ToastNotifier

# Инициализация параметров
ProjectDir = os.path.dirname(os.path.realpath(__file__))
UserDir = os.path.expanduser('~')
file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/Settings.json'),'r',encoding='utf-8')
DataFile = json.load(file)
CATEGORIES = DataFile['CATEGORIES']
CATEGORIES_TARGET = DataFile['CATEGORIES_TARGET']
file.close()
NAMES = ['Артём','Артемий','Артёша','Артемьюшка','Артя','Артюня','Артюха','Артюша','Артёмка','Артёмчик','Тёма']
file = open(os.path.join(ProjectDir,'NeuralNetworkSettings/ArtyomAnswers.json'),'r',encoding='utf-8')
ANSWERS  = json.load(file)
file.close()
language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000 # 48000
speaker = 'aidar' # aidar, baya, kseniya, xenia, random
put_accent = True
put_yo = True
device = torch.device('cpu') # cpu или gpu
MusicManager = MusicManager()
Preprocessing = PreprocessingDataset() 
network = NeuralNetwork(CATEGORIES,CATEGORIES_TARGET)
network.load()
owm = OWM('2221d769ed67828e858caaa3803161ea')
logger.add(os.path.join(ProjectDir,'Logs/ArtyomAssistant.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

class ArtyomAssistant:
    def __init__(self):
        self.Functions = {
            'communication':self.CommunicationCommand,'weather':self.WeatherCommand,
            'time':self.TimeCommand,'youtube':self.YoutubeCommand,
            'webbrowser':self.WebbrowserCommand,'hibernation':self.HibernationCommand,'reboot':self.RebootCommand,
            'shutdown':self.ShutdownCommand,'news':self.NewsCommand,
            'todo':self.TodoCommand,'calendar':self.CalendarCommand,
            'joikes':self.JoikesCommand,'exit':self.ExitCommand,
            'gratitude':self.GratitudeCommand
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

    def StopwatchCommand(self,command):
        pass

    def HibernationCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['hibernation']))
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def RebootCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['reboot']))
            os.system("shutdown -t 0 -r -f")
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def ShutdownCommand(self):
        if platform.system() == 'Windows':
            self.Tell(random.choice(ANSWERS['shutdown']))
            os.system('shutdown /p /f')
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def NewsCommand(self):
        pass

    def TodoCommand(self):
        self.Tell("Эта функция пока не доступна")

    def CalendarCommand(self):
        self.Tell("Эта функция пока не доступна")

    def JoikesCommand(self):
        pass

    def ExitCommand(self):
        self.Tell(random.choice(ANSWERS['exit']))

    def GratitudeCommand(self):
        self.Tell(random.choice(ANSWERS['gratitude']))

    def VSCode(self):
        if platform.system() == 'Windows':
            if os.path.exists(os.path.join(UserDir,'/AppData/Local/Programs/Microsoft VS Code/Code.exe')):
                self.Tell(random.choice(ANSWERS['vscode']))
                os.startfile(os.path.join(UserDir,'/AppData/Local/Programs/Microsoft VS Code/Code.exe'))
            else:
                self.Tell(random.choice(['У вас не установлена эта программа','Редактор кода не установлен на этом компьютере','Программа не установлена на этом компьютере']))
        elif platform.system() == 'Linux':
            self.Tell("Эта функция пока не доступна")
        elif platform.system() == 'Darwin':
            self.Tell("Эта функция пока не доступна")

    def CommandManager(self,PredictedValue):
        if PredictedValue == "don't_know":
            self.Tell(random.choice(ANSWERS["don't_know"]))
        else:
            operation = CATEGORIES[PredictedValue]

        if operation == 'music' or operation == 'off-music' or operation == 'pause-music' or operation == 'unpause-music':
            self.MusicCommand(operation)
        elif operation == 'stopwatch' or operation == 'off-stopwatch' or operation == 'pause-stopwatch' or 'unpause-stopwatch':
            self.StopwatchCommand(operation)
        else:
            self.Functions[operation]()

    def Start(self):
        for text in self.SpeechRecognition():
            print(text)
            for name in NAMES:
                if name.lower() in text and len(text.split()) > 1:
                    print('hello')
                    Input = text.replace(name.lower(),"")
                    Input = [text]
                    Input = Preprocessing.PreprocessingText(PredictArray = Input,mode = 'predict')
                    PredictedValue = network.predict(Input)
                    self.CommandManager(PredictedValue)
                    break
                elif name.lower() in text and len(text.split()) == 1:
                    self.Tell('Чем могу помочь?')
                    break
                
class TodoManager(ArtyomAssistant):
    def __init__(self) -> None:
        self.UpdateNotes()
        self.DefaultDate = date.today().strftime("%B %d, %Y")
        self.TodoNotes = {"notes":{}}

    def UpdateNotes(self):
        if os.path.exists(os.path.join(ProjectDir,'AssistantSettings/TodoNotes.json')):
            file = open('AssistantSettings/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()
        else:
            file = open('AssistantSettings/TodoNotes.json','w',encoding='utf-8')
            json.dump({"notes":{}},file,ensure_ascii=False,sort_keys=True, indent=2)
            file.close()
            file = open('AssistantSettings/TodoNotes.json','r',encoding='utf-8')
            DataFile = json.load(file)
            self.TodoNotes = DataFile
            file.close()

    def UpdateDate(self):
        self.DefaultDate = date.today().strftime("%B %d, %Y")

    def SaveNotes(self):
        file = open('AssistantSettings/TodoNotes.json','w',encoding='utf-8')
        json.dump(self.TodoNotes,file,ensure_ascii=False,sort_keys=True, indent=2)
        file.close()

    def Notification(self,title,message):
        if platform.system() == 'Windows':
            ToastNotifier().show_toast(title=title,msg=message,duration=5)

    def CreateNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),local_time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        self.UpdateNotes()
        # print(self.TodoNotes)
        if date in self.TodoNotes["notes"]:
            if local_time in self.TodoNotes["notes"][date]:
                if not text in self.TodoNotes["notes"][date][local_time]:
                    self.TodoNotes["notes"][date][local_time].append(text)
                    print(self.TodoNotes)
        else:
            self.TodoNotes.update(
                {
                    "notes":{
                            date:{
                                    local_time:[text]
                                }
                            }
                }
            )
        self.SaveNotes()
        print(self.TodoNotes)

    def RemoveNote(self,text:str,date:str = date.today().strftime("%B %d, %Y"),local_time:str = (f"{time.strftime('%H')}:{time.strftime('%M')}")):
        self.UpdateNotes()
        if date in self.TodoNotes["notes"]:
            if local_time in self.TodoNotes["notes"][date]:
                if text in self.TodoNotes["notes"][date][local_time]:
                    self.TodoNotes["notes"][date][local_time].remove(text)
                    print("Hello")
                    self.SaveNotes()
        # else:
        #     self.Tell(random.choice(["Заметка отсутствует","Я не нашёл такой заметки","Заметка не найдена","Такой заметки нет","Такой заметки не существует"]))
        

    def CheckNote(self):
        self.UpdateNotes()
        while True:
            self.UpdateDate()
            local_time = f"{time.strftime('%H')}:{time.strftime('%M')}"
            if self.DefaultDate in self.TodoNotes["notes"]:
                if local_time in self.TodoNotes["notes"][self.DefaultDate]:
                    for note in self.TodoNotes["notes"][self.DefaultDate][local_time]:
                        print(note)
                        self.Notification("Заметка",note)
                        if len(self.TodoNotes["notes"][self.DefaultDate][local_time]) >= 2:
                            self.TodoNotes["notes"][self.DefaultDate][local_time].remove(note)
                            self.SaveNotes()
                        elif len(self.TodoNotes["notes"][self.DefaultDate][local_time]) == 1:
                            self.TodoNotes["notes"][self.DefaultDate].pop(local_time)
                            self.SaveNotes()



if __name__ == '__main__':
    todo_manager = TodoManager()
    todo_manager.CreateNote("123456789",date = todo_manager.DefaultDate,local_time = "22:11")
    todo_manager.CreateNote("Hello,BRO :)",date = todo_manager.DefaultDate,local_time = "22:12")
    # todo_manager.RemoveNote("123456789",date = todo_manager.DefaultDate,local_time = "11:51")
    todo_manager.CheckNote()
    # Artyom = ArtyomAssistant()
    # Artyom.Start()
