from AssistantCore import Core
import vosk
import sys
import sounddevice as sd
import queue
import json
import asyncio
import os
from loguru import logger

# Инициализация параметров для распознавания речи
SAMPLE_RATE = 16000
DEVICE = 1
# Инициализация параметров для логирования
ProjectDir = os.path.dirname(os.path.realpath(__file__))
logger.add(os.path.join(ProjectDir,'Logs/ArtyomAssistant.log'),format="{time} {level} {message}",level="INFO",rotation="200 MB",diagnose=True)

class ArtyomAssistant():
    def __init__(self):
        super(ArtyomAssistant).__init__()
        self.model = vosk.Model("model_small")
        self.queue = queue.Queue()

    def q_callback(self,indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.queue.put(bytes(indata))

    def SpeechRecognition(self):
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, device=DEVICE, dtype='int16',channels=1, callback=self.q_callback):
            self.Recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
            while True:
                data = self.queue.get()
                if self.Recognizer.AcceptWaveform(data):
                    answer = json.loads(self.Recognizer.Result())
                    # print(json.loads(self.Recognizer.Result())["text"])
                    if answer["text"]:
                        yield answer["text"]

    async def Start(self):
        for text in self.SpeechRecognition():
            print(text)

if __name__ == "__main__":
    AsyncioLoop = asyncio.get_event_loop()
    assistant = ArtyomAssistant()
    # assistant.Start()
    AsyncioLoop.run_until_complete(assistant.Start())
