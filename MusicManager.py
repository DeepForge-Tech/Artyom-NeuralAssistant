# Импортирование необходимых модулей
import pygame
import os

# Инициализация параметров
pygame.mixer.init(96000, -16, 2, 8192)
pygame.mixer.music.set_volume(2.0)
HomeDir = os.path.expanduser("~")
MusicFormats = ['.mp3','.flac','.ogg','.aac','.wav','.aiff','.dsd','.mqa','.wma','.alac','.pcm']
ProjectDir = os.path.dirname(os.path.realpath(__file__))
MusicPath = HomeDir + r'\Music'
MusicFiles = []

class MusicManager:
    def __init__(self):
        self.MusicNumber = 0
        self.PausedMusic = False
        self.PlayingMusic = False

    def StopMusic(self):
        pass

    def PauseMusic(self):
        if self.PausedMusic == False and self.PlayingMusic == True:
            pygame.mixer.music.pause

    def UnpauseMusic(self):
        pass

    def PlayMusic(self):
        for dir, subdir, files in os.walk(MusicPath):
            for file in files:
                print(os.path.join(dir, file))
                file = os.path.normpath( os.path.join(dir, file))
                format = os.path.splitext(os.path.join(dir, file))[1]
                for MusicFormat in MusicFormats:
                    if format == MusicFormat:
                        MusicFiles.append(file)
        for Music in MusicFiles:
            pygame.mixer.music.load(Music)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pos = pygame.mixer.music.get_pos()/ 1000
            self.MusicNumber += 1
