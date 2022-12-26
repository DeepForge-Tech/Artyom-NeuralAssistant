import threading 
import time

class Timer:
    def __init__(self):
        self.TimeStarted = False
        self.TimeStopped = False
        self.TimeUnpaused = False
        self.TimePaused = False
        self.Time = 0
        self.TimeThread = None

    def Main(self,hours,minutes,seconds):
        self.TimeThread = threading.Thread(target = self.Start,args=(hours,minutes,seconds)).start()

    def Start(self,hours,minutes,seconds):
        if self.TimeStarted == False  and self.TimeUnpaused == False and self.TimePaused == False:
            self.TimeStarted = True
            self.TimeStopped = False
            self.TimePaused = False
            self.TimeUnpaused = False

        self.Time = int((hours * 3600) + (minutes * 60) + seconds)
        while self.Time != 0:
            if self.TimeStopped == True:
                break
            elif self.TimePaused:
                continue
            else:
                self.Time -= 1
                time.sleep(1)
                print(self.Time)

    def Stop(self):
        if self.TimeStarted == True and self.TimeStopped == False:
            self.TimeStopped = True
            self.TimeStarted = True
            self.TimeUnpaused = False
            self.TimePaused = False

    def Pause(self):
        if self.TimeStarted == True and self.TimeStopped == False and self.TimePaused == False:
            self.TimePaused = True
            self.TimeStopped = False
            self.TimeStarted = True
            self.TimeUnpaused = False
    
    def Unpause(self):
        if self.TimeStarted == True and self.TimeStopped == False and self.TimeUnpaused == False and self.TimePaused == True:
            self.TimePaused = False
            self.TimeStarted = True
            self.TimeStopped = False
            self.TimeUnpaused = True

if __name__ == "__main__":
    timer = Timer()
    timer.Main(0,1,2)
    time.sleep(5)
    timer.Pause()
    time.sleep(2)
    timer.Unpause()
    time.sleep(2)
    timer.Stop()
