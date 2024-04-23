from vosk import Model, KaldiRecognizer
import pyaudio
from announce import *
from DT import date, time
from orbanalysis import currency_detection

model = Model(r"D:\sem 8\majorproject 2\currency_old\vosk-model-small-en-in-0.4")

recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()
#announcer("please say something...")
while(True):

    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        text = recognizer.Result()
        print(text[14:-3])
        captures_str = text[14:-3]
        if"stop" in captures_str:
            break
        elif "date" in captures_str:
            date()
        elif "time" in captures_str:
            time()
        elif "currency" in captures_str or "denomination" in captures_str or "note" in captures_str:
            currency_detection()
