import speech_recognition as sr
from DT.date import *
from DT.time import tiMe
from orbanalysis import *
from execution_code import facerecognition


def audio_reader():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        announcer("please say something...")
        audio = r.listen(source)
        try:
            final_str = "You have said: \n" + r.recognize_google(audio)
            print(final_str)
            announcer(final_str)
            return final_str

        except:
            announcer("please try again")
            print("ERROR :")


captures_str = audio_reader()
if captures_str is None:
    exit()
if "date" in captures_str:
    date()
if "time" in captures_str:
    tiMe()
if "currency" in captures_str or "denomination" in captures_str or "note" in captures_str:
    currency_detection()
if "face" in captures_str or "who" in captures_str:
    facerecognition()

