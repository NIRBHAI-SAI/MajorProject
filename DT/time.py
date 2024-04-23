import datetime
import pytz
from tts.announce import announcer

def tiMe():

    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    H = current_time.hour
    M = current_time.minute
    sub = " A M "
    if H > 12:
        H = H - 12
        sub = " P M "
    announcer(str(H) + " " + str(M) + " " + sub)