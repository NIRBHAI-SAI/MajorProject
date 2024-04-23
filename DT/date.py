import datetime
import pytz
from tts.announce import announcer


def month(n):
    if (n == 1):
        return "january"
    if (n == 2):
        return "february"
    if (n == 3):
        return "march"
    if (n == 4):
        return "april"
    if (n == 5):
        return "may"
    if (n == 6):
        return "june"
    if (n == 7):
        return "july"
    if (n == 8):
        return "august"
    if (n == 9):
        return "september"
    if (n == 10):
        return "october"
    if (n == 11):
        return "november"
    if (n == 12):
        return "december"


def date():
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    d = str(current_time.day)
    m = current_time.month
    y = str(current_time.year)

    final_D = d + " " + month(m) + " of Year " + y

    announcer(final_D)



