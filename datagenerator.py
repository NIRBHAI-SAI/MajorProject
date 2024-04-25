import csv

import cv2

cap = cv2.VideoCapture(0)
with open("faces//Users.csv", "r") as Csvfile:
    read = csv.reader(Csvfile)
    users = next(read)

while 1:
    name = input("enter Name: ").replace(" ", "").lower()
    if name not in users:
        print("\n"+ name + " is not present in Data Model\n \nDo you want to Consider it as 'UNKNOWN'\n 1)yes  2)No ")
        Choice = int(input("\n Enter a Choice:"))
        if Choice == 1:
            name = "unknown"
            break
        elif Choice == 2:
            print("Enter a new name,", end="")
        else:
            print("invalid Choice, enter name again,", end="")
    else:
        break

capt = cv2.VideoCapture(0)
count = 0
while count < 15:
    count = count + 1
    print(count)
    if capt.isOpened():
        ret, frame = capt.read()
    else:
        ret = False

    while ret:
        print("Capturing image...")
        ret, frame = capt.read()
        path = "D:\\sem 8\\majorproject 2\\currency_old\\faces\\testData\\"
        Str = name + str(count) + ".jpg"
        cv2.imwrite(path + Str, frame)
        break
capt.release()
cv2.destroyAllWindows()
