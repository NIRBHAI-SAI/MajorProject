import csv


def data_creation():
    import cv2

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces == ():
            return None

        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
            cv2.imshow("FACE", cropped_face)

        return cropped_face

    cap = cv2.VideoCapture(0)
    count = 0
    with open("faces//Users.csv", "r") as Csvfile:
        read = csv.reader(Csvfile)
        users = next(read)
        # print(users)
    while 1:
        name = input("Enter user name: ").replace(" ","").lower()
        # print(name)
        if name in users:
            print(name + " is already present in data please enter a new name, ", end="")
        else:
            users.append(name)
            with open("faces//Users.csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(users)
            break


    file1 = open("id_text", "r")
    value = int(file1.read()) + 1
    file1.close()
    file_to_delete = open("id_text", 'w')
    file_to_delete.close()
    file1 = open("id_text", 'w')
    file1.write(str(value))

    ID = value

    while True:

        ret, frame = cap.read()

        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'D:\\sem 8\\majorproject 2\\currency_old\\faces\\user\\' + name + '.' + str(
                ID) + "." + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face is not found ")
            pass

        if cv2.waitKey(1) == 13 or count == 150:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")

data_creation()
