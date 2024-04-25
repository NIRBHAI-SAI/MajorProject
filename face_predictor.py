import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def facerecognition():

    data_path = 'D:\\sem 8\\majorproject 2\\currency_old\\faces\\user\\'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []
    label_name = {}

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        ID = int(os.path.split(image_path)[-1].split('.')[1])
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        label_name[ID] = os.path.split(image_path)[-1].split('.')[0]
        Labels.append(ID)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained successfully")

    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_detector(img, size=0.5):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces == ():
            return img, []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))
        return img, roi

    sample = []
    output = []
    path = "faces//testData"
    print("Collecting outputs")
    for f in sorted(os.listdir(path)):
        # print(f)
        frame = cv2.imread(os.path.join(path, f))
        # cv2.imshow("input", frame)
        # cv2.waitKey(0)
        name = ""
        for i in f:
            if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                break
            name = name + i
        sample.append(name)

        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            results = model.predict(face)

            confidence = int(100 * (1 - (results[1]) / 400))

            if confidence >= 90:
                output.append(label_name[results[0]])

            else:
                output.append('unknown')

        except:
            output.append('noface')
    # print(output)
    # print(sample)
    return np.array(sample), np.array(sample), list(label_name.values())


predicted, actual, labels = facerecognition()
labels.append("noface")
labels.append("unknown")
labels.sort()
print(labels)
print(actual)
print(predicted)

# cm = confusion_matrix(actual, predicted)
#
# sns.heatmap(cm,
#             annot=True,
#             fmt='g',
#             xticklabels=labels,
#             yticklabels=labels)
# plt.ylabel('Actual', fontsize=13)
# plt.xlabel('Prediction', fontsize=13)
# plt.title('Confusion Matrix', fontsize=17)
# plt.show()

accuracy = accuracy_score(actual, predicted)
print("Accuracy   :", accuracy)

