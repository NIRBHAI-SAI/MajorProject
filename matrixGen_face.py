import os

import face_predictor
import helper
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sample = []
output = []
path = "faces//testData"
print("Collecting outputs")
for f in sorted(os.listdir(path)):
    print(f)
    name = ""
    for i in f:
        if i in ['0','1','2','3','4','5','6','7','8','9']:
            break
        name = name+i
    sample.append(name)
    output.append(face_predictor.facerecognition(f))

#
actual = np.array(sample)
predicted = np.array(output)
#
print(actual)
print(predicted)
#
cm = confusion_matrix(actual, predicted)

sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=['0', '10', '100', '101', '11', '20', '200', '21', '500', '51'],
            yticklabels=['0', '10', '100', '101', '11', '20', '200', '21', '500', '51'])
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Prediction', fontsize=13)
plt.title('Confusion Matrix', fontsize=17)
plt.show()

accuracy = accuracy_score(actual, predicted)
print("Accuracy   :", accuracy)