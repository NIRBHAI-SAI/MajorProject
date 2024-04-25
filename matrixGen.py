import os
import helper
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sample = ['0', '0', '0', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100']
output = ['0', '0', '0', '100', '100', '100', '100', '100', '100', '100', '100', '100', '100']
path = "samples"
print("Collecting outputs")
for f in sorted(os.listdir(path)):
    currency = ""
    for i in f:
        if i == ' ':
            break
        currency = currency + i
    sample.append(currency)
    r = helper.currency_detection(os.path.join(path, f))
    output.append(r)

actual = np.array(sample)
predicted = np.array(output)

print(actual)
print(predicted)

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