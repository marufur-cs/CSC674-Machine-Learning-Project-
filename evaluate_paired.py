# For same class 0, for different class 1
import pickle
import numpy as np
import pandas as pd
from lenet import LeNet
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model =   LeNet(numChannels=2, classes=1)
model = pickle.load(open('paired_trained_model.pkl', 'rb'))

test_data = pd.read_csv('test_data.csv',header=None).to_numpy()
x0_ground = pd.read_csv('x0_data.csv',header=None).to_numpy()
x1_ground = pd.read_csv('x1_data.csv',header=None).to_numpy()

x_test = test_data[:, :-1]
y_test = test_data[:, -1]
# p = model(x_val.reshape(-1, channel, 128, 128)).squeeze().detach()

proba =[]
pred = []
g = 20

for i in tqdm(range(x_test.shape[0])):
    x = x_test[i].reshape((128,128))
    p1 = 0
    for j in range(g):
        x1 = np.array([x, x1_ground[j].reshape((128, 128))]).reshape(-1, 2, 128, 128)
        x1 = torch.Tensor(np.array(x1))
        p1 += model(x1).squeeze().detach()
    p1 /= g
    proba.append(1-p1)
    if p1<0.5:
        pred.append(1)
    else:
        pred.append(0)

def plot_roc(p, y):
    fpr, tpr, thresholds = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

print(sum(pred==y_test))
print(confusion_matrix(pred, y_test))
plot_roc(proba, y_test)
