import torch
import numpy as np
import pandas as pd
import sklearn
import os
import cv2
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch import nn
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import pickle
from lenet import LeNet
from load_data import load_data
print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")



##
## Pairup starategy - to keep paired set balanced
##

def create_pairs(x0, x1):
    pairedX = []
    pairedY = []

    x0_tr = x0
    x1_tr = x1

    # pairs for positive class = 192*191/2 = ~18300
    for i in range(x1_tr.shape[0] - 2):
        for j in range(i, x1_tr.shape[0] - 1):
            pair = [x1_tr[i], x1_tr[j]]
            pairedX.append(pair)
            pairedY.append(1)  # same class = 1

    positive_only_pair_size = len(pairedX)
    print("positive_only_pair_size ", positive_only_pair_size)

    # pairs for positive and negative = 192*192 ~ 36800
    for i in range(x1_tr.shape[0]):
        for j in range(x1_tr.shape[0]):  # this to create pairs where j = 192x -ve for each +ve
            pair = []
            a = int(
                np.random.uniform(low=0, high=x0_tr.shape[0] - 1, size=None))  # pick a random number/img from -ve class

            # alternate pairing - 0+1 or 1+0
            if j % 2 == 0:
                pair.append(x1_tr[i])
                pair.append(x0_tr[a])
            else:
                pair.append(x0_tr[a])
                pair.append(x1_tr[i])

            pairedX.append(pair)
            pairedY.append(0)  # same class = 0 (not same class)

    not_same_pair_size = len(pairedX) - positive_only_pair_size
    print("not_same_pair_size ", not_same_pair_size)

    # pairs for negative class only = 36800 - 18300 ~ 18500. Therefore, every -ve with 18-19 other
    range_for_x0_pairs = (x1_tr.shape[0] * x1_tr.shape[0]) // (2 * x0_tr.shape[0])
    for i in range(x0_tr.shape[0]):
        for j in range(range_for_x0_pairs):
            pair = []
            a = int(np.random.uniform(low=0, high=x0_tr.shape[0] - 1, size=None))
            pair.append(x0_tr[i])
            pair.append(x0_tr[a])

            pairedX.append(pair)
            pairedY.append(1)  # two negatives, so same class.

    negative_only_pair_size = len(pairedX) - positive_only_pair_size - not_same_pair_size
    print("negative_only_pair_size ", negative_only_pair_size)

    return pairedX, pairedY



def paired_data(x0, x1, y0, y1):
    x0_tr, x0_ts, y0_tr, y0_ts = train_test_split(x0, y0, test_size=0.2)
    x1_tr, x1_ts, y1_tr, y1_ts = train_test_split(x1, y1, test_size=0.2)

    x0_ground = []
    x1_ground = []

    x_0 = np.array(x0_tr)
    x_1 = np.array(x1_tr)

    for i in range (50):
        a = random.randint(0, x_0.shape[0]-1)
        x0_ground.append(x_0[a])
        a = random.randint(0, x_1.shape[0] - 1)
        x1_ground.append(x_1[a])

    x0_ground = np.array(x0_ground)
    x1_ground = np.array(x1_ground)


    x_train, y_train = create_pairs(x0_tr, x1_tr)
    x_test, y_test = create_pairs(x0_ts, x1_ts)
    x_val, y_val = x_test, y_test

    return x_train, x_val, y_train, y_val, x_test, y_test, x0_ground, x1_ground



def model_fit(model, opt, lossFn, train_dataloader, val_dataloader):
    model.to(device)
    # loop over our epochs
    for e in tqdm(range(0, EPOCHS)):
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        trainSteps = 0
        valSteps = 0
        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in train_dataloader:
            # send the input to the device
            x = x.reshape(-1, channel, 128, 128).to(device)
            # perform a forward pass and calculate the training loss
            pred = model(x).type(torch.float).squeeze().cpu()
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainSteps += 1
            pred = torch.round(pred)
            trainCorrect += (pred == y).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in val_dataloader:
                # send the input to the device
                x = x.reshape(-1, channel, 128, 128).to(device)
                # make the predictions and calculate the validation loss
                pred = model(x).type(torch.float).squeeze().cpu()
                totalValLoss += lossFn(pred, y)
                valSteps += 1
                # calculate the number of correct predictions
                pred = torch.round(pred)
                valCorrect += (pred == y).type(torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(train_dataloader.dataset)
        valCorrect = valCorrect / len(val_dataloader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy().item())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy().item())
        H["val_acc"].append(valCorrect)
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
        print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
            avgValLoss, valCorrect))
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


def paired_train():
    channel = 2
    (x0, x1, y0, y1) = load_data()
    (x_train, x_val, y_train, y_val, x_test, y_test, x0_ground, x1_ground) = paired_data(x0, x1, y0, y1)

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(x0_ground.shape)
    print(x1_ground.shape)

    x_train = torch.Tensor(x_train)  # transform to torch tensor
    y_train = torch.Tensor(y_train)
    x_val = torch.Tensor(x_val)
    y_val = torch.Tensor(y_val)
    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=bsize)  # create your dataloader
    val_dataset = TensorDataset(x_val, y_val)  # create your datset
    val_dataloader = DataLoader(val_dataset, batch_size=bsize)  # create your dataloader

    # initialize the LeNet model
    model = LeNet(numChannels=channel, classes=1)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=learingRate)
    lossFn = nn.BCELoss()

    model_fit(model, opt, lossFn, train_dataloader, val_dataloader)

    # %%
    # p = model(x_test.reshape(-1, channel, 128, 128).to(device)).squeeze().cpu().detach()

    model.cpu()

    # with open('paired_trained_model.pkl', 'wb') as file:
    #     pickle.dump(model, file)

    # df = pd.DataFrame(np.concatenate((x_test.reshape(-1 ,128*128),y_test.reshape(-1,1)),axis=1))
    # df.to_csv("test_data.csv", index=False, header=False)
    #
    # df = pd.DataFrame(x0_ground.reshape(-1, 128*128))
    # df.to_csv("x0_data.csv", index=False, header=False)
    #
    # df = pd.DataFrame(x1_ground.reshape(-1, 128 * 128))
    # df.to_csv("x1_data.csv", index=False, header=False)


    p = model(x_val.reshape(-1, channel, 128, 128)).squeeze().detach()
    print(confusion_matrix(torch.round(p), y_val))
    plot_roc(p, y_val)

if __name__ == '__main__':
    bsize = 100
    learingRate = 0.001
    EPOCHS = 25

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    #
    # channel = 1
    # normal_train()

    channel = 2
    paired_train()


