import os
import cv2
import numpy as np
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import pickle
from lenet import LeNet

def load_data():
    hernia_path_train = '/Users/nayem/Desktop/Research/MS_Thesis/Hernia PA (154) Train/'
    hernia_path_test = '/Users/nayem/Desktop/Research/MS_Thesis/Hernia PA (38) Test/'
    normal_path = '/Users/nayem/Desktop/Research/MS_Thesis/Normal PA (1000)/'
    gen_path = '/Users/nayem/Desktop/Research/MS_Thesis/Generated/'

    x1_tr = []
    y1_tr = []
    x1_ts = []
    y1_ts = []
    x0 = []
    y0 = []

    for filename in os.listdir(hernia_path_train):
        img = cv2.imread(os.path.join(hernia_path_train, filename), 0)
        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x1_tr.append(img)
            y1_tr.append(1)

    for filename in os.listdir(hernia_path_test):
        img = cv2.imread(os.path.join(hernia_path_test, filename), 0)
        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x1_ts.append(img)
            y1_ts.append(1)

    for filename in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path, filename), 0)
        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x0.append(img)
            y0.append(0)

    for filename in os.listdir(gen_path):
        img = cv2.imread(os.path.join(gen_path, filename), 0)
        img = cv2.resize(img, (128, 128))

        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x1_tr.append(img)
            y1_tr.append(1)

    return (x0, y0, x1_tr, y1_tr, x1_ts, y1_ts)




print(f"PyTorch version: {torch.__version__}")
# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

def model_fit(model, opt, lossFn, train_dataloader, val_dataloader):
    model.to(device)
    max_val_acc = 0
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
    with open('pretrained_diff_aug_model.pkl', 'wb') as file:
        pickle.dump(model, file)
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


def normal_train():
    channel = 1


    (x0, y0, x1_tr, y1_tr, x1_ts, y1_ts) = load_data();

    x_train = np.concatenate((np.array(x0)[:800, :, :], np.array(x1_tr)), axis=0)
    y_train = np.concatenate((np.array(y0)[:800], np.array(y1_tr)), axis=0)
    x_test = np.concatenate((np.array(x0)[800:, :], np.array(x1_ts)), axis=0)
    y_test = np.concatenate((np.array(y0)[800:], np.array(y1_ts)), axis=0)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(sum(y_test))

    x_train, xts, y_train, yts = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
    x_train = np.concatenate((x_train, xts), axis=0)
    y_train = np.concatenate((y_train, yts), axis=0)

    x_train = torch.Tensor(x_train)  # transform to torch tensor
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    # (x0, y0, x1_tr, y1_tr, x1_ts, y1_ts) = load_data();
    # x_train = np.concatenate((np.array(x0)[:800, :, :], np.array(x1_tr)), axis=0)
    # y_train = np.concatenate((np.array(y0)[:800], np.array(y1_tr)), axis=0)
    # x_test = np.concatenate((np.array(x0)[800:, :], np.array(x1_ts)), axis=0)
    # y_test = np.concatenate((np.array(y0)[800:], np.array(y1_ts)), axis=0)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(sum(y_test))
    # x_train = torch.Tensor(x_train)  # transform to torch tensor
    # y_train = torch.Tensor(y_train)
    # x_test = torch.Tensor(x_test)
    # y_test = torch.Tensor(y_test)

    train_dataset = TensorDataset(x_train, y_train)  # create your datset
    train_dataloader = DataLoader(train_dataset, batch_size=bsize)  # create your dataloader
    test_dataset = TensorDataset(x_test, y_test)  # create your datset
    test_dataloader = DataLoader(test_dataset, batch_size=bsize)  # create your dataloader
    model = LeNet(numChannels=channel, classes=1)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=laringRate)
    lossFn = nn.BCELoss()

    model_fit(model, opt, lossFn, train_dataloader, test_dataloader)

    model = pickle.load(open('pretrained_diff_aug_model.pkl', 'rb'))
    model.cpu()
    p = model(x_test.reshape(-1, channel, 128, 128)).squeeze().detach()
    print(confusion_matrix(torch.round(p), y_test))
    plot_roc(p, y_test)



if __name__ == '__main__':
    bsize = 100
    laringRate = 0.001
    EPOCHS = 25

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    #
    channel = 1
    normal_train()



