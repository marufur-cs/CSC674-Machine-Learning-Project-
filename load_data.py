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


def load_data():
    hernia_path = '/Users/sarojbhatta/Downloads/Dataset/Hernia192/'
    normal_path = '/Users/sarojbhatta/Downloads/Dataset/Normal1000/'
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for filename in os.listdir(hernia_path):
        img = cv2.imread(os.path.join(hernia_path, filename), 0)
        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x1.append(img)
            y1.append(1)
    for filename in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path, filename), 0)
        if img is not None:
            img = (img / 255.0).astype(np.float32)
            # store loaded image
            assert (img.shape == (128, 128))
            x0.append(img)
            y0.append(0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    x0 = np.array(x0)
    y0 = np.array(y0)
    return x0 ,x1, y0, y1