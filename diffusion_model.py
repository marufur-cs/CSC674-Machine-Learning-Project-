import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from load_data2 import load_data
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pickle
from UNet_for_diffusion import UNet
from sklearn.model_selection import train_test_split

# Setting up device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on: ",device)
# Hyperparameters
num_steps = 10   # Number of diffusion steps
beta_start = 0.0001     # Starting noise level
beta_end = 0.0005     # Ending noise level
img_size = 128           # Size of MNIST images
bsize = 50

# Linear schedule for noise levels
beta = torch.linspace(beta_start, beta_end, num_steps).to(device)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)  # Cumulative product for alpha

# Forward diffusion process
def forward_diffusion(x, t):
    """Add noise to an image `x` at timestep `t`."""
    noise = torch.randn_like(x)
    alpha_t = alpha_bar[t].reshape(-1, 1, 1, 1)
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise
# U-Net architecture for denoising

# Training setup
def train(model, dataloader, optimizer, epochs=5):
    l = 100
    model.train()
    criterion = nn.MSELoss()
    for epoch in tqdm(range(epochs)):
        for images, _ in dataloader:
            images = images.reshape(-1, 1, 128, 128).to(device)
            batch_size = images.size(0)
            t = torch.randint(0, num_steps, (batch_size,), device=device)  # Random timestep for each batch
            noisy_images, noise = forward_diffusion(images, t)
            predicted_noise = model(noisy_images, t)
            loss = criterion(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        if loss.item()<l:
            l = loss.item()
            print("Saving the model")
            with open('diffusion_model.pkl', 'wb') as file:
                pickle.dump(model, file)


(x1, y1) = load_data();

x = np.array(x1)
y = np.array(y1)
print("Data Loaded: Shape of x: ", x.shape, "Shape of y: ", y.shape)
x_train = torch.Tensor(x)  # transform to torch tensor
y_train = torch.Tensor(y)

train_dataset = TensorDataset(x_train, y_train)  # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=bsize)  # create your dataloader

# Model and optimizer
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training the model
train(model, train_dataloader, optimizer, epochs=500)


