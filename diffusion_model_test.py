import torch
import matplotlib.pyplot as plt
from load_data2 import load_data
import numpy as np
import pickle
from UNet_for_diffusion import UNet


num_steps = 10  # Number of diffusion steps
beta_start = 0.0001     # Starting noise level
beta_end = 0.0005
beta = torch.linspace(beta_start, beta_end, num_steps)
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)


(x0, y0, x1_tr, y1_tr, x1_ts, y1_ts) = load_data();

x = np.array(x1_tr)
y = np.array(y1_tr)
x = torch.Tensor(x)
y = torch.Tensor(y)

@torch.no_grad()
def generate_images(model, img):
    model.eval()
    noise = torch.randn_like(img)
    alpha_t = alpha_bar[num_steps-1].reshape(-1, 1, 1, 1)
    x = torch.sqrt(alpha_t) * img + torch.sqrt(1 - alpha_t) * noise
    xx=x
    xx = xx.cpu()
    plt.imshow(xx.reshape((128,128)))
    plt.title("Noisy")
    plt.show()
    for t in reversed(range(num_steps)):
        z = torch.randn_like(img) if t>1 else 0
        predicted_noise = model(x, t)
        # x = torch.sqrt(alpha_bar[t]) * x + torch.sqrt(1 - alpha_bar[t]) * predicted_noise  # Denoising step
        x = (1/torch.sqrt(alpha[t])) * (x - (((1-alpha[t])/(torch.sqrt(1-alpha_bar[t])))* predicted_noise)) + (beta[t] * z)  # Denoising step

    x = x.clamp(0, 1)
    print(x)
    return x.cpu()


model = UNet()
model = pickle.load(open('diffusion_model.pkl','rb'))
model.to('cpu')
i = 15
img = x[i]
plt.imshow(img)
plt.title("Original")
plt.show()
# Generating and displaying images
generated_image= generate_images(model, img)

plt.imshow(generated_image.reshape((128,128)).detach().numpy())
plt.title("Generated")
plt.show()
