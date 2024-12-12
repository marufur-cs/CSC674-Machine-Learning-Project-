from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from PIL import Image
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")



# Load the pre-trained model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

# Load and prepare the input image
input_image = Image.open("hernia.png")

# Generate a new image based on the input
output_image = pipe(prompt="chest x-ray image with hernia disease", image=input_image, strength=0.7).images[0]

# Save or display the output
output_image.save("generated_image.jpg")
