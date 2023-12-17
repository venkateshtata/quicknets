import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms
from models import vae



model = vae.VAE()
model.load_model("vae_fashion.pth")

def generate_samples(model, num_samples=10):
    with torch.no_grad():
        # Sample random points in the latent space
        z = torch.randn(num_samples, 20)  # 20 is the dimension of your latent space

        # Generate images from these points
        samples = model.decode(z).cpu()

        return samples


def plot_samples(samples, num_samples=10):
    samples = samples.view(num_samples, 28, 28)  # Reshape to 2D images (28x28 is the size of Fashion-MNIST images)
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')

    plt.show()


# Set the model to evaluation mode
model.eval()

# Generate samples
generated_samples = generate_samples(model, num_samples=16)  # Feel free to change the number of samples

# Plot the samples
plot_samples(generated_samples, num_samples=16)

