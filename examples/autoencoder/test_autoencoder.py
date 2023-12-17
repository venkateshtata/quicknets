import sys
sys.path.append('../../')

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import autoencoder




def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Load the trained model
model = autoencoder.ConvAutoencoder(1)
model.load_state_dict(torch.load("ConvAutoEncoder.pth"))
model.eval()  # Set the model to evaluation mode

# Load MNIST test dataset
test_dataset = torchvision.datasets.MNIST(root="../../datasets", train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)

# Get a batch of test images
dataiter = iter(test_loader)
images, _ = dataiter.next()

# Pass the images through the autoencoder
reconstructed = model(images)


# Plot the original and reconstructed images
plt.figure(figsize=(10, 4))
for i in range(len(images)):
    # Original Image
    plt.subplot(2, len(images), i+1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.axis('off')

    # Reconstructed Image
    plt.subplot(2, len(images), i+1+len(images))
    plt.imshow(reconstructed[i].detach().squeeze(), cmap='gray')
    plt.axis('off')

plt.show()