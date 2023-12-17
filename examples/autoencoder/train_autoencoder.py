import sys
sys.path.append('../../')

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import autoencoder


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="../../datasets", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


input_size = 1
model = autoencoder.ConvAutoencoder(1)

learning_rate = 0.001
epochs = 50
model.train_model(train_loader, learning_rate, epochs)

model.save_model("ConvAutoEncoder.pth")