import sys
sys.path.append('../../')

import torch
import torchvision
import torchvision.transforms as transforms
from models import vae

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST('../../datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model = vae.VAE()

learning_rate = 0.001
epochs = 10

model.train_model(train_loader, learning_rate, epochs)
model.save_model("vae_fashion.pth")