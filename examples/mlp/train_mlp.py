import sys
sys.path.append('../../')

import torch
import torchvision
import torchvision.transforms as transforms
from models import mlp

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x:torch.flatten(x))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)


input_size = 3*32*32
num_classes = 10
model = mlp.MLP(input_size, num_classes)


learning_rate = 0.001
epochs = 2
model.train_model(train_loader, learning_rate, epochs)
model.test_model(test_loader)

model.save_model('mlp_cifar10.pth')
