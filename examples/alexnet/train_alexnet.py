import sys
sys.path.append('../../')

import torch
import torchvision
import torchvision.transforms as transforms
from models import alexnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

valid_transform = transforms.Compose([
    transforms.Resize((277,277)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])


# load the dataset
train_dataset = torchvision.datasets.CIFAR10(root="../../datasets", train=True, download=True, transform=train_transform)
valid_dataset = torchvision.datasets.CIFAR10(root="../../datasets", train=False, download=True, transform=valid_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2)

num_classes = 10
model = alexnet.AlexNet(num_classes)


learning_rate = 0.001
epochs = 10
model.train_model(train_loader, learning_rate, epochs)
model.test_model(valid_loader)

model.save_model('alexnet_cifar10.pth')


