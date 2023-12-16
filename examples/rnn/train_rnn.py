import sys
sys.path.append('../../')

import torch
import torchvision
import torchvision.transforms as transforms
from models import rnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="../../datasets", train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.MNIST(root="../../datasets", train=False, download=True, transform=train_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

num_classes = 10
input_size = 28
model = rnn.RNN(input_size, num_classes)


learning_rate = 0.001
epochs = 10
model.train_model(train_loader,learning_rate, epochs)
model.test_model(test_loader)

model.save_model('rnn_mnist.pth')