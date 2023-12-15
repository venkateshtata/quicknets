import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.avgpool = nn.AvgPool2d(2)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgpool(x)

        x = F.relu(self.conv2(x))
        x = self.avgpool(x)

        x = x.view(-1, 400)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def train_model(self, train_loader, learning_rate, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:    # print every 1000 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                    running_loss = 0.0

            print(f'Finished training')

    def test_model(self, test_loader):
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct/total
        print(f'Accuracy: {accuracy:.2f}%')


    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_mode(self, file_path):
        self.load_state_dict(torch.load(file_path))


    



