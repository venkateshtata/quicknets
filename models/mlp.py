import numpy
import torch
import torch.nn as nn
import torch.optim as optim



class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.final_layer = nn.Linear(128, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.final_layer(out)

        return out
    
    def train_model(self, train_loader, learning_rate, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

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



