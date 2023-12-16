import torch
import torch.nn as nn
import torch.optim as optim



class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
    def train_model(self, train_loader, learning_rate, epochs):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for inputs, outputs in train_loader:
                predictions = self.forward(inputs)
                loss = criterion(predictions, outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')


    def validate_model(self, valid_loader):
        criterion = nn.MSELoss()
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in valid_loader:
                predictions = self.forward(inputs)
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}%')

    
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)


    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()   
                
    