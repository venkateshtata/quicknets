import torch
import torch.nn as nn
import torch.optim as optim


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_size, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def train_model(self, train_loader, learning_rate, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for imgs, labels in train_loader:
                output = self.forward(imgs)
                loss = criterion(output, imgs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)


    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

