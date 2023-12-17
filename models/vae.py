import torch
from torch import nn
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(64*14*14, 400)
        self.fc21 = nn.Linear(400, 20)  #mean
        self.fc22 = nn.Linear(400, 20)  #log variance

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 64*14*14)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)


    def encode(self, x):
        h1 = torch.relu(self.conv1(x))
        h1 = torch.relu(self.conv2(h1))
        h1 = h1.view(-1, 64*14*14)
        h1 = torch.relu(self.fc1(h1))
        return self.fc21(h1), self.fc22(h1)
    

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3 = torch.relu(self.fc4(h3))
        h3 = h3.view(-1, 64, 14, 14)
        h3 = torch.relu(self.deconv1(h3))
        return torch.sigmoid(self.deconv2(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def vae_loss(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train_model(self, train_loader, learning_rate, epochs):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = 0
        for epoch in range(epochs):
            for batch_idx, (data, _) in enumerate(train_loader):
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.forward(data)
                loss = self.vae_loss(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss+=loss.item()
                optimizer.step()

                if batch_idx % 100==0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')

            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()
