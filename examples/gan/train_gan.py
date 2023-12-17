import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0002
batch_size = 64
image_size = 64
channels_size = 64
channels_img = 1
z_dim = 100
num_epochs = 5
features_disc = 64
features_gen = 64


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root="../../datasets", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

netD = gan.Discriminator(channels_img, features_disc).to(device)
netG = gan.Generator(z_dim, channels_img, features_gen).to(device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.rand(batch_size, z_dim, 1, 1).to(device)
        fake = netG(noise)

        # Train Discriminator
        netD.zero_grad()
        
        #  Label for fake image batches
        label = (torch.ones(real.size(0)) * 0.9).to(device)

        output = netD(real).reshape(-1)
        lossD_real = criterion(output, label)
        
        # Label for real image batches
        label = (torch.ones(fake.size(0)) * 0.1).to(device)
        
        output = netD(fake.detach()).reshape(-1)
        lossD_fake = criterion(output, label)
        
        lossD = (lossD_real + lossD_fake)/2
        lossD.backward()
        optimizerD.step()


        # Train Generator
        netG.zero_grad()

        # Label all the batches as 1
        label = torch.ones(real.size(0)).to(device)
        
        # Prediction from Discriminator on generated(fake) image
        output = netD(fake).reshape(-1)
        
        # Calculate loss between discriminator's predicion & label
        lossG = criterion(output, label)
        
        lossG.backward()
        optimizerG.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')

        