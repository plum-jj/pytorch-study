# %%
# Converted from: Chapter 4 숫자 이미지 생성하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 4 숫자 이미지 생성하기

# %%
# Jupyter-only command: !pip install -q torchinfo

# %%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
from torchvision.datasets.mnist import MNIST

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch
from torch import nn, optim
import torchvision
from torchvision.utils import make_grid
from torchinfo import summary

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 130
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 4.2 데이터와 데이터셋

# %%
base_dir = "dataset"

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])
mnist = DataLoader(MNIST(base_dir, train=True, download=True, transform=transform),
                   batch_size=30, shuffle=True)

for images, labels in mnist:
  for i in range(len(images)):
    plt.subplot(3, 10, i+1)
    plt.title(labels[i].item())
    plt.imshow(images[i].squeeze())
    plt.axis("off")
  break

# %%


# %% [markdown]
# ### 4.3 CGAN 모델

# %%
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.label_emb = nn.Embedding(10, 10)
    self.model = nn.Sequential(
      nn.Linear(794, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )

  def forward(self, x, labels):
    x = nn.Flatten()(x)
    c = self.label_emb(labels)
    x = torch.cat([x, c], dim=1)
    out = self.model(x)
    return out.squeeze()

# %%
class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.label_emb = nn.Embedding(10, 10)
    self.model = nn.Sequential(
      nn.Linear(110, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 784),
      nn.Tanh()
    )

  def forward(self, z, labels):
    z = nn.Flatten()(z)
    c = self.label_emb(labels)
    x = torch.cat([z, c], 1)
    out = self.model(x)
    return out.reshape(-1, 28, 28)

# %%
dummy_z = torch.randn(1, 100)
dummy_c = torch.randint(0, 10, (1,))
summary(Generator(), input_data=(dummy_z, dummy_c))

# %%
dummy_x = torch.randn(1, 28, 28)
dummy_c = torch.randint(0, 10, (1,))
summary(Discriminator(), input_data=(dummy_x, dummy_c))

# %% [markdown]
# ### 4.4 모델 학습

# %%
def generator_train_step(batch_size, discriminator, generator,
                         g_optimizer, criterion, device):
  z = torch.randn(batch_size, 100).to(device)
  fake_labels = torch.randint(0, 10, (batch_size,)).to(device)

  g_optimizer.zero_grad()

  fake_images = generator(z, fake_labels)
  validity = discriminator(fake_images, fake_labels)

  g_loss = criterion(validity, torch.ones(batch_size).to(device))
  g_loss.backward()
  g_optimizer.step()

  return g_loss.item()

# %%
def discriminator_train_step(batch_size, discriminator, generator,
                             d_optimizer, criterion,
                             images, labels, device):
  d_optimizer.zero_grad()

  # train with real images
  real_validity = discriminator(images.to(device), labels.to(device))
  real_loss = criterion(real_validity, torch.ones(batch_size).to(device))

  # train with fake images
  z = torch.randn(batch_size, 100).to(device)
  fake_labels = torch.randint(0, 10, (batch_size,)).to(device)

  fake_images = generator(z, fake_labels)
  fake_validity = discriminator(fake_images, fake_labels)
  fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))

  d_loss = real_loss + fake_loss
  d_loss.backward()
  d_optimizer.step()

  return d_loss.item()

# %%
def train(generator, discriminator, dataloader, criterion,
          g_optimizer, d_optimizer, epoch, device, n_critic=3):

  generator.train()
  discriminator.train()

  running_g_loss = 0
  running_d_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, labels) in enumerate(pbar):

      batch_size = len(images)

      # train discriminator in n_critic
      d_loss = 0
      for _ in range(n_critic):
        d_loss += discriminator_train_step(batch_size, discriminator,
                                           generator, d_optimizer, criterion,
                                           images, labels, device)
      d_loss /= n_critic

      # train generator
      g_loss = generator_train_step(batch_size, discriminator, generator,
                                    g_optimizer, criterion, device)

      running_d_loss += d_loss
      running_g_loss += g_loss
      pbar.set_postfix(d_loss=d_loss, g_loss=g_loss)

  return running_d_loss/len(dataloader), running_g_loss/len(dataloader)

# %%
def validation(generator, epoch, device):
  generator.eval()

  z = torch.randn(10, 100).to(device)
  labels = torch.arange(0, 10).to(device)
  images = generator(z, labels).unsqueeze(1)

  grid = make_grid(images, nrow=10, normalize=True)

  fig, ax = plt.subplots(figsize=(10, 1))
  ax.imshow(grid.permute(1, 2, 0).cpu(), cmap='binary')
  ax.axis('off')
  plt.show()

# %%
EPOCH = 30
BATCH_SIZE = 64
NUM_WORKERS = 2
LR = 0.0001

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
])
train_loader = DataLoader(
    MNIST(base_dir, train=True, download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)

# Start Training
for epoch in range(EPOCH):
  print(f"Epoch: {epoch}")

  gloss, dloss = train(generator, discriminator, train_loader, criterion,
                       g_optimizer, d_optimizer, epoch, device)

  validation(generator, epoch, device)

print(f"Save Model: {gloss}")
torch.save(generator.state_dict(), "last.pth")

# %% [markdown]
# ### 4.5 추론

# %%
z = torch.randn(100, 100).to(device)
labels = (torch.arange(0, 100) // 10).cuda()

with torch.no_grad():
  images = generator(z, labels).unsqueeze(1)

grid = make_grid(images, nrow=10, normalize=True)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).cpu(), cmap='binary')
ax.axis('off')

