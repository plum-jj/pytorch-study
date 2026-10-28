# %%
# Converted from: Chapter 2 오토 인코더로 이미지 노이즈 제거하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 2 오토 인코더로 이미지 노이즈 제거하기

# %%
# Jupyter-only command: !pip install -q torchinfo

# %%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#reading files
import os
from torchvision.datasets.mnist import MNIST

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

#plots
import matplotlib.pyplot as plt

#model
import torch
from torch import nn, optim
import torchvision
from torchinfo import summary

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 322
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 2.1 데이터 분석

# %%
# Jupyter-only command: %cd /content/drive/MyDrive/Colab\ Notebooks/프로젝트로\ 시작하는\ Pytorch/6.2
# Jupyter-only command: %ls

# %%
def gaussian_noise(image, scale=0.8):
  noised = image + np.random.normal(loc=0, scale=scale, size=image.shape)
  noised = np.clip(noised, 0, 1)

  noised = torch.tensor(noised).float()
  return noised

# %%
base_dir = "dataset"
mnist = MNIST(base_dir, train=True, download=True)

# %%
image = mnist.data[123]
plt.subplot(1, 2, 1)
plt.title("origin")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title("noised")
plt.imshow(gaussian_noise(image))

# %% [markdown]
# ### 2.2 데이터셋

# %%
class NoisedDataset(Dataset):
  def __init__(self, root, train):
    self.mnist = MNIST(root=root, train=train, download=True)

  def __getitem__(self, i):
    origin = self.mnist.data[i].unsqueeze(0)
    noised = gaussian_noise(origin)
    return noised/255., origin/255.

  def __len__(self):
    return len(self.mnist)

# %%
trainset = NoisedDataset(base_dir, True)

# %%
noised, origin = trainset[123]
plt.subplot(1, 2, 1)
plt.title("origin")
plt.imshow(origin.squeeze())
plt.subplot(1, 2, 2)
plt.title("noised")
plt.imshow(noised.squeeze())

# %%
def get_dataset(root):
  train = NoisedDataset(root, True)
  val = NoisedDataset(root, False)

  return train, val

trainset, valset  = get_dataset(base_dir)
print(len(trainset), len(valset))

# %% [markdown]
# ### 2.3 오토 인코더(Auto-Encoder) 모델

# %%
class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_dims):
    super(BasicBlock, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, hidden_dims, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(hidden_dims, out_channels, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

    self.block = nn.Sequential(
      self.conv1,
      self.relu,
      self.conv2,
      self.relu
    )

  def forward(self, x):
    return self.block(x)

# %%
block = BasicBlock(1, 10, 4)
block(torch.rand((1, 8, 8))).shape

# %%
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.conv1 = BasicBlock(in_channels=1, out_channels=16, hidden_dims=16)
    self.conv2 = BasicBlock(in_channels=16, out_channels=8, hidden_dims=8)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.encoder = nn.Sequential(
      self.conv1,
      self.pool,
      self.conv2,
      self.pool
    )

  def forward(self, x):
    return self.encoder(x)

# %%
encoder = Encoder()
encoder(torch.rand((1, 8, 8))).shape

# %%
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()

    self.conv1 = BasicBlock(in_channels=8, out_channels=8, hidden_dims=8)
    self.conv2 = BasicBlock(in_channels=8, out_channels=16, hidden_dims=16)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=1,
                           kernel_size=3, padding=1)

    self.upsample1 = nn.ConvTranspose2d(in_channels=8, out_channels=8,
                                        kernel_size=2, stride=2)
    self.upsample2 = nn.ConvTranspose2d(in_channels=16, out_channels=16,
                                        kernel_size=2, stride=2)
    self.decoder = nn.Sequential(
      self.conv1,
      self.upsample1,
      self.conv2,
      self.upsample2,
      self.conv3
    )

  def forward(self, x):
    return self.decoder(x)

# %%
decoder = Decoder()
decoder(torch.rand((8, 2, 2))).shape

# %%
class CAE(nn.Module):
  def __init__(self):
    super(CAE, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x.squeeze()

# %%
cae = CAE()
cae(torch.rand((1, 8, 8))).shape

# %%
summary(cae, (1, 28, 28))

# %% [markdown]
# ### 2.4 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, targets) in enumerate(pbar):
      images, targets = images.to(device), targets.to(device)

      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs.squeeze(), targets.squeeze())

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  running_loss = 0.
  with tqdm(dataloader) as pbar:
    with torch.no_grad():
      for i, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

  print(f'[{epoch}/{EPOCH}] valid loss: {running_loss/len(dataloader):.4f}\n')

  return running_loss/len(dataloader)

# %%
EPOCH = 10
BATCH_SIZE = 32
NUM_WORKERS = 1
LR = 0.001

trainset, valset = get_dataset(base_dir)

# dataloader
train_loader = DataLoader(
  dataset=trainset,
  shuffle=True,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
  dataset=valset,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
)

# Model
model = CAE()

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

model = model.to(device)
criterion = criterion.to(device)

# Start Training
min_loss = 999
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  tloss = train(model, train_loader, criterion, optimizer, epoch, device)
  vloss = validation(model, val_loader, criterion, epoch, device)
  scheduler.step()

  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    torch.save(model.state_dict(), "best.pth")
    min_loss = vloss

# %% [markdown]
# ### 2.5 추론

# %%
model = CAE()
# load best model
model.load_state_dict(torch.load("best.pth"))
model.eval()

with torch.no_grad():
  image, target = valset[0]
  print(image.shape, target.shape)
  print(nn.MSELoss()(image.squeeze(), target.squeeze()))

  output = model(image)

  plt.subplot(1, 3, 1)
  plt.imshow(image.squeeze())
  plt.subplot(1, 3, 2)
  plt.imshow(target.squeeze())
  plt.subplot(1, 3, 3)
  plt.imshow(output.squeeze())

