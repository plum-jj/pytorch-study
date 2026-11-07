# Converted from: Chapter 4 숫자 이미지 생성하기.ipynb
# Original .ipynb is preserved alongside this file.

# ## Chapter 4 숫자 이미지 생성하기

# Jupyter-only command: !pip install -q torchinfo

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
# MNIST는 0~9 손글씨 숫자 이미지 데이터셋이다.
from torchvision.datasets.mnist import MNIST

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# transforms는 이미지의 텐서 변환과 정규화를 순서대로 구성한다.
from torchvision import transforms

import torch
from torch import nn, optim
import torchvision
# make_grid(): 여러 이미지를 한 장의 격자 이미지로 배치한다.
from torchvision.utils import make_grid
from torchinfo import summary

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 130
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)  # gpu seed

# ### 4.2 데이터와 데이터셋

base_dir = "dataset"

transform = transforms.Compose([
        # ToTensor()는 픽셀값을 0~1 범위의 텐서로 바꾼다.
        transforms.ToTensor(),
        # 평균·표준편차를 각각 0.5로 지정해 0~1 값을 -1~1 범위로 바꾼다.
        # 생성자의 마지막 Tanh 출력 범위도 -1~1이므로 실제 이미지와 범위를 맞춘다.
        transforms.Normalize([0.5], [0.5])
])
# 한 배치에 30장을 담고, 매 에포크마다 데이터 순서를 섞는다.
mnist = DataLoader(MNIST(base_dir, train=True, download=True, transform=transform),
                   batch_size=30, shuffle=True)

for images, labels in mnist:
  for i in range(len(images)):
    plt.subplot(3, 10, i+1)
    plt.title(labels[i].item())
    plt.imshow(images[i].squeeze())
    plt.axis("off")
  break



# ### 4.3 CGAN 모델

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    # Embedding(10, 10)은 숫자 클래스 0~9를 각각 길이 10의 학습 가능한 벡터로 바꾼다.
    self.label_emb = nn.Embedding(10, 10)
    self.model = nn.Sequential(
      # 펼친 이미지 784개 픽셀과 라벨 임베딩 10개를 합쳐 총 794개를 입력한다.
      nn.Linear(794, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 1),
      # 0은 가짜, 1은 진짜에 가까운 확률을 출력한다.
      nn.Sigmoid()
    )

  def forward(self, x, labels):
    # Flatten(): 28×28 이미지를 샘플마다 길이 784의 1차원 벡터로 편다.
    x = nn.Flatten()(x)
    c = self.label_emb(labels)
    # 이미지 특징과 해당 숫자 라벨 정보를 열 방향으로 결합한다.
    x = torch.cat([x, c], dim=1)
    out = self.model(x)
    return out.squeeze()

class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.label_emb = nn.Embedding(10, 10)
    self.model = nn.Sequential(
      # 무작위 잠재 벡터 100개와 라벨 임베딩 10개를 합친 110개 값을 입력한다.
      nn.Linear(110, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 784),
      # Tanh는 MNIST 정규화 범위와 같은 -1~1 범위의 픽셀값을 출력한다.
      nn.Tanh()
    )

  def forward(self, z, labels):
    z = nn.Flatten()(z)
    c = self.label_emb(labels)
    x = torch.cat([z, c], 1)
    out = self.model(x)
    # 784개 출력을 배치별 28×28 숫자 이미지로 되돌린다.
    return out.reshape(-1, 28, 28)

dummy_z = torch.randn(1, 100)
# randint(0, 10)은 0 이상 10 미만의 숫자 라벨을 무작위로 만든다.
dummy_c = torch.randint(0, 10, (1,))
summary(Generator(), input_data=(dummy_z, dummy_c))

dummy_x = torch.randn(1, 28, 28)
dummy_c = torch.randint(0, 10, (1,))
summary(Discriminator(), input_data=(dummy_x, dummy_c))

# ### 4.4 모델 학습

def generator_train_step(batch_size, discriminator, generator,
                         g_optimizer, criterion, device):
  # 생성자의 입력으로 사용할 표준 정규분포 잠재 벡터와 목표 숫자 라벨을 만든다.
  z = torch.randn(batch_size, 100).to(device)
  fake_labels = torch.randint(0, 10, (batch_size,)).to(device)

  g_optimizer.zero_grad()

  fake_images = generator(z, fake_labels)
  validity = discriminator(fake_images, fake_labels)

  # 생성자는 가짜 이미지를 판별자가 진짜(1)라고 판단하도록 학습한다.
  g_loss = criterion(validity, torch.ones(batch_size).to(device))
  g_loss.backward()
  g_optimizer.step()

  return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator,
                             d_optimizer, criterion,
                             images, labels, device):
  d_optimizer.zero_grad()

  # train with real images
  # 실제 이미지는 정답 1로 학습한다.
  real_validity = discriminator(images.to(device), labels.to(device))
  real_loss = criterion(real_validity, torch.ones(batch_size).to(device))

  # train with fake images
  z = torch.randn(batch_size, 100).to(device)
  fake_labels = torch.randint(0, 10, (batch_size,)).to(device)

  # detach()로 생성자 계산 그래프를 분리해 이 단계에서는 판별자만 학습한다.
  fake_images = generator(z, fake_labels).detach()
  fake_validity = discriminator(fake_images, fake_labels)
  # 생성 이미지에는 가짜를 뜻하는 정답 0을 사용한다.
  fake_loss = criterion(fake_validity, torch.zeros(batch_size).to(device))

  d_loss = real_loss + fake_loss
  d_loss.backward()
  d_optimizer.step()

  return d_loss.item()

def train(generator, discriminator, dataloader, criterion,
          g_optimizer, d_optimizer, epoch, device, n_critic=3):

  generator.train()
  discriminator.train()

  running_g_loss = 0
  running_d_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, labels) in enumerate(pbar):

      batch_size = len(images)

      # 판별자를 n_critic번 학습한 뒤 생성자를 한 번 학습해 두 모델의 균형을 맞춘다.
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

def validation(generator, epoch, device):
  generator.eval()

  # 0~9 라벨을 하나씩 지정해 각 숫자를 한 장씩 생성한다.
  z = torch.randn(10, 100).to(device)
  labels = torch.arange(0, 10).to(device)
  with torch.no_grad():
    images = generator(z, labels).unsqueeze(1)

  # 채널 축을 추가한 10장의 이미지를 한 줄짜리 격자로 만든다.
  grid = make_grid(images, nrow=10, normalize=True)

  fig, ax = plt.subplots(figsize=(10, 1))
  ax.imshow(grid.permute(1, 2, 0).cpu(), cmap='binary')
  ax.axis('off')
  plt.show()

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
# 생성자와 판별자는 서로 다른 파라미터를 가지므로 옵티마이저도 각각 만든다.
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)

# Start Training
for epoch in range(EPOCH):
  print(f"Epoch: {epoch}")

  gloss, dloss = train(generator, discriminator, train_loader, criterion,
                       g_optimizer, d_optimizer, epoch, device)

  validation(generator, epoch, device)

# 마지막 에포크까지 학습한 생성자 가중치를 저장한다.
print(f"Save Model: {gloss}")
torch.save(generator.state_dict(), "last.pth")

# ### 4.5 추론

# 저장된 생성자를 새로 만들어 불러오므로 이 추론 셀만 따로 실행할 수도 있다.
generator = Generator().to(device)
generator.load_state_dict(torch.load("last.pth", map_location=device))
generator.eval()

z = torch.randn(100, 100).to(device)
# 0~99를 10으로 나눈 몫을 사용해 0~9 라벨을 각각 10번씩 만든다.
labels = (torch.arange(0, 100) // 10).to(device)

with torch.no_grad():
  images = generator(z, labels).unsqueeze(1)

grid = make_grid(images, nrow=10, normalize=True)

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).cpu(), cmap='binary')
ax.axis('off')
