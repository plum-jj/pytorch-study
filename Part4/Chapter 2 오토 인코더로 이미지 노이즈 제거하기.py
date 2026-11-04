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

import os

# MNIST는 28×28 손글씨 숫자 이미지 데이터셋이다.
from torchvision.datasets.mnist import MNIST

from PIL import Image

# Dataset은 데이터 한 개를 반환하는 규칙을 정의하고,
# DataLoader는 Dataset을 배치 단위로 반복해서 가져온다.
from torch.utils.data import Dataset, DataLoader

# ToTensor는 이미지를 PyTorch 텐서로 변환한다.
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torchvision

# summary는 입력 크기를 기준으로 모델의 계층과 출력 모양을 요약한다.
from torchinfo import summary

# %%
# CUDA GPU 사용 가능 여부를 확인한 뒤 모델과 텐서를 계산할 장치를 정한다.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 322

# 파이썬, NumPy, PyTorch가 같은 난수 흐름을 사용하도록 시드를 고정한다.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device는 torch.device 객체이므로 실제 장치 종류는 device.type으로 확인한다.
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ### 2.1 데이터 분석

# %%
# 이 셀은 Colab의 현재 작업 폴더를 Chapter 2 실습 폴더로 이동하고 파일을 확인한다.
# 로컬 파이썬에서는 실제 데이터 경로에 맞게 수정해야 한다.
# Jupyter-only command: %cd /content/drive/MyDrive/Colab\ Notebooks/프로젝트로\ 시작하는\ Pytorch/6.2
# Jupyter-only command: %ls

# %%
def gaussian_noise(image, scale=0.8):
  # np.random.normal(): image와 같은 모양으로 평균 0, 표준편차 scale인 가우시안 잡음을 만든다.
  # 평균이 0이라는 것은 생성 범위가 0부터 시작한다는 뜻이 아니라, 잡음이 0을 중심으로 분포한다는 뜻이다.
  noised = image + np.random.normal(loc=0, scale=scale, size=image.shape)
  # np.clip(): 잡음을 더한 픽셀값을 0~1 범위 밖으로 나가지 않게 잘라 낸다.
  noised = np.clip(noised, 0, 1)

  # torch.tensor().float(): NumPy 배열을 모델 입력에 사용할 float32 텐서로 바꾼다.
  noised = torch.tensor(noised).float()
  return noised

# %%
base_dir = "dataset"
# MNIST(): 손글씨 이미지 데이터를 내려받아 dataset 폴더에 저장한다.
# train=True이므로 학습용 이미지 60,000개를 불러온다.
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
    # train 값이 True이면 학습용 60,000개, False이면 평가용 10,000개를 사용한다.
    self.mnist = MNIST(root=root, train=train, download=True)

  def __getitem__(self, i):
    # MNIST 원본 한 장은 (28, 28)이므로 채널 축을 추가해 (1, 28, 28)로 만든다.
    origin = self.mnist.data[i].unsqueeze(0)
    noised = gaussian_noise(origin)
    # 모델에는 픽셀값을 255로 나눈 0~1 범위의 입력과 정답을 한 쌍으로 전달한다.
    return noised/255., origin/255.

  def __len__(self):
    # DataLoader가 전체 데이터 개수를 알 수 있도록 MNIST의 길이를 반환한다.
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
  # 같은 Dataset 클래스를 train 값만 다르게 호출해 학습용과 검증용 데이터를 나눈다.
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

    # kernel_size=3, padding=1이므로 합성곱 전후의 이미지 높이와 너비는 유지된다.
    self.conv1 = nn.Conv2d(in_channels, hidden_dims, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(hidden_dims, out_channels, kernel_size=3, padding=1)
    self.relu = nn.ReLU()

    # nn.Sequential(): 등록한 계층을 위에서 아래 순서대로 실행하는 하나의 블록으로 묶는다.
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
# 입력은 배치 축을 생략한 (채널 1, 높이 8, 너비 8)이고 출력은 (10, 8, 8)이 된다.
block(torch.rand((1, 8, 8))).shape

# %%
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()

    self.conv1 = BasicBlock(in_channels=1, out_channels=16, hidden_dims=16)
    self.conv2 = BasicBlock(in_channels=16, out_channels=8, hidden_dims=8)
    # AvgPool2d(2, 2)는 높이와 너비를 절반으로 줄인다: 28×28 → 14×14 → 7×7.
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

    # ConvTranspose2d(2, 2)는 인코더에서 줄인 크기를 두 배씩 복원한다: 7×7 → 14×14 → 28×28.
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
    # 인코더가 입력을 작은 특징으로 압축하고, 디코더가 다시 원본 크기의 이미지로 복원한다.
    x = self.encoder(x)
    x = self.decoder(x)
    # squeeze(): 크기가 1인 축을 제거한다. 배치가 1일 때는 배치 축까지 사라질 수 있다.
    return x.squeeze()

# %%
cae = CAE()
cae(torch.rand((1, 8, 8))).shape

# %%
# summary(): (채널 1, 28, 28) 입력을 기준으로 계층별 출력 크기와 파라미터 수를 확인한다.
summary(cae, (1, 28, 28))

# %% [markdown]
# ### 2.4 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  # train(): Dropout, BatchNorm 등이 학습 모드로 동작하게 한다.
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, targets) in enumerate(pbar):
      images, targets = images.to(device), targets.to(device)

      optimizer.zero_grad()

      outputs = model(images)
      # 출력과 정답 모두 크기가 1인 채널 축을 제거해 같은 모양으로 손실을 계산한다.
      loss = criterion(outputs.squeeze(), targets.squeeze())

      # backward()가 손실의 기울기를 계산하고 step()이 그 기울기로 가중치를 갱신한다.
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch, device):
  # eval(): 검증 중 모델을 평가 모드로 전환한다.
  model.eval()

  running_loss = 0.
  with tqdm(dataloader) as pbar:
    # no_grad(): 검증에서는 가중치를 바꾸지 않으므로 기울기 기록을 꺼 메모리와 계산량을 줄인다.
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
# Adam(): 계산된 기울기를 바탕으로 모델 파라미터를 갱신하는 옵티마이저이다.
optimizer = optim.Adam(model.parameters(), lr=LR)
# MSELoss(): 복원 이미지와 원본 이미지의 픽셀별 제곱 오차 평균을 계산한다.
criterion = nn.MSELoss()
# StepLR(): 3에포크마다 학습률을 기존 값의 0.5배로 줄인다.
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

  # 검증 손실이 지금까지 가장 낮을 때의 가중치만 best.pth에 저장한다.
  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    torch.save(model.state_dict(), "best.pth")
    min_loss = vloss

# %% [markdown]
# ### 2.5 추론

# %%
model = CAE()
# load best model
# load_state_dict(): best.pth에 저장된 가중치를 같은 구조의 CAE 모델에 적용한다.
# GPU에서 저장한 파일도 CPU에서 열 수 있게 하려면 torch.load(..., map_location="cpu")를 사용할 수 있다.
model.load_state_dict(torch.load("best.pth", map_location="cpu"))
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

