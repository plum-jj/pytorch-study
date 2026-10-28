# %%
# Converted from: Chapter 1 손글씨 숫자 분류와 CNN.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 딥러닝의 기본 이미지 분류하기
# ## Chapter 1 손글씨 숫자 분류와 CNN

# %% [markdown]
# ### 1.5 학습 준비

# %%
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import matplotlib.pyplot as plt

# %%
is_cuda = torch.cuda.is_available()   # GPU가 사용가능한지
device = torch.device("cuda" if is_cuda else "cpu") # GPU가 사용가능하다면 GPU를 사용 아니면 CPU 사용
print(is_cuda, device)

# %%
seed = 1010
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 1.6 MNIST 데이터셋

# %%
from torch.utils.data import random_split


def get_mnist_dataloaders(batch_size, val=0.2, num_workers=0):
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  # split
  mnist_train_origin = datasets.MNIST(
      root='MNIST/',
      train=True,
      download=True,
      transform=transform,
  )
  val_samples = int(len(mnist_train_origin) * val)
  train_samples = len(mnist_train_origin) - val_samples

  # dataset
  mnist_train, mnist_val = random_split(
      mnist_train_origin,
       (train_samples, val_samples),
  )
  mnist_test = datasets.MNIST(
      root='MNIST/',
      train=False,
      download=True,
      transform=transform,
  )

  # dataloader
  train_loader = DataLoader(
      dataset=mnist_train,
      batch_size=batch_size,
      drop_last=True,
      shuffle=True,
      num_workers=num_workers,
  )
  val_loader = DataLoader(
      dataset=mnist_val,
      batch_size=batch_size,
      num_workers=num_workers,
  )
  test_loader = DataLoader(
      dataset=mnist_test,
      batch_size=batch_size,
      num_workers=num_workers,
  )

  return train_loader, val_loader, test_loader

# %% [markdown]
# ### 1.7 LeNet

# %%
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# %% [markdown]
# ### 1.8 학습, 검증, 테스트

# %%
def train(model, dataloader, criterion, optimizer, epoch):
  # train mode
  model.train()

  # 학습 통계
  running_loss = 0
  correct = 0

  for i, (data, targets) in enumerate(dataloader):
    data.to(device), targets.to(device)   # data device로 이동

    optimizer.zero_grad()       # gradient 초기화
    outputs = model(data)       # forward
    loss = criterion(outputs, targets)    # loss 계산
    loss.backward()             # gradient 계산
    optimizer.step()            # weight update

    running_loss += loss.item()

    # Accuracy 계산
    _, predicted = torch.max(outputs, 1)
    correct += predicted.eq(targets.view_as(predicted)).sum()

  # Accuracy 출력
  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num
  print(
      f"[{epoch}/{EPOCH}]",
      f"train loss: {running_loss/len(dataloader):.4f}",
      f"train acc: {correct}/{data_num} ({acc:.2f}%)"
  )

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch):
  # eval 모드
  model.eval()

  # 검증 통계
  correct = 0
  running_loss = 0.

  with torch.no_grad():   # gradient 계산 안하기
    for i, (data, targets) in enumerate(dataloader):
      data.to(device), targets.to(device)   # data device로 이동

      outputs = model(data)               # forward
      loss = criterion(outputs, targets)  # loss 계산

      running_loss += loss.item()

      # Accuracy 계산
      _, predicted = torch.max(outputs, 1)
      correct += predicted.eq(targets.view_as(predicted)).sum()

  # Accuracy 계산
  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num
  print(f'[{epoch}/{EPOCH}] valid loss: {running_loss/len(dataloader):.4f} valid acc: {correct}/{data_num} ({acc:.2f}%)\n')

  return running_loss/len(dataloader)

# %%
def test(model, dataloader):
  # eval 모드
  model.eval()

  # 테스트 통계
  correct = 0

  with torch.no_grad():
    for data, targets in dataloader:
      data.to(device), targets.to(device)

      outputs = model(data)    # forward

      # Accuracy 계산
      _, predicted = torch.max(outputs, 1)
      correct += predicted.eq(targets.view_as(predicted)).sum()

  # Accuracy 계산
  data_num = len(dataloader.dataset)
  print(f'Test Accuracy: {correct}/{data_num} ({100. * correct / data_num:.2f}%)')

# %%
# Hyperparameters
# Training
EPOCH = 5
BATCH_SIZE = 32
NUM_WORKERS = 1
LR = 0.001

# Model
model = LeNet()
print(model)

# MNIST dataloader
train_loader, val_loader, test_loader = \
get_mnist_dataloaders(BATCH_SIZE, val=0.2, num_workers=NUM_WORKERS)

# Loss, Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(EPOCH):
  tloss = train(model, train_loader, criterion, optimizer, epoch)
  vloss = validation(model, val_loader, criterion, epoch)

  train_losses.append(tloss)
  val_losses.append(vloss)

test(model, test_loader)

# %%
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.plot(val_losses)

