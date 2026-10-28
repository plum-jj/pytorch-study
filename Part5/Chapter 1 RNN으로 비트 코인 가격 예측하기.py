# %%
# Converted from: Chapter 1 RNN으로 비트 코인 가격 예측하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 1 RNN으로 비트 코인 가격 예측하기

# %%
# Jupyter-only command: !pip install -q torchinfo

# %%
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torchinfo import summary

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 1202
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 1.3 비트코인 가격 데이터

# %%
# Jupyter-only command: !wget https://www.cryptodatadownload.com/cdd/Bithumb_BTCKRW_d.csv

# %%
df = pd.read_csv("Bithumb_BTCKRW_d.csv", skiprows=1)
df['Date'] = df['Date'].apply(pd.Timestamp)
df = df.sort_values('Date')
df.set_index('Date', inplace=True)
print(df.shape)
print(df)

# %%
df_close = df.loc[:, ["Close"]]
print(df_close.tail())

# %%
ax = df_close.plot(rot=45)
ax.axvline("2017-06-01", linestyle="--", c="black")
ax.axvline("2020-01-01", linestyle="--", c="red")
ax.axvline("2020-06-01", linestyle="--", c="green")

# %%
df_train = df_close[
  (df_close.index >= "2017-06-01")&(df_close.index < "2020-01-01")
]
print(df_train[:5])

# %%
scaler = MinMaxScaler()
data = scaler.fit_transform(df_train[:5])
data

# %%
scaler.inverse_transform(data)

# %%


# %%
class BitcoinDataset(Dataset):
  def __init__(self, split="train", seq_len=4):
    self.split = split
    self.seq_len = seq_len

    self.date_range = {
      "train": ["2017-06-01", "2020-01-01"],
      "val": ["2020-01-01", "2020-06-01"],
      "test": ["2020-06-01", "2020-11-21"]
    }

    self.data = self._load_data(self.date_range[self.split])
    self.data, self.labels = self._transform(self.data, self.seq_len)

  def _load_data(self, date_range):
    df = pd.read_csv("Bithumb_BTCKRW_d.csv", skiprows=1)
    df['Date'] = df['Date'].apply(pd.Timestamp)
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    df_close = df.loc[:, ["Close"]]
    df = df_close[
      (df_close.index >= date_range[0])&(df_close.index < date_range[1])
    ]

    self.scaler = MinMaxScaler()
    data = self.scaler.fit_transform(df)
    return data

  def _transform(self, data, seq_len):
    x, y = [], []

    for i in range(len(data) - seq_len - 1):
      x_i = data[i:i+seq_len]
      y_i = data[i+1:i+seq_len+1]
      x.append(x_i)
      y.append(y_i)

    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)

    return x_tensor, y_tensor

  def __getitem__(self, i):
    return self.data[i], self.labels[i]

  def __len__(self):
    return len(self.data)

# %%
trainset = BitcoinDataset()
valset = BitcoinDataset(split="val")
testset = BitcoinDataset(split="test")

print(len(trainset), len(valset), len(testset))

# %%
def plot_sequence(x, y):
  plt.ylabel("Scaled Close")
  plt.xlabel("Time Bars")
  plt.plot(range(len(x)), x, color='r', label='Features')
  plt.plot(range(1, len(x)+1), y, color='b', linestyle='--', label='Target')
  plt.legend()

# %%
plt.figure(figsize=(10,2))
for i, (x, y) in enumerate(trainset):
  plt.subplot(1, 3, i+1)
  plot_sequence(x, y)

  if i == 2:
    break

# %% [markdown]
# ### 1.4 LSTM 모델
#

# %%
model = nn.LSTM(4, 4)
summary(model, (1, 4))

# %%
output, (hidden_state, cell_state) = model(torch.randn((1,4)).to(device))
print(output)
print(hidden_state)
print(cell_state)

# %% [markdown]
# ### 1.5 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (features, targets) in enumerate(pbar):
      features = features.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      outputs, _ = model(features)
      loss = criterion(outputs, targets)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    with torch.no_grad():
      for i, (features, targets) in enumerate(pbar):
        features = features.to(device)
        targets = targets.to(device)

        outputs, _ = model(features)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

# %%
EPOCH = 60
BATCH_SIZE = 128
NUM_WORKERS = 2
LR = 0.1

trainset = BitcoinDataset()
valset = BitcoinDataset(split="val")
testset = BitcoinDataset(split="test")

train_loader = DataLoader(
  dataset=trainset,
  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
  dataset=valset,
  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
  dataset=testset,
  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)

# Model
model = nn.LSTM(4, 4)

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

model = model.to(device)
criterion = criterion.to(device)

# Start Training
min_loss = 999
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  tloss = train(model, train_loader, criterion, optimizer, epoch, device)
  vloss = validation(model, val_loader, criterion, epoch, device)
  print(f'[{epoch}/{EPOCH}] valid loss: {vloss:.4f}\n')
  scheduler.step()

  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    torch.save(model.state_dict(), "best.pth")
    min_loss = vloss

# %% [markdown]
# ### 1.6 추론

# %%
def test(model, dataloader, device):
  model.eval()

  predicts = []

  with tqdm(dataloader) as pbar:
    with torch.no_grad():
      for i, (features, _) in enumerate(pbar):
        features = features.to(device)

        outputs, _ = model(features)
        predicts.append(outputs[:, -1])

  return torch.cat(predicts)

# %%
def show_plot(model, dataloader, device):
  scaler = dataloader.dataset.scaler

  actual = dataloader.dataset.data[:, -1]
  predicts = test(model, dataloader, device).cpu().numpy()

  df = pd.DataFrame({"actual": actual, "predicted": predicts})
  df = df.apply(
    lambda x: scaler.inverse_transform(np.array(x).reshape(-1, 1)).reshape((-1))
  )
  df.plot(figsize=(14,5))

# %%
model = nn.LSTM(4, 4).to(device)
model.load_state_dict(torch.load("best.pth"))

# %%
show_plot(model, train_loader, device)

# %%
show_plot(model, val_loader, device)

# %%
show_plot(model, test_loader, device)

