# Converted from: Chapter 1 RNN으로 비트 코인 가격 예측하기.ipynb
# Original .ipynb is preserved alongside this file.

# ## Chapter 1 RNN으로 비트 코인 가격 예측하기

# Jupyter-only command: !pip install -q torchinfo

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# MinMaxScaler는 각 특성을 지정한 범위(기본 0~1)로 변환하고 원래 값으로 복원할 수 있다.
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn, optim
from torchinfo import summary

# CUDA GPU 사용 가능 여부에 따라 모델과 텐서를 계산할 장치를 정한다.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 1202
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
# torch.device 객체의 실제 장치 종류는 device.type으로 확인한다.
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)  # gpu seed

# ### 1.3 비트코인 가격 데이터

# Jupyter-only command: !wget https://www.cryptodatadownload.com/cdd/Bithumb_BTCKRW_d.csv

# skiprows=1은 파일 맨 위의 안내 문구 한 줄을 제외하고 표를 읽는다는 뜻이다.
df = pd.read_csv("Bithumb_BTCKRW_d.csv", skiprows=1)
# 문자열 날짜를 Pandas Timestamp로 바꿔 날짜 비교와 정렬이 가능하게 한다.
df['Date'] = df['Date'].apply(pd.Timestamp)
# 날짜가 과거에서 미래 순서가 되도록 행을 정렬한다.
df = df.sort_values('Date')
# Date 열을 행 인덱스로 옮긴다. inplace=True이므로 df 자체가 변경된다.
df.set_index('Date', inplace=True)
print(df.shape)
print(df)

# loc[:, ["Close"]]는 모든 행(:)에서 Close 열만 선택하며,
# 대괄호를 두 겹 사용해 Series가 아닌 2차원 DataFrame 형태를 유지한다.
df_close = df.loc[:, ["Close"]]
print(df_close.tail())

ax = df_close.plot(rot=45)
ax.axvline("2017-06-01", linestyle="--", c="black")
ax.axvline("2020-01-01", linestyle="--", c="red")
ax.axvline("2020-06-01", linestyle="--", c="green")

df_train = df_close[
  # 날짜 인덱스에 조건을 적용해 학습 기간의 행만 남긴다.
  (df_close.index >= "2017-06-01")&(df_close.index < "2020-01-01")
]
print(df_train[:5])

scaler = MinMaxScaler()
# fit_transform()은 최솟값과 최댓값을 학습한 뒤 데이터를 0~1 범위로 변환한다.
data = scaler.fit_transform(df_train[:5])
data

# inverse_transform()은 정규화된 값을 원래 가격 단위로 되돌린다.
scaler.inverse_transform(data)



class BitcoinDataset(Dataset):
  def __init__(self, split="train", seq_len=4):
    self.split = split
    self.seq_len = seq_len

    self.date_range = {
      "train": ["2017-06-01", "2020-01-01"],
      "val": ["2020-01-01", "2020-06-01"],
      "test": ["2020-06-01", "2020-11-21"]
    }

    # split 이름에 맞는 날짜 구간을 읽고, 연속된 시계열 입력과 정답으로 변환한다.
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

    # 학습·검증·테스트 Dataset은 각자 자신의 기간에 맞는 스케일 정보를 보관한다.
    self.scaler = MinMaxScaler()
    data = self.scaler.fit_transform(df)
    return data

  def _transform(self, data, seq_len):
    x, y = [], []

    for i in range(len(data) - seq_len - 1):
      # 입력 x_i는 현재부터 seq_len일 동안의 가격이고,
      # 정답 y_i는 하루 뒤부터 시작하는 같은 길이의 가격이다.
      # 예: seq_len=4이면 [0,1,2,3]으로 [1,2,3,4]를 예측한다.
      x_i = data[i:i+seq_len]
      y_i = data[i+1:i+seq_len+1]
      x.append(x_i)
      y.append(y_i)

    # 회귀 모델 계산에 사용할 수 있도록 실수형 텐서로 변환한다.
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)

    return x_tensor, y_tensor

  def __getitem__(self, i):
    # DataLoader가 i번째 시계열 입력과 정답을 한 쌍으로 가져간다.
    return self.data[i], self.labels[i]

  def __len__(self):
    # 만들어진 시계열 입력 묶음의 개수가 Dataset의 길이이다.
    return len(self.data)

trainset = BitcoinDataset()
valset = BitcoinDataset(split="val")
testset = BitcoinDataset(split="test")

print(len(trainset), len(valset), len(testset))

def plot_sequence(x, y):
  plt.ylabel("Scaled Close")
  plt.xlabel("Time Bars")
  plt.plot(range(len(x)), x, color='r', label='Features')
  plt.plot(range(1, len(x)+1), y, color='b', linestyle='--', label='Target')
  plt.legend()

plt.figure(figsize=(10,2))
for i, (x, y) in enumerate(trainset):
  plt.subplot(1, 3, i+1)
  plot_sequence(x, y)

  if i == 2:
    break

# ### 1.4 LSTM 모델
#

# 각 시점의 종가 1개를 입력받아 다음 시점의 종가 1개를 예측한다.
# batch_first=True이므로 입력 모양은 (배치, 시점 4개, 특성 1개)이다.
model = nn.LSTM(1, 1, batch_first=True)
# summary()는 입력 모양을 기준으로 계층별 출력 크기와 파라미터 수를 보여준다.
summary(model, input_size=(1, 4, 1))

# LSTM은 모든 시점의 출력과 마지막 은닉 상태·셀 상태를 함께 반환한다.
# 여기서는 (배치 1개, 시점 4개, 입력 특성 1개) 형태로 동작을 확인한다.
output, (hidden_state, cell_state) = model(torch.randn((1, 4, 1)).to(device))
print(output)
print(hidden_state)
print(cell_state)

# ### 1.5 모델 학습

def train(model, dataloader, criterion, optimizer, epoch, device):
  # train()은 모델을 학습 모드로 전환한다.
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (features, targets) in enumerate(pbar):
      features = features.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      # 두 번째 반환값은 마지막 hidden/cell state이며 손실 계산에는 필요하지 않아 버린다.
      outputs, _ = model(features)
      loss = criterion(outputs, targets)

      # 역전파로 기울기를 계산하고 optimizer가 LSTM 가중치를 갱신한다.
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

def validation(model, dataloader, criterion, epoch, device):
  # eval()은 모델을 평가 모드로 전환한다.
  model.eval()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    # 검증에서는 가중치를 바꾸지 않으므로 기울기 기록을 끈다.
    with torch.no_grad():
      for i, (features, targets) in enumerate(pbar):
        features = features.to(device)
        targets = targets.to(device)

        outputs, _ = model(features)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

EPOCH = 60
BATCH_SIZE = 128
NUM_WORKERS = 2
LR = 0.1

trainset = BitcoinDataset()
valset = BitcoinDataset(split="val")
testset = BitcoinDataset(split="test")

train_loader = DataLoader(
  dataset=trainset,
  # Dataset의 시계열 샘플을 최대 128개씩 묶어 모델에 전달한다.
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
model = nn.LSTM(1, 1, batch_first=True)

# Optimizer, Loss, Scheduler
# Adam은 model.parameters()에 들어 있는 학습 가능한 가중치를 갱신한다.
optimizer = optim.Adam(model.parameters(), lr=LR)
# MSELoss는 예측 가격과 정답 가격의 제곱 오차 평균을 계산한다.
criterion = nn.MSELoss()
# StepLR은 20 Epoch마다 학습률에 0.2를 곱해 점차 작게 만든다.
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

  # 검증 손실이 가장 낮을 때의 비트코인 예측 가중치만 저장한다.
  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    torch.save(model.state_dict(), "bitcoin_best.pth")
    min_loss = vloss

# ### 1.6 추론

def test(model, dataloader, device):
  model.eval()

  predicts = []

  with tqdm(dataloader) as pbar:
    # 추론에서는 역전파가 필요 없으므로 기울기 기록을 끈다.
    with torch.no_grad():
      for i, (features, _) in enumerate(pbar):
        features = features.to(device)

        outputs, _ = model(features)
        # 각 시계열에서 마지막 시점의 예측만 모아 다음 가격 예측값으로 사용한다.
        predicts.append(outputs[:, -1])

  # 여러 배치의 예측을 첫 번째 차원으로 이어 하나의 텐서로 만든다.
  return torch.cat(predicts)

def show_plot(model, dataloader, device):
  # Dataset이 정규화할 때 저장한 scaler를 사용해야 같은 가격 단위로 복원된다.
  scaler = dataloader.dataset.scaler

  actual = dataloader.dataset.data[:, -1]
  predicts = test(model, dataloader, device).cpu().numpy()

  df = pd.DataFrame({"actual": actual, "predicted": predicts})
  # 실제값과 예측값 두 열에 같은 역정규화를 적용해 원래 원화 가격으로 되돌린다.
  df = df.apply(
    lambda x: scaler.inverse_transform(np.array(x).reshape(-1, 1)).reshape((-1))
  )
  df.plot(figsize=(14,5))

model = nn.LSTM(1, 1, batch_first=True).to(device)
# CPU에서도 GPU 저장 모델을 읽을 수 있도록 현재 장치로 위치를 변환해 불러온다.
model.load_state_dict(torch.load("bitcoin_best.pth", map_location=device))

show_plot(model, train_loader, device)

show_plot(model, val_loader, device)

show_plot(model, test_loader, device)
