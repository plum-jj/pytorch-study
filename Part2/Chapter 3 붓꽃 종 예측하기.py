# %%
# Converted from: Chapter 3 붓꽃 종 예측하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # Part 2. 딥러닝 몸풀기
# ## Chapter 3. 붓꽃 종 예측하기

# %% [markdown]
# ### 3.1 데이터 살펴보기

# %%
from sklearn.datasets import load_iris

dataset = load_iris()
dataset.keys()

# %%
print(dataset.DESCR)

# %%
import pandas as pd

data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset.target

print(data.head())

# %% [markdown]
# ### 3.2 파이토치 데이터 유틸 사용하기

# %%
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class IrisDataset(Dataset):
  def __init__(self, train=True):
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3, random_state=827
    )
    if train:
      self.data = torch.FloatTensor(X_train)
      self.target = torch.LongTensor(y_train)
    else:
      self.data = torch.FloatTensor(X_test)
      self.target = torch.LongTensor(y_test)

  def __getitem__(self, i):
    return self.data[i], self.target[i]

  def __len__(self):
    return len(self.data)

# %% [markdown]
# ### 3.3 모델 구현 및 학습
#
# 이어서 학습코드를 살펴보겠습니다

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 하이퍼파라미터
batch_size = 64
learning_rate = 1e-3
epochs = 2000

# 모델
model = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

# 데이터셋, 데이터로더
train_dataset = IrisDataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# loss
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):

  # 배치
  for data, target in train_dataloader:
    optimizer.zero_grad()

    pred = model(data)

    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

  # 학습 기록
  if epoch % 100 == 99:
    print("epoch:", epoch+1, "loss:", loss.item())

# %% [markdown]
# ### 3.4 모델 성능 평가

# %%
test_dataset = IrisDataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_correct = 0

with torch.no_grad():
  for data, target in test_dataloader:
    output = model(data)
    pred = torch.max(output, 1)[1]

    corr = pred.eq(target).sum().item()
    num_correct += corr

  print("Accuracy:", (num_correct/len(test_dataset.data))*100, "%")

