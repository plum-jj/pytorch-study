# %%
# Converted from: Chapter 2 당뇨병 진행도 예측하기 copy.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # Part 2. 딥러닝 몸풀기
# ## Chapter 2. 당뇨병 진행도 예측하기

# %% [markdown]
# ### 2.1 데이터 살펴보기
# 당뇨병 진행도 데이터셋은 머신러닝 라이브러리중 하나인 scikit-learn에서 제공하고 있습니다. 다음과 같이 불러와 사용할 수 있습니다.

# %%
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
print(dataset.keys())

# %%
print(dataset.DESCR)

# %%
import pandas as pd

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

print(df.head())

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
for i in range(10):
  plt.subplot(3, 4, i+1)
  plt.scatter(df.iloc[:, i], df['target'])
  plt.title(df.columns[i])
plt.show()

# %% [markdown]
# 각 피쳐값과 타겟값의 상관성을 보기위해 분산형 차트로 데이터셋을 나타내 보았습니다.

# %% [markdown]
# ### 2.2 모델 구현 및 학습

# %%
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

X = torch.FloatTensor(df.iloc[:, :10].values)
Y = torch.FloatTensor(df['target'].values).reshape(-1, 1)

batch_size = 64
learning_rate = 1e-4
epochs = 30000

optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):

  # 배치 설명
  for i in range(len(X) // batch_size):
    start = i*batch_size
    end = start + batch_size

    x = X[start:end]
    y = Y[start:end]

    optimizer.zero_grad()

    pred_y = model(x)

    loss = nn.MSELoss()(pred_y, y)
    loss.backward()
    optimizer.step()

  if epoch % 2000 == 0:
    print("epoch:", epoch, "loss:", loss.item())

# %% [markdown]
# ### 2.3 모델 성능 평가

# %%
import matplotlib.pyplot as plt

pred = model(torch.FloatTensor(X))
plt.scatter(Y.numpy(), pred.reshape(1, -1).clone().detach().numpy())

# %%
pred = model(torch.FloatTensor(X))

mae = (pred - Y).abs().mean()
rmse = ((pred - Y) ** 2).mean().sqrt()

print("MAE:", mae.item(), "RMSE:", rmse.item())

# %%
untrained_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

pred = untrained_model(torch.FloatTensor(X))
plt.scatter(Y.numpy(), pred.reshape(1, -1).clone().detach().numpy())

pred = untrained_model(torch.FloatTensor(X))

mae = (pred - Y).abs().mean()
rmse = ((pred - Y) ** 2).mean().sqrt()

print("MAE:", mae.item(), "RMSE:", rmse.item())

