# %%
# Converted from: Chapter 1 사인 함수 예측하기 copy.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # Part 2. 딥러닝 몸풀기
#

# %% [markdown]
# ## Chapter 1. 사인 함수 예측하기

# %% [markdown]
# ### 1.2 3차 다항식 모델 정의하기

# %%
import math
import torch

x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# %% [markdown]
# 이번에 사용할 3차 다항식 모델은 다음과 같이 표현이 가능합니다.
#
# $ax^{3}+bx^{2}+cx+d=0,a\neq 0$
#
# 이를 식으로 표현해 보겠습니다.

# %%
#초기화
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# 모델
y_random = (a * x**3) + (b * x**2) + (c * x) + d

# %%
import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("cos ground truth")
plt.plot(x,y)

plt.subplot(2,1,2)
plt.title("cos random")
plt.plot(x, y_random)

plt.show()

# %% [markdown]
# ### 1.3 3차 다항식 모델 학습하기

# %%
# 하이퍼파라미터
learning_rate = 1e-6

# 데이터셋
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 모델 가중치 초기화
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

for epoch in range(2000):
  # 순전파
  y_pred = (a * x**3) + (b * x**2) + (c * x) + d

  # 오차 계산 (MSE)
  loss = (y_pred - y).pow(2).sum().item()

  # 백워드, 미분값(gradient) 계산
  grad_y_pred = 2.0 * (y_pred - y)
  grad_a = (grad_y_pred * x ** 3).sum()
  grad_b = (grad_y_pred * x ** 2).sum()
  grad_c = (grad_y_pred * x).sum()
  grad_d = grad_y_pred.sum()

  # 가중치 업데이트
  a -= learning_rate * grad_a
  b -= learning_rate * grad_b
  c -= learning_rate * grad_c
  d -= learning_rate * grad_d

  # 출력
  if epoch % 100 == 0:
    print("epoch: ", epoch, "loss: ", loss)

# %%
print(a, b, c, d)

# %%
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x,y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

# %% [markdown]
# 랜덤으로 초기화 했을때 보다 확실히 비슷하게 예측하고 있는것을 확인할 수 있습니다.
#
# 학습 가중치 하나하나마다 미분도하고 업데이트도 해주고 해야할 일이 너무 많다구요? 학습할 가중치가 1조개가 되면 이걸 다 어떻게 하냐구요? 맞습니다. 1조개 가중치를 하나하나 한다면 1조줄 짜리 코드가 생기겠죠. 걱정하지마세요. 파이토피에서는 자동 미분과 최적화기를 통해 기울기를 구하는 과정과 가중치를 업데이트 하는 과정을 편리하게 할 수 있도록 제공하고 있습니다. 다음 부터 하나씩 적용해가며 어떻게 사용하는지 알아보겠습니다.

# %% [markdown]
# ### 1.4 자동 미분(Auto-grad)

# %%
# 하이퍼파라미터
learning_rate = 1e-6

# 데이터셋
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 모델 가중치 초기화
a = torch.tensor(torch.randn(()), requires_grad=True)
b = torch.tensor(torch.randn(()), requires_grad=True)
c = torch.tensor(torch.randn(()), requires_grad=True)
d = torch.tensor(torch.randn(()), requires_grad=True)

for epoch in range(2000):
  # 순전파
  y_pred = (a * x**3) + (b * x**2) + (c * x) + d

  # 오차 계산 (MSE)
  loss = (y_pred - y).pow(2).sum()

  # 백워드
  loss.backward()

  # 가중치 업데이트
  with torch.no_grad():
    a -= learning_rate * a.grad
    b -= learning_rate * b.grad
    c -= learning_rate * c.grad
    d -= learning_rate * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

  # 출력
  if epoch % 100 == 0:
    print("epoch: ", epoch, "loss: ", loss.item())

# %%
y_pred = y_pred.detach().numpy()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x,y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

# %% [markdown]
# ### 1.5 파이토치 모듈 사용하기
#
# 파이토치는 모델을 nn.Module을 상속받는 클래스 형태로 정의하여 사용하는 스타일을 권장하고 있습니다. 앞서 사용한 3차 다항식모델을 nn.Module을 이용한 모델로 만들어 보겠습니다.

# %%
import torch
import torch.nn as nn

class LegendrePolynomial3(nn.Module):
  def __init__(self):
    super().__init__()

    self.a = nn.Parameter(torch.randn(()))
    self.b = nn.Parameter(torch.randn(()))
    self.c = nn.Parameter(torch.randn(()))
    self.d = nn.Parameter(torch.randn(()))

  def forward(self, x):
    return (self.a * x ** 3) + (self.b * x ** 2) + (self.c * x) + self.d

# %%
# 하이퍼파라미터
learning_rate = 1e-6

# 데이터셋
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 모델 초기화
model = LegendrePolynomial3()

for epoch in range(2000):
  # 순전파
  y_pred = model(x)

  # 오차 계산 (MSE)
  loss = nn.MSELoss(reduction='sum')(y_pred, y) # SSE, Sum Square Error

  # 백워드
  loss.backward()

  # 가중치 업데이트
  with torch.no_grad():
    model.a -= learning_rate * model.a.grad
    model.b -= learning_rate * model.b.grad
    model.c -= learning_rate * model.c.grad
    model.d -= learning_rate * model.d.grad

    model.a.grad = None
    model.b.grad = None
    model.c.grad = None
    model.d.grad = None

  # 출력
  if epoch % 100 == 0:
    print("epoch: ", epoch, "loss: ", loss.item())

# %%
x = x.squeeze()
y_pred = model(x).detach().squeeze().numpy()
y = y.squeeze()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x,y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

# %% [markdown]
# ### 1.6 최적화 함수

# %%
# 하이퍼파라미터
learning_rate = 1e-6

# 데이터셋
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 모델 초기화
model = LegendrePolynomial3()

# 손실함수
criterion = nn.MSELoss(reduction='sum')

# 최적화 함수
print(list(model.parameters()))
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

for epoch in range(2000):
  # 미분값 초기화
  optimizer.zero_grad()

  # 순전파
  y_pred = model(x)

  # 오차 계산
  loss = criterion(y_pred, y)

  # 백워드
  loss.backward()

  # 가중치 업데이트
  optimizer.step()

  # 출력
  if epoch % 100 == 0:
    print("epoch: ", epoch, "loss: ", loss.item())

# %%
x = x.squeeze()
y_pred = model(x).clone().detach().squeeze().numpy()
y = y.squeeze()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x,y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

