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

# torch.linspace(): 시작값부터 끝값까지 일정한 간격으로 지정한 개수만큼 텐서를 만든다.
# -π와 π를 모두 포함하여 총 1,000개의 x 좌표를 생성한다.
x = torch.linspace(-math.pi, math.pi, 1000)

# torch.sin(): 텐서의 모든 값에 사인 함수를 각각 적용한다.
# x 좌표 1,000개에 대응하는 정답 y값을 생성한다.
y = torch.sin(x)

# 결과:
# x: -π부터 π까지 일정한 간격으로 배치된 1차원 텐서
# y: 각 x에 대한 sin(x) 값이 들어 있는 1차원 텐서
# x와 y의 원소 개수는 각각 1,000개이다.

# %% [markdown]
# 이번에 사용할 3차 다항식 모델은 다음과 같이 표현이 가능합니다.
#
# $ax^{3}+bx^{2}+cx+d=0,a\neq 0$
#
# 이를 식으로 표현해 보겠습니다.

# %%
# torch.randn(()): 평균 0, 표준편차 1인 정규분포에서 임의의 숫자 하나를 생성한다.
# 빈 튜플 ()을 전달했기 때문에 원소 하나만 가진 0차원 스칼라 텐서가 만들어진다.
# 실행할 때마다 값이 달라질 수 있다.
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# a, b, c, d는 학습을 통해 조정할 3차 다항식의 가중치이다.
# x의 1,000개 값에 각각 ax³ + bx² + cx + d를 적용한다.
y_random = (a * x**3) + (b * x**2) + (c * x) + d

# %%
import matplotlib.pyplot as plt

# plt.subplot(2, 1, 1): 그래프 영역을 2행 1열로 나누고 첫 번째 영역을 선택한다.
plt.subplot(2,1,1)

# plt.subplots_adjust(): 위아래 그래프 사이의 간격을 조절한다.
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x, y)

# 두 번째 그래프 영역을 선택한다.
plt.subplot(2,1,2)
plt.title("sin random")
plt.plot(x, y_random)

plt.show()

# %% [markdown]
# ### 1.3 3차 다항식 모델 학습하기

# %%
# learning_rate는 한 번의 학습에서 가중치를 얼마나 변경할지 결정한다.
# 1e-6은 0.000001을 의미한다.
learning_rate = 1e-6

# 학습에 사용할 입력값과 정답
x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# 학습을 시작하기 전 가중치를 무작위 값으로 초기화한다.
a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

# 전체 데이터를 2,000번 반복하여 학습한다.
for epoch in range(2000):
    # 현재 가중치로 예측값을 계산하는 순전파 과정
    y_pred = (a * x**3) + (b * x**2) + (c * x) + d

    # pow(2): 예측값과 정답의 차이를 각각 제곱한다.
    # sum(): 1,000개 데이터의 제곱 오차를 모두 더한다.
    # 평균을 구하지 않고 합계를 구하므로 정확히는 MSE가 아니라 SSE이다.
    # item(): 원소 하나인 텐서에서 파이썬 숫자를 꺼낸다.
    loss = (y_pred - y).pow(2).sum().item()

    # 예측값에 대한 손실의 기울기
    grad_y_pred = 2.0 * (y_pred - y)

    # 연쇄법칙을 이용해 각 가중치에 대한 기울기를 직접 계산한다.
    # sum()은 1,000개 데이터에서 발생한 기울기를 하나로 합친다.
    grad_a = (grad_y_pred * x**3).sum()
    grad_b = (grad_y_pred * x**2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    # 기울기의 반대 방향으로 가중치를 조금씩 이동시킨다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

    # 100번 학습할 때마다 현재 반복 횟수와 오차를 출력한다.
    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", loss)

# %%
# a는 x³, b는 x², c는 x, d는 상수항에 곱해지는 학습된 가중치이다.
# 무작위 값에서 시작하기 때문에 실행할 때마다 최종 값이 조금씩 달라질 수 있다.
print(a, b, c, d)

# %%
# 셀 6: 실제 사인 함수와 학습된 3차 다항식 비교

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x, y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

plt.show()

# %% [markdown]
# 랜덤으로 초기화 했을때 보다 확실히 비슷하게 예측하고 있는것을 확인할 수 있습니다.
#
# 학습 가중치 하나하나마다 미분도하고 업데이트도 해주고 해야할 일이 너무 많다구요? 학습할 가중치가 1조개가 되면 이걸 다 어떻게 하냐구요? 맞습니다. 1조개 가중치를 하나하나 한다면 1조줄 짜리 코드가 생기겠죠. 걱정하지마세요. 파이토피에서는 자동 미분과 최적화기를 통해 기울기를 구하는 과정과 가중치를 업데이트 하는 과정을 편리하게 할 수 있도록 제공하고 있습니다. 다음 부터 하나씩 적용해가며 어떻게 사용하는지 알아보겠습니다.

# %% [markdown]
# ### 1.4 자동 미분(Auto-grad)

# %%
learning_rate = 1e-6

x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# requires_grad=True: 예측값과 손실을 계산하는 과정을 기록한다.
# backward()를 실행하면 이 기록을 이용해 각 가중치의 기울기를 구할 수 있다.
a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

for epoch in range(2000):
    y_pred = (a * x**3) + (b * x**2) + (c * x) + d
    loss = (y_pred - y).pow(2).sum()

    # backward(): 기록된 계산 과정을 역으로 따라가며 기울기를 계산한다.
    loss.backward()

    # torch.no_grad(): 기울기로 가중치의 값은 변경하지만 그 변경 과정은 기록하지 않는다.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # 기울기는 backward()를 실행할 때마다 누적되므로 다음 반복 전에 제거한다.
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

    # 다음 반복에서는 변경된 가중치로 새로운 계산 과정을 다시 기록한다.
    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", loss.item())

# %%
# detach(): y_pred를 자동 미분 계산 기록에서 분리한다.
# NumPy 배열은 CPU 메모리만 사용하므로 GPU 텐서는 먼저 cpu()로 CPU에 옮겨야 한다.
# 현재 y_pred는 CPU 텐서이므로 cpu() 없이 바로 NumPy 배열로 변환할 수 있다.
# GPU 텐서라면 y_pred.detach().cpu().numpy() 순서로 변환한다.
y_pred = y_pred.detach().numpy()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x, y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

plt.show()

# %% [markdown]
# ### 1.5 파이토치 모듈 사용하기
#
# 파이토치는 모델을 nn.Module을 상속받는 클래스 형태로 정의하여 사용하는 스타일을 권장하고 있습니다. 앞서 사용한 3차 다항식모델을 nn.Module을 이용한 모델로 만들어 보겠습니다.

# %%
import torch
import torch.nn as nn

# nn.Module은 PyTorch 모델을 만들 때 상속받는 기본 클래스이다.
# 상속하면 가중치 관리, 자동 미분, 장치 이동 등의 기능을 사용할 수 있다.
class LegendrePolynomial3(nn.Module):
    def __init__(self):
        # super().__init__(): 부모 클래스인 nn.Module의 초기화 기능을 실행한다.
        # 이 코드가 있어야 PyTorch가 모델 내부의 가중치를 정상적으로 등록할 수 있다.
        super().__init__()

        # nn.Parameter(): 텐서를 모델이 학습해야 하는 공식 가중치로 등록한다.
        # Parameter는 기본적으로 requires_grad=True이므로 기울기가 자동 계산된다.
        # 일반 텐서에 requires_grad=True만 지정해도 기울기는 계산되지만
        # model.parameters()와 state_dict()에는 자동으로 포함되지 않는다.
        # Parameter는 모델 저장·불러오기, 최적화기 연결, CPU·GPU 장치 이동에 포함된다.
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    # forward(): 입력값 x를 받았을 때 모델이 예측값을 계산하는 방법을 정의한다.
    # model(x)를 실행하면 PyTorch가 내부적으로 forward(x)를 호출한다.
    def forward(self, x):
        return (
            (self.a * x**3)
            + (self.b * x**2)
            + (self.c * x)
            + self.d
        )

# %%
learning_rate = 1e-6

x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

# LegendrePolynomial3은 이전 셀에서 정의한 모델 클래스이다.
# 뒤에 ()를 붙이면 __init__()이 실행되어 실제 모델 객체가 만들어진다.
# 생성된 모델에는 무작위로 초기화된 a, b, c, d Parameter가 들어 있다.
model = LegendrePolynomial3()

for epoch in range(2000):
    # model(x)를 실행하면 모델의 forward(x)가 내부적으로 호출된다.
    y_pred = model(x)

    # nn.MSELoss(): 예측값과 정답 사이의 평균 제곱 오차를 계산한다.
    # reduction="sum"이므로 평균 대신 모든 제곱 오차를 더한 SSE가 된다.
    loss = nn.MSELoss(reduction="sum")(y_pred, y)

    loss.backward()

    # Parameter도 일반 requires_grad 텐서처럼 grad에 기울기가 저장된다.
    with torch.no_grad():
        model.a -= learning_rate * model.a.grad
        model.b -= learning_rate * model.b.grad
        model.c -= learning_rate * model.c.grad
        model.d -= learning_rate * model.d.grad

        model.a.grad = None
        model.b.grad = None
        model.c.grad = None
        model.d.grad = None

    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", loss.item())

# %%
# squeeze(): 크기가 1인 차원을 제거한다.
# 예를 들어 (1000, 1) 모양의 텐서를 (1000,) 모양으로 바꾼다.
# 현재 x와 y는 이미 (1000,)이므로 실제 모양은 바뀌지 않는다.
# squeeze()는 크기가 1인 모든 차원을 제거하므로 배치 크기가 1일 때는
# squeeze(dim=1)처럼 제거할 차원을 지정하는 편이 안전하다.
x = x.squeeze()
y_pred = model(x).detach().squeeze().numpy()
y = y.squeeze()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x, y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

plt.show()

# %% [markdown]
# ### 1.6 최적화 함수

# %%
learning_rate = 1e-6

x = torch.linspace(-math.pi, math.pi, 1000)
y = torch.sin(x)

model = LegendrePolynomial3()

# 손실 함수 객체를 미리 만들어 반복해서 사용한다.
criterion = nn.MSELoss(reduction="sum")

# model.parameters(): 모델에 Parameter로 등록된 모든 가중치를 가져온다.
# 이 모델에서는 a, b, c, d가 출력된다.
print(list(model.parameters()))

# torch는 PyTorch 패키지이고 optim은 여러 최적화기를 모아둔 모듈이다.
# SGD는 optim 모듈이 제공하는 경사하강법 최적화기 클래스이다.
# 뒤에 ()를 붙이면 모델 가중치와 학습률을 관리할 최적화기 객체가 만들어진다.
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=learning_rate,
)

for epoch in range(2000):
    # zero_grad(): 이전 반복에서 계산된 모든 Parameter의 기울기를 제거한다.
    optimizer.zero_grad()

    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()

    # step(): 각 가중치에 대해 가중치 = 가중치 - 학습률 × 기울기를 수행한다.
    # 최적화기가 업데이트하므로 직접 torch.no_grad()에서 가중치를 수정할 필요가 없다.
    optimizer.step()

    if epoch % 100 == 0:
        print("epoch:", epoch, "loss:", loss.item())

# %%
x = x.squeeze()

# clone(): 텐서의 데이터를 별도 메모리에 복사해 독립된 텐서를 만든다.
# 이 셀에서는 예측값을 수정하지 않으므로 clone() 없이 detach()만 사용해도 된다.
y_pred = model(x).detach().clone().squeeze().numpy()

y = y.squeeze()

plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.3)
plt.title("sin ground truth")
plt.plot(x, y)

plt.subplot(2,1,2)
plt.title("sin pred")
plt.plot(x, y_pred)

plt.show()

