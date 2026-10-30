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

# load_diabetes(): scikit-learn에 포함된 당뇨병 진행도 예제 데이터를 불러온다.
# 인터넷에서 새로 다운로드하는 것이 아니라 라이브러리에 포함된 데이터를 사용한다.
# 반환되는 dataset은 데이터와 설명을 이름별로 묶어 보관하는 Bunch 객체이다.
# Bunch는 딕셔너리처럼 dataset["data"]로 접근할 수 있고
# 일반 객체처럼 dataset.data로도 같은 값에 접근할 수 있다.
# Bunch 자체가 학습 데이터는 아니며 내부의 data와 target을 꺼내 사용한다.
dataset = load_diabetes()

# keys(): dataset에 어떤 종류의 정보가 들어 있는지 키 목록을 확인한다.
# data에는 입력 데이터, target에는 예측할 진행도, feature_names에는 특성 이름이 들어 있다.
print(dataset.keys())

# %%
# DESCR에는 데이터의 출처, 표본 수, 특성 정보와 각 특성의 의미가 문자열로 저장되어 있다.
# Bunch 객체이므로 dataset["DESCR"]처럼 키로 접근하거나 dataset.DESCR처럼 속성으로 접근할 수 있다.
print(dataset.DESCR)

# %%
import pandas as pd

# dataset.data에는 환자 442명의 특성 10개가 NumPy 배열로 들어 있다.
# dataset.feature_names에는 각 특성에 대응하는 열 이름 10개가 들어 있다.
# 두 값을 함께 전달해 행 442개, 열 10개의 표를 만든다.
df = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names,
)

# dataset.target에는 각 환자의 당뇨병 진행도 정답이 들어 있다.
# target이라는 새로운 열을 만들어 기존 특성 10개 옆에 정답을 추가한다.
df["target"] = dataset.target

# head(): DataFrame의 가장 위에 있는 행 5개를 반환한다.
print(df.head())

# %%
import matplotlib.pyplot as plt

# plt.figure(): 여러 그래프를 올려놓을 새로운 전체 도화지인 Figure를 만든다.
# figsize=(20, 10)은 각 그래프가 아니라 전체 Figure의 크기를 가로 20인치,
# 세로 10인치로 지정한다.
plt.figure(figsize=(20,10))

# 하나의 Figure를 3행 4열, 총 12칸으로 나누고 그중 10칸을 사용한다.
for i in range(10):
    plt.subplot(3, 4, i+1)

    # df.iloc[:, i]는 모든 행에서 i번째 열을 위치 기준으로 선택한다.
    # i가 0부터 9까지이므로 target을 제외한 입력 특성 10개만 차례로 사용한다.
    plt.scatter(df.iloc[:, i], df["target"])
    plt.title(df.columns[i])

plt.show()

# %% [markdown]
# 각 피쳐값과 타겟값의 상관성을 보기위해 분산형 차트로 데이터셋을 나타내 보았습니다.

# %% [markdown]
# ### 2.2 모델 구현 및 학습

# %%
import torch
import torch.nn as nn

# nn.Sequential(): 여러 신경망 계층을 작성된 순서대로 연결한다.
# 입력 특성 10개 → 은닉값 64개 → 은닉값 16개 → 예측값 1개로 전달된다.
model = nn.Sequential(
    # nn.Linear(): 입력값에 가중치를 곱하고 편향을 더하는 선형 계층이다.
    nn.Linear(10, 64),

    # nn.ReLU(): 음수는 0으로, 양수는 그대로 통과시키는 활성화 함수이다.
    nn.ReLU(),

    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

# 모든 행에서 입력 특성 10개를 선택하면 (442, 10) 모양의 DataFrame이 된다.
# values로 NumPy 배열을 꺼낸 뒤 FloatTensor로 float32 텐서로 변환한다.
X = torch.FloatTensor(df.iloc[:, :10].values)

# target 열은 처음에 (442,) 모양이며 reshape(-1, 1)로 (442, 1)이 된다.
# -1은 데이터 개수 442를 PyTorch가 자동으로 계산하라는 의미이다.
Y = torch.FloatTensor(df["target"].values).reshape(-1, 1)

# 한 번에 모델에 전달할 환자 데이터 개수
batch_size = 64
learning_rate = 1e-4
epochs = 30000

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=learning_rate,
)

for epoch in range(epochs):
    # 0, 64, 128처럼 시작 위치를 옮기며 배치를 만든다.
    # 마지막에는 X[384:448]을 요청하지만 슬라이싱은 존재하는 442번째까지만 가져온다.
    # 따라서 마지막 x와 y는 각각 (58, 10), (58, 1)이며 패딩 없이 정상 처리된다.
    for start in range(0, len(X), batch_size):
        end = start + batch_size

        x = X[start:end]
        y = Y[start:end]

        optimizer.zero_grad()

        pred_y = model(x)
        loss = criterion(pred_y, y)

        loss.backward()
        optimizer.step()

    # loss는 해당 epoch의 마지막 배치에서 계산된 손실이다.
    if epoch % 2000 == 0:
        print("epoch:", epoch, "loss:", loss.item())

# %% [markdown]
# ### 2.3 모델 성능 평가

# %%
import matplotlib.pyplot as plt

# eval(): 모델을 학습 모드에서 평가 모드로 전환한다.
# 현재 모델에는 Dropout이나 BatchNorm이 없어 결과는 달라지지 않지만
# 일반적으로 모델을 평가할 때 호출한다.
model.eval()

# 평가할 때는 가중치를 학습하지 않으므로 기울기 계산 기록을 만들지 않는다.
with torch.no_grad():
    # X는 이미 FloatTensor이므로 다시 변환할 필요가 없다.
    # 모델 출력은 (442, 1)이고 squeeze(dim=1) 후에는 (442,)가 된다.
    pred = model(X).squeeze(dim=1)

# 정답도 예측값과 같은 (442,) 모양으로 맞춘다.
actual = Y.squeeze(dim=1)

plt.scatter(actual.numpy(), pred.numpy())
plt.show()

# %%
# 환자별로 예측값에서 실제 진행도를 뺀 오차를 계산한다.
# pred와 actual은 모두 (442,) 모양이므로 같은 위치끼리 계산된다.
errors = pred - actual

# MAE: 오차의 절댓값을 구한 뒤 평균을 계산한다.
# 실제값과 예측값이 평균적으로 얼마나 차이 나는지 나타낸다.
mae = errors.abs().mean()

# RMSE: 오차를 제곱하고 평균을 구한 뒤 제곱근을 계산한다.
# 큰 오차를 제곱하므로 MAE보다 큰 오차에 더 민감하다.
rmse = errors.pow(2).mean().sqrt()

print("MAE:", mae.item(), "RMSE:", rmse.item())

# %%
# 학습한 모델과 구조는 같지만 가중치가 무작위인 새 모델을 만든다.
# 최적화 과정을 실행하지 않으므로 학습 전 기준 성능을 확인할 수 있다.
untrained_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

untrained_model.eval()

with torch.no_grad():
    untrained_pred = untrained_model(X).squeeze(dim=1)

plt.scatter(actual.numpy(), untrained_pred.numpy())
plt.show()

# 학습하지 않은 모델의 MAE와 RMSE를 계산한다.
untrained_errors = untrained_pred - actual
untrained_mae = untrained_errors.abs().mean()
untrained_rmse = untrained_errors.pow(2).mean().sqrt()

print(
    "MAE:",
    untrained_mae.item(),
    "RMSE:",
    untrained_rmse.item(),
)

