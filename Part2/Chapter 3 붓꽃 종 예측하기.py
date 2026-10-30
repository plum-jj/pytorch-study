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

# load_iris(): scikit-learn에 포함된 붓꽃 분류용 예제 데이터를 불러온다.
# 데이터는 붓꽃 150개의 특성 4개와 붓꽃 종류 3개로 구성되어 있다.
# 반환값은 앞에서 사용한 당뇨병 데이터와 마찬가지로 Bunch 객체이다.
dataset = load_iris()

# .py 파일에서도 결과가 출력되도록 print()를 사용한다.
print(dataset.keys())

# %%
# DESCR에는 표본 수, 특성 이름, 단위, 클래스 구성과 데이터 출처가 들어 있다.
# 입력 특성은 꽃받침 길이·너비와 꽃잎 길이·너비의 총 4개이다.
# 예측 대상은 setosa, versicolor, virginica의 세 종류이다.
print(dataset.DESCR)

# %%
import pandas as pd

# dataset.data는 붓꽃 150개의 특성 4개를 가진 (150, 4) 배열이다.
# feature_names를 열 이름으로 지정해 표를 만든다.
data = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names,
)

# target은 붓꽃 종류를 숫자로 표현한 정답이다.
# 0은 setosa, 1은 versicolor, 2는 virginica를 의미한다.
data["target"] = dataset.target

print(data.head())

# %% [markdown]
# ### 3.2 파이토치 데이터 유틸 사용하기

# %%
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Dataset을 상속하면 DataLoader가 데이터 개수와 개별 데이터를 가져올 수 있다.
class IrisDataset(Dataset):
    # train=True이면 학습 데이터, False이면 평가 데이터를 사용한다.
    def __init__(self, train=True):
        dataset = load_iris()

        # train_test_split(): 데이터를 학습용과 평가용으로 나눈다.
        # test_size=0.3이므로 150개 중 105개는 학습용, 45개는 평가용이다.
        # random_state를 고정하면 실행할 때마다 같은 방식으로 분리된다.
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=0.3,
            random_state=827,
        )

        if train:
            # train=True이면 학습용 데이터 105개를 저장한다.
            # X_train의 모양은 (105, 4)이며 붓꽃별 특성 4개가 들어 있다.
            # self.data로 저장하면 다른 메서드에서도 이 데이터를 사용할 수 있다.
            # Linear 계층과 계산할 수 있도록 float32 텐서로 변환한다.
            self.data = torch.FloatTensor(X_train)

            # y_train의 모양은 (105,)이며 값은 클래스 번호 0, 1, 2이다.
            # 0은 setosa, 1은 versicolor, 2는 virginica를 의미한다.
            # CrossEntropyLoss는 int64 클래스 번호를 정답으로 받으므로
            # LongTensor로 변환하며 원-핫 인코딩은 하지 않는다.
            self.target = torch.LongTensor(y_train)
        else:
            # train=False이면 평가용 데이터 45개를 저장한다.
            # X_test의 모양은 (45, 4), y_test의 모양은 (45,)이다.
            self.data = torch.FloatTensor(X_test)
            self.target = torch.LongTensor(y_test)

    # __getitem__(): 지정된 위치의 입력 데이터와 정답을 한 쌍으로 반환한다.
    # DataLoader가 배치를 만들 때 이 함수를 반복해서 호출한다.
    def __getitem__(self, i):
        return self.data[i], self.target[i]

    # __len__(): 현재 Dataset에 들어 있는 전체 데이터 개수를 반환한다.
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

batch_size = 64
learning_rate = 1e-3
epochs = 2000

# 입력 특성 4개를 받아 붓꽃 종류 3개에 대한 점수를 출력한다.
# 마지막 출력 3개는 각각 setosa, versicolor, virginica의 점수이다.
model = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

train_dataset = IrisDataset(train=True)

# DataLoader(): Dataset에서 데이터를 가져와 배치 단위로 묶어준다.
# shuffle=True이면 epoch마다 데이터 순서를 무작위로 섞는다.
# 학습 데이터는 105개이므로 64개와 41개의 배치 두 개가 만들어지며
# 마지막 41개는 패딩하지 않고 그대로 사용한다.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Adam은 PyTorch가 제공하는 최적화기로 가중치마다 학습 속도를 조절한다.
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=learning_rate,
)

# CrossEntropyLoss(): 여러 클래스 중 정답 클래스를 맞히도록 학습하는 손실 함수이다.
# 모델 출력은 (배치 크기, 3), 정답은 클래스 번호를 담은 (배치 크기,) 모양이다.
# 내부에서 Softmax에 해당하는 계산을 처리하므로 모델 뒤에 Softmax를 추가하지 않는다.
criterion = nn.CrossEntropyLoss()

model.train()

for epoch in range(epochs):
    for features, target in train_dataloader:
        optimizer.zero_grad()

        pred = model(features)
        loss = criterion(pred, target)

        loss.backward()
        optimizer.step()

    # 100, 200, 300번째처럼 100번 학습할 때마다 출력한다.
    if epoch % 100 == 99:
        print(
            "epoch:",
            epoch + 1,
            "loss:",
            loss.item(),
        )

# %% [markdown]
# ### 3.4 모델 성능 평가

# %%
test_dataset = IrisDataset(train=False)

# 평가 데이터는 45개이고 batch_size는 64이므로 하나의 배치로 처리된다.
# shuffle=False로 데이터 순서를 그대로 유지한다.
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

model.eval()

num_correct = 0

# 블록 안의 예측 계산은 정상적으로 실행하지만 자동 미분 기록은 만들지 않는다.
# 평가 중에는 backward()나 가중치 업데이트를 하지 않으므로 기울기가 필요 없다.
# 계산 그래프를 생략해 메모리 사용량을 줄이고 평가 계산을 가볍게 한다.
with torch.no_grad():
    for features, target in test_dataloader:
        # output의 모양은 (45, 3)이며 각 붓꽃 종류에 대한 점수가 들어 있다.
        output = model(features)

        # torch.max(output, dim=1)은 각 행에서 가장 큰 값과 그 위치를 반환한다.
        # 가장 큰 값 자체는 사용하지 않아 _에 저장하고 위치를 예측 클래스로 사용한다.
        _, pred = torch.max(output, dim=1)

        # 예측과 정답이 같은 True의 개수를 세어 맞힌 개수를 구한다.
        correct = pred.eq(target).sum().item()
        num_correct += correct

# 맞힌 개수를 전체 평가 데이터 45개로 나누고 100을 곱해 백분율로 만든다.
accuracy = (num_correct / len(test_dataset)) * 100

print("Accuracy:", accuracy, "%")

