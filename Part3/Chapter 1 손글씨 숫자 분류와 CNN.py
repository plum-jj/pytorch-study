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

# optim에는 SGD, Adam과 같은 PyTorch 최적화기가 들어 있다.
from torch import optim

# DataLoader는 Dataset의 데이터를 배치 단위로 가져온다.
from torch.utils.data import DataLoader

# datasets는 MNIST 같은 데이터셋을 제공하고,
# transforms는 이미지 데이터를 텐서로 변환하거나 전처리할 때 사용한다.
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# %%
# torch.cuda.is_available(): 현재 환경에서 CUDA GPU를 사용할 수 있는지 확인한다.
# NVIDIA GPU와 CUDA를 지원하는 PyTorch가 준비되어 있으면 True를 반환한다.
is_cuda = torch.cuda.is_available()

# torch.device(): 텐서와 모델을 어느 장치에서 계산할지 나타내는 장치 객체를 만든다.
# GPU를 사용할 수 있으면 cuda, 사용할 수 없으면 cpu를 선택한다.
# device 객체만 만들어서는 이동하지 않으며 이후 .to(device)를 사용해야 한다.
device = torch.device("cuda" if is_cuda else "cpu")

print(is_cuda, device)

# %%
seed = 1010

# random.seed(): 파이썬 기본 random 모듈의 난수 시드를 고정한다.
random.seed(seed)

# np.random.seed(): NumPy의 난수 시드를 고정한다.
np.random.seed(seed)

# torch.manual_seed(): PyTorch의 난수 시드를 고정한다.
torch.manual_seed(seed)

# device는 문자열이 아니라 torch.device 객체이므로 device.type으로 확인한다.
if device.type == "cuda":
    # 여러 CUDA GPU에서 사용하는 PyTorch 난수 시드를 모두 고정한다.
    torch.cuda.manual_seed_all(seed)

# 시드를 고정해도 일부 GPU 연산은 구현 방식에 따라
# 실행 결과가 완전히 동일하지 않을 수 있다.

# %% [markdown]
# ### 1.6 MNIST 데이터셋

# %%
from torch.utils.data import random_split


def get_mnist_dataloaders(batch_size, val=0.2, num_workers=0):
    # transforms.Compose(): 여러 이미지 전처리를 순서대로 묶는다.
    transform = transforms.Compose([
        # ToTensor(): 이미지를 float32 텐서로 변환한다.
        # 픽셀값은 0~255에서 0~1 범위로 바뀌고,
        # 이미지 모양은 (28, 28)에서 (1, 28, 28)이 된다.
        transforms.ToTensor(),
    ])

    # datasets.MNIST(): torchvision이 제공하는 MNIST 데이터셋을 불러온다.
    # root는 파일 저장 위치이고 train=True이면 학습용 60,000개를 사용한다.
    # 로컬에 파일이 없을 때 download=True이면 인터넷으로 내려받는다.
    # transform은 이미지를 가져올 때마다 위의 전처리를 적용한다.
    mnist_train_origin = datasets.MNIST(
        root="MNIST/",
        train=True,
        download=True,
        transform=transform,
    )

    # 검증 데이터 비율이 0.2이면 60,000개의 20%인 12,000개를 사용한다.
    val_samples = int(len(mnist_train_origin) * val)
    train_samples = len(mnist_train_origin) - val_samples

    # random_split(): 하나의 Dataset을 지정한 개수에 맞춰 무작위로 나눈다.
    # 앞에서 PyTorch 시드를 고정했으므로 같은 실행 순서에서는 분할도 재현된다.
    mnist_train, mnist_val = random_split(
        mnist_train_origin,
        (train_samples, val_samples),
    )

    # train=False이면 별도로 제공되는 테스트 데이터 10,000개를 불러온다.
    mnist_test = datasets.MNIST(
        root="MNIST/",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,

        # 마지막 배치가 batch_size보다 작으면 해당 데이터를 버린다.
        # 모든 데이터를 사용하려면 drop_last=False로 지정한다.
        drop_last=True,

        # 학습할 때는 epoch마다 데이터 순서를 섞는다.
        shuffle=True,

        # 데이터를 불러올 별도 작업 프로세스 수이다.
        # 0이면 현재 메인 프로세스에서 데이터를 불러온다.
        num_workers=num_workers,
    )

    # 검증과 테스트에서는 기본값인 shuffle=False, drop_last=False를 사용한다.
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
# LeNet은 손글씨 숫자처럼 간단한 이미지를 분류하기 위해 만들어진
# 초기 형태의 합성곱 신경망(CNN) 구조이다.
#
# 합성곱 계층에서 이미지의 특징을 찾고,
# 풀링 계층에서 데이터 크기를 줄인 다음,
# 완전연결 계층에서 이미지가 0~9 중 어떤 숫자인지 분류한다.
#
# 이 코드는 원래 LeNet-5 구조를 MNIST에 맞게 변형한 것으로,
# ReLU 활성화 함수와 최대 풀링(Max Pooling)을 사용한다.
#
# 데이터 모양의 변화:
# (1, 28, 28)
# → (6, 24, 24)
# → (6, 12, 12)
# → (16, 8, 8)
# → (16, 4, 4)
# → 256
# → 120
# → 84
# → 10개의 클래스 점수

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Conv2d(입력 채널, 출력 채널, 커널 크기):
        # 이미지의 특징을 추출하는 합성곱 계층이다.
        # MNIST는 흑백 이미지이므로 입력 채널이 1이다.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # nn.MaxPool2d(): 2×2 영역마다 가장 큰 값만 남겨
        # 이미지의 가로와 세로 크기를 절반으로 줄인다.
        self.pool = nn.MaxPool2d(2, 2)

        # nn.Linear(): 입력값에 가중치와 편향을 적용하는 완전연결 계층이다.
        # 두 번의 합성곱과 풀링을 거치면 각 이미지가
        # 16개 채널을 가진 4×4 크기가 되므로 입력 특성은 16×4×4=256개이다.
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)

        # 마지막 출력 10개는 숫자 0~9 각 클래스에 대한 점수이다.
        self.fc3 = nn.Linear(84, 10)

        # nn.ReLU(): 음수는 0으로 바꾸고 양수는 그대로 유지하는 활성화 함수이다.
        self.relu = nn.ReLU()

    def forward(self, x):
        # 입력 모양: (배치 크기, 1, 28, 28)
        x = self.relu(self.conv1(x))  # (배치 크기, 6, 24, 24)
        x = self.pool(x)              # (배치 크기, 6, 12, 12)

        x = self.relu(self.conv2(x))  # (배치 크기, 16, 8, 8)
        x = self.pool(x)              # (배치 크기, 16, 4, 4)

        # view(): 배치 차원은 유지하면서 각 이미지를 256개의 1차원 값으로 펼친다.
        # -1은 나머지 차원의 크기인 256을 PyTorch가 자동으로 계산하라는 의미이다.
        x = x.view(x.size(0), -1)     # (배치 크기, 256)

        x = self.relu(self.fc1(x))    # (배치 크기, 120)
        x = self.relu(self.fc2(x))    # (배치 크기, 84)
        x = self.fc3(x)               # (배치 크기, 10)

        # 이후 CrossEntropyLoss가 클래스 점수를 직접 사용하므로
        # 여기서는 Softmax를 적용하지 않는다.
        return x

# %% [markdown]
# ### 1.8 학습, 검증, 테스트

# %%
def train(model, dataloader, criterion, optimizer, epoch):
    # model.train(): 모델을 학습 모드로 전환한다.
    # Dropout이나 BatchNorm처럼 학습과 평가 때 동작이 다른 계층에 영향을 준다.
    model.train()

    # 현재 epoch의 전체 손실과 정답 개수를 누적한다.
    running_loss = 0
    correct = 0

    for data, targets in dataloader:
        # .to(device)는 원본 텐서를 직접 변경하지 않고
        # 해당 장치로 이동한 새 텐서를 반환하므로 변수에 다시 저장해야 한다.
        data = data.to(device)
        targets = targets.to(device)

        # 이전 반복에서 계산된 기울기를 제거한다.
        optimizer.zero_grad()

        # 모델에 이미지를 입력하여 숫자 0~9의 클래스 점수를 계산한다.
        outputs = model(data)

        # 예측 점수와 실제 정답을 비교하여 손실을 계산한다.
        loss = criterion(outputs, targets)

        # 역전파로 각 가중치의 기울기를 계산한다.
        loss.backward()

        # 계산된 기울기를 이용해 모델의 가중치를 변경한다.
        optimizer.step()

        # loss.item(): 텐서에 들어 있는 손실값을 파이썬 숫자로 가져온다.
        running_loss += loss.item()

        # torch.max(outputs, dim=1):
        # 각 이미지의 10개 클래스 점수 중 가장 큰 값과 그 위치를 반환한다.
        # 값은 사용하지 않고 위치인 predicted만 사용한다.
        _, predicted = torch.max(outputs, dim=1)

        # view_as(): targets의 모양을 predicted와 동일하게 맞춘다.
        # eq(): 예측값과 정답이 같은 위치를 True로 만들고,
        # sum()으로 True의 개수를 세어 이번 배치의 정답 개수를 구한다.
        correct += predicted.eq(targets.view_as(predicted)).sum().item()

    data_num = len(dataloader.dataset)
    acc = 100.0 * correct / data_num

    print(
        f"[{epoch}/{EPOCH}]",
        f"train loss: {running_loss / len(dataloader):.4f}",
        f"train acc: {correct}/{data_num} ({acc:.2f}%)",
    )

    # 전체 배치에서 발생한 손실의 평균을 반환한다.
    return running_loss / len(dataloader)

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
    # 테스트는 검증과 마찬가지로 가중치를 변경하지 않는다.
    # 검증 데이터는 학습 중 모델 선택에 사용하지만,
    # 테스트 데이터는 학습이 끝난 모델의 최종 성능 확인에만 사용한다.
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()

    data_num = len(dataloader.dataset)
    accuracy = 100.0 * correct / data_num
    print(f"Test Accuracy: {correct}/{data_num} ({accuracy:.2f}%)")

# %%
# 하이퍼파라미터는 학습으로 자동 결정되지 않고 사용자가 미리 정하는 값이다.
# EPOCH은 전체 학습 데이터를 반복해서 학습할 횟수이다.
EPOCH = 5
# BATCH_SIZE는 한 번의 가중치 갱신에 사용할 이미지 개수이다.
BATCH_SIZE = 32
# NUM_WORKERS는 DataLoader가 데이터를 준비할 작업 프로세스 개수이다.
NUM_WORKERS = 1
# LR은 한 번의 가중치 갱신 크기를 조절하는 학습률이다.
LR = 0.001

# 모델도 데이터와 같은 장치에 있어야 계산할 수 있다.
model = LeNet().to(device)
print(model)

# 학습용 80%, 검증용 20%, 별도의 테스트 데이터를 배치 단위로 준비한다.
train_loader, val_loader, test_loader = get_mnist_dataloaders(
    BATCH_SIZE,
    val=0.2,
    num_workers=NUM_WORKERS,
)

# CrossEntropyLoss는 다중 클래스 분류에서 클래스 점수와 정답 번호를 비교한다.
criterion = nn.CrossEntropyLoss()

# Adam은 각 매개변수의 학습률을 적응적으로 조절하는 최적화 알고리즘이다.
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(EPOCH):
    tloss = train(model, train_loader, criterion, optimizer, epoch + 1)
    vloss = validation(model, val_loader, criterion, epoch + 1)

    # epoch별 평균 손실을 저장해 이후 학습 추이를 확인한다.
    train_losses.append(tloss)
    val_losses.append(vloss)

test(model, test_loader)

# %%
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.plot(val_losses)
