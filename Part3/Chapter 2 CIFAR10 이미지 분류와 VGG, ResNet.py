# %%
# Converted from: Chapter 2 CIFAR10 이미지 분류와 VGG, ResNet.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 딥러닝의 기본 이미지 분류하기
# ## Chapter 2 CIFAR10 이미지 분류와 VGG, ResNet

# %%
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

import matplotlib.pyplot as plt

# CUDA 사용 가능 여부에 따라 GPU 또는 CPU를 계산 장치로 선택한다.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# 같은 난수 결과를 재현할 수 있도록 라이브러리별 시드를 고정한다.
seed = 1010
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ### 2.1 CIFAR10 데이터셋

# %%
from torchvision import datasets

# CIFAR-10은 32×32 컬러 이미지로 구성된 10개 클래스 분류 데이터셋이다.
# train=True는 학습 데이터 50,000개, False는 테스트 데이터 10,000개를 가져온다.
trainset = datasets.CIFAR10("CIFAR", train=True, download=True)
testset = datasets.CIFAR10("CIFAR", train=False, download=True)

# %%
import pandas as pd

# targets에는 각 이미지의 정답이 0~9의 클래스 번호로 저장되어 있다.
df = pd.DataFrame(trainset.targets)

# 클래스 번호별 행을 묶은 뒤 각 클래스의 이미지 개수를 센다.
df_count = df.groupby(0)[0].count()
print(df_count)

df = pd.DataFrame(testset.targets)
df_count = df.groupby(0)[0].count()
print(df_count)

# %%
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for i in range(24):
    # subplot(3, 8, i+1): 하나의 화면을 3행 8열로 나누고
    # i+1번째 영역을 현재 이미지를 그릴 위치로 선택한다.
    plt.subplot(3, 8, i + 1)
    img, label = trainset[i]

    plt.gca().set_title(classes[label])
    plt.imshow(img)
    plt.axis("off")
plt.show()

# %% [markdown]
# ## 학습

# %%
from torch.utils.data import random_split


def get_cifar_datasets(val=0.2, root_dir='CIFAR/'):
    # RandomHorizontalFlip(): 학습 이미지를 50% 확률로 좌우 반전한다.
    # 원본을 다양하게 변형하는 데이터 증강으로 과적합을 줄이는 데 도움을 준다.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # 검증과 테스트에서는 결과가 매번 달라지지 않도록 무작위 변형을 제외한다.
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_origin = datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_origin = datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    val_samples = int(len(train_origin) * val)
    train_samples = len(train_origin) - val_samples

    # 같은 인덱스를 두 Dataset에 적용해 학습과 검증의 전처리만 다르게 사용한다.
    indices = torch.randperm(len(train_origin)).tolist()
    train_indices = indices[:train_samples]
    val_indices = indices[train_samples:]
    trainset = torch.utils.data.Subset(train_origin, train_indices)
    valset = torch.utils.data.Subset(val_origin, val_indices)

    testset = datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )
    return trainset, valset, testset

dataset_list = get_cifar_datasets()

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0
    correct = 0

    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 정확도 계산에는 역전파가 필요 없으므로 계산 기록을 만들지 않는다.
        with torch.no_grad():
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()

    data_num = len(dataloader.dataset)
    acc = 100.0 * correct / data_num
    print(
        f"[{epoch}/{EPOCH}]",
        f"train loss: {running_loss / len(dataloader):.4f}",
        f"train acc: {correct}/{data_num} ({acc:.2f}%)",
    )

    return running_loss / len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch, device):
    model.eval()
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()

    data_num = len(dataloader.dataset)
    acc = 100.0 * correct / data_num
    print(
        f"[{epoch}/{EPOCH}]",
        f"valid loss: {running_loss / len(dataloader):.4f}",
        f"valid acc: {correct}/{data_num} ({acc:.2f}%)\n",
    )

    return running_loss / len(dataloader)

# %%
def test(model, dataloader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()

    data_num = len(dataloader.dataset)
    accuracy = 100.0 * correct / data_num
    print(f"Test Accuracy: {correct}/{data_num} ({accuracy:.2f}%)")

# %% [markdown]
# ## 모델

# %%
# Jupyter-only command: !pip install -q torchsummary

# %% [markdown]
# ### VGG

# %%
import math
from collections import OrderedDict


class VGG(nn.Module):
    # VGG는 작은 3×3 합성곱을 여러 번 쌓아 이미지 특징을 단계적으로 추출한다.
    # 숫자는 합성곱의 출력 채널 수이고 "M"은 최대 풀링 계층을 의미한다.
    def __init__(self, model, num_classes):
        super().__init__()
        vgg_types = {
            "VGG16": [
                64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                512, 512, 512, "M", 512, 512, 512, "M",
            ],
        }

        self.model = model
        self.feature_extractor = self.create_feature_extractor(vgg_types[model])

        # OrderedDict를 사용하면 각 계층에 이름을 붙인 순차 모델을 만들 수 있다.
        # 키 이름은 중복되면 앞 계층이 덮어써지므로 모두 고유해야 한다.
        self.fc = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout()),
            ("fc1", nn.Linear(512, 512)),
            ("relu1", nn.ReLU(True)),
            ("dropout2", nn.Dropout()),
            ("fc2", nn.Linear(512, 512)),
            ("relu2", nn.ReLU(True)),
            ("fc3", nn.Linear(512, num_classes)),
        ]))

    def create_feature_extractor(self, layers, in_channels=3):
        feature_extractor = []

        for layer in layers:
            if isinstance(layer, int):
                out_channels = layer

                # padding=1인 3×3 합성곱은 가로와 세로 크기를 유지한다.
                # BatchNorm2d는 채널별 출력 분포를 정규화해 학습을 안정시킨다.
                feature_extractor += [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
                in_channels = out_channels
            elif layer == "M":
                # 풀링을 한 번 지날 때마다 32→16→8→4→2→1로 크기가 줄어든다.
                feature_extractor.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

        # 리스트의 계층을 순서대로 실행하는 하나의 모듈로 묶는다.
        return nn.Sequential(*feature_extractor)

    def forward(self, x):
        x = self.feature_extractor(x)  # (배치 크기, 512, 1, 1)

        # start_dim=1은 배치 차원은 유지하고 나머지를 1차원으로 펼친다.
        x = torch.flatten(x, start_dim=1)  # (배치 크기, 512)
        x = self.fc(x)                     # (배치 크기, num_classes)
        return x

# %%
from torchsummary import summary

# summary(): 입력 크기를 기준으로 계층별 출력 모양과 매개변수 수를 보여준다.
summary(VGG('VGG16', 10).to(device), (3, 32, 32))

# %%
# VGG16 학습에 사용할 하이퍼파라미터이다.
EPOCH = 50
BATCH_SIZE = 32
NUM_WORKERS = 1
LR = 0.0001  # 1e-4와 같은 값이다.

# CIFAR-10의 클래스가 10개이므로 마지막 출력 크기를 10으로 지정한다.
model = VGG("VGG16", num_classes=10).to(device)

# 학습 데이터 중 20%를 검증 데이터로 분리한다.
trainset, valset, testset = get_cifar_datasets(val=0.2)

train_loader = DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    dataset=valset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(1, EPOCH + 1):
    tloss = train(model, train_loader, criterion, optimizer, epoch, device)
    vloss = validation(model, val_loader, criterion, epoch, device)

    train_losses.append(tloss)
    val_losses.append(vloss)

test(model, test_loader, device)
# %% [markdown]
# ## ResNet

# %%
class BasicBlock(nn.Module):
    # BasicBlock의 마지막 출력 채널은 planes와 같으므로 확장 배수는 1이다.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # 잔차 연결은 입력 x를 합성곱 결과에 직접 더하는 지름길이다.
        # 공간 크기나 채널 수가 달라지면 1×1 합성곱으로 모양을 맞춘다.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 학습할 변화량과 원래 입력을 더해 깊은 신경망의 학습을 돕는다.
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    # 1×1, 3×3, 1×1 합성곱을 사용하며 마지막 채널을 4배로 확장한다.
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # 각 단계의 첫 블록에서 stride=2를 사용해 크기를 절반으로 줄이고
        # 채널 수를 64→128→256→512로 늘린다.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # 첫 블록만 전달받은 stride를 사용하고 나머지는 공간 크기를 유지한다.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 4×4 특징 맵 전체의 평균을 내어 채널마다 하나의 값만 남긴다.
        out = F.avg_pool2d(out, kernel_size=4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    # 네 단계에 BasicBlock을 2개씩 쌓는다.
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    # 50층부터는 채널을 확장하는 Bottleneck 블록을 사용한다.
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# %%
# ResNet18에 3×32×32 이미지가 들어갈 때의 계층별 모양을 확인한다.
summary(ResNet18().to(device), (3, 32, 32))

# %%
# 같은 학습 조건에서 모델만 ResNet18로 바꾸어 성능을 비교한다.
EPOCH = 50
BATCH_SIZE = 32
NUM_WORKERS = 1
LR = 0.0001  # 1e-4

# Model
model = ResNet18().to(device)

# CIFAR dataloader
trainset, valset, testset = get_cifar_datasets(val=0.2)

# dataloader
train_loader = DataLoader(
    dataset=trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    dataset=valset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
test_loader = DataLoader(
    dataset=testset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Loss, Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_losses = []

for epoch in range(1, EPOCH + 1):
    tloss = train(model, train_loader, criterion, optimizer, epoch, device)
    vloss = validation(model, val_loader, criterion, epoch, device)

    train_losses.append(tloss)
    val_losses.append(vloss)

test(model, test_loader, device)
