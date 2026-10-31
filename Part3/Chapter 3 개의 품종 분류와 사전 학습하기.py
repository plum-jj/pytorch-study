# %%
# Converted from: Chapter 3 개의 품종 분류와 사전 학습하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 딥러닝의 기본 이미지 분류하기
# ## Chapter 3 개의 품종 분류와 사전 학습하기

# %%
# Jupyter-only command: !pip install -q torchsummary

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

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# 실험을 다시 실행해도 같은 난수 흐름을 사용하도록 시드를 고정한다.
seed = 1010
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ### 3.2 데이터셋(개의 품종 분류)

# %%
# 아래 명령은 파이썬 코드가 아니라 Colab에서 실행하는 운영체제 명령이다.
# 첫 번째 ls는 압축 파일을 확인하고, unzip은 dataset 폴더에 압축을 풀며,
# 마지막 ls는 압축 해제 결과를 확인한다. 로컬에서는 실제 경로에 맞게 수정해야 한다.
# Jupyter-only command: !ls "drive/MyDrive/Colab Notebooks/Learning-PyTorch-Deeplearning-with-Hands-On-Projects/Part3/3. 개 품종 분류"
# Jupyter-only command: !unzip "drive/MyDrive/Colab Notebooks/Learning-PyTorch-Deeplearning-with-Hands-On-Projects/Part3/3. 개 품종 분류/dog-breed-identification.zip" -d dataset
# Jupyter-only command: !ls dataset

# %%
# ls로 파일 이름을 출력하고 wc -l로 줄 수를 세어 이미지 개수를 확인한다.
# Jupyter-only command: !ls dataset/train | wc -l
# Jupyter-only command: !ls dataset/test | wc -l

# %% [markdown]
# #### EDA(탐색적 데이터 분석)

# %%
import pandas as pd
from IPython.display import display

df = pd.read_csv("dataset/labels.csv")

# head()는 품종 라벨 CSV의 처음 5개 행을 보여준다.
display(df.head())

# %%
# 품종별 행을 묶어 이미지 개수를 센다.
df_count = df.groupby(["breed"])["breed"].count()

# 그룹 이름인 breed를 일반 열로 되돌리고 집계 결과 열의 이름을 count로 정한다.
df_count = df_count.reset_index(name="count")

# 이미지가 많은 품종부터 확인할 수 있도록 내림차순으로 정렬한다.
df_count = df_count.sort_values(["count"], ascending=False)
display(df_count.head())

print("max:", df_count["count"].max())
print("min:", df_count["count"].min())
print("mean:", df_count["count"].mean())
print("median:", df_count["count"].median())

# 품종마다 데이터 수가 크게 치우쳐 있는지 막대그래프로 확인한다.
df_count.plot(kind="bar", x="breed", y="count", figsize=(20, 3))

# %%
import matplotlib.pyplot as plt
from PIL import Image

for i in range(9):
    plt.subplot(3, 3, i + 1)

    # Image.open(): 현재 행의 id와 일치하는 JPEG 파일을 PIL 이미지로 읽는다.
    img = Image.open(f"dataset/train/{df.iloc[i].id}.jpg")

    plt.gca().set_title(df.iloc[i].breed)
    plt.imshow(img)
    plt.axis("off")

# %% [markdown]
# ##### 데이터셋

# %%
from os.path import join

from PIL import Image
from torch.utils.data import Dataset, Subset

class Dogs(Dataset):
    # Dataset을 상속하면 __len__과 __getitem__을 통해
    # DataLoader가 이미지와 정답을 필요한 순서대로 가져갈 수 있다.
    def __init__(self, root, train=True, transforms=None):
        self.train = train
        self.transforms = transforms

        # join(): 운영체제에 맞는 경로 구분자를 사용해 경로를 연결한다.
        df = pd.read_csv(join(root, "labels.csv"))

        # unique()로 중복 없는 품종 목록을 만들며 각 위치가 클래스 번호가 된다.
        self.class_names = df["breed"].unique().tolist()

        if train:
            folder = "train"
            self.labels = df["breed"].tolist()
        else:
            folder = "test"

            # 테스트 데이터에는 정답이 없으므로 제출 양식에서 이미지 id를 읽는다.
            df = pd.read_csv(join(root, "sample_submission.csv"))

        self.images = [
            join(root, folder, f"{image_id}.jpg")
            for image_id in df["id"]
        ]

    def __len__(self):
        # len(dataset)을 호출했을 때 전체 이미지 개수를 반환한다.
        return len(self.images)

    def __getitem__(self, i):
        # dataset[i]를 호출하면 i번째 이미지를 RGB 형식으로 읽는다.
        image = Image.open(self.images[i]).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        if self.train:
            # 품종 문자열을 CrossEntropyLoss가 사용할 0부터 시작하는 번호로 바꾼다.
            label = self.class_names.index(self.labels[i])
        else:
            # 테스트 데이터에는 공개된 정답이 없음을 -1로 표시한다.
            label = -1

        return image, label

# %%
def get_dogs_datasets(
    val=0.2, root='dataset',
    train_transforms=None, test_transforms=None):
    # 학습용과 검증용 Dataset을 따로 만들어 서로 다른 전처리를 적용한다.
    # random_split의 Subset에 transforms를 직접 지정하면 원본 Dataset에는
    # 반영되지 않으므로, 같은 인덱스를 가진 두 Dataset을 사용하는 방식이다.
    train_origin = Dogs(
        root=root,
        train=True,
        transforms=train_transforms,
    )
    val_origin = Dogs(
        root=root,
        train=True,
        transforms=test_transforms,
    )

    val_samples = int(len(train_origin) * val)
    indices = torch.randperm(len(train_origin)).tolist()
    val_indices = indices[:val_samples]
    train_indices = indices[val_samples:]

    # Subset은 원본 Dataset에서 지정된 인덱스만 보이도록 감싼다.
    trainset = Subset(train_origin, train_indices)
    valset = Subset(val_origin, val_indices)
    testset = Dogs(
        root=root,
        train=False,
        transforms=test_transforms,
    )

    return trainset, valset, testset


dataset_list = get_dogs_datasets(root="dataset")

# %% [markdown]
# ### 3.3 사전 학습된 ResNet50 모델

# %%
from torchsummary import summary

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
print(model)

# %%
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 120)
)

print(model)

# %%
summary(model.to('cuda'), (3, 224, 224))

# %%
def load_resnet():
  model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

  for param in model.parameters():
    param.requires_grad = False

  model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 120)
  )

  return model

# %%
model = load_resnet()
summary(model.to('cuda'), (3, 224, 224))

# %% [markdown]
# ### 3.4 학습, 검증, 테스트

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  # train mode
  model.train()

  # 학습 통계
  running_loss = 0
  correct = 0

  for i, (data, targets) in enumerate(dataloader):
    data, targets = data.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    # Accuracy 계산
    with torch.no_grad():
      _, predicted = torch.max(outputs, 1)
      correct += predicted.eq(targets.view_as(predicted)).sum()

  # Accuracy 출력
  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num
  print(
      f"[{epoch}/{EPOCH}]",
      f"train loss: {running_loss/len(dataloader):.4f}",
      f"train acc: {correct}/{data_num} ({acc:.2f}%)"
  )

  return running_loss/len(dataloader), acc

# %%
def validation(model, dataloader, criterion, epoch, device):
  # eval 모드
  model.eval()

  # 검증 통계
  correct = 0
  running_loss = 0.

  with torch.no_grad():   # gradient 계산 안하기
    for i, (data, targets) in enumerate(dataloader):
      data, targets = data.to(device), targets.to(device)   # data device로 이동

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

  return running_loss/len(dataloader), acc

# %%
# Hyperparameters
# Training
EPOCH = 30
BATCH_SIZE = 64
NUM_WORKERS = 1
LR = 1e-4
MODEL_ROOT = 'checkpoint'
os.makedirs(MODEL_ROOT, exist_ok=True)

# transform
train_transform = transforms.Compose([
  transforms.RandomResizedCrop((256, 256)),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(brightness=0.4,
                         contrast=0.4,
                         saturation=0.4),
  transforms.CenterCrop((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.CenterCrop((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# dataloader
trainset, valset, testset = get_dogs_datasets(
  val=0.1,
  train_transforms=train_transform,
  test_transforms=test_transform,
)

# dataloader
train_loader = DataLoader(
    dataset=trainset,
    shuffle=True,
    batch_size=BATCH_SIZE,
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

# Model
model = load_resnet()
model = model.to(device)

# Loss, Optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
val_losses = []
max_acc = 0
last_checkpoint = None

for epoch in range(1, EPOCH+1):
  tloss, tacc = train(model, train_loader, criterion, optimizer, epoch, device)
  vloss, vacc = validation(model, val_loader, criterion, epoch, device)

  train_losses.append(tloss)
  val_losses.append(vloss)

  scheduler.step()

  if vacc >= max_acc:
    max_acc = vacc
    last_checkpoint = f"{MODEL_ROOT}/resnet50-ep{epoch}-{vacc:.2f}.pt"
    torch.save(model.state_dict(), last_checkpoint)


# %% [markdown]
# #### 추론

# %%
def inference(model, image, device, root='dataset'):
  df = pd.read_csv(join(root, 'labels.csv'))
  class_names = df['breed'].unique().tolist()

  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
  ])

  image = transform(image)
  image = image.unsqueeze(0).to(device)

  model.eval()
  model = model.to(device)

  outputs = model(image)
  _, predicted = torch.max(outputs, 1)
  print(predicted)
  print(class_names[predicted.item()])

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

last_checkpoint = "/content/drive/MyDrive/Colab Notebooks/프로젝트로 시작하는 Pytorch/5.3/checkpoint/resnet50-ep50-70.35.pt"
model = load_resnet()
model.load_state_dict(torch.load(last_checkpoint))

# %%
img = Image.open("mong.jpg")
img

# %%
inference(model, img, device)
# %%
img = Image.open("coco.jpg")
img

# %%
inference(model, img, device)
