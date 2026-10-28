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

is_cuda = torch.cuda.is_available()   # GPU가 사용가능한지
device = torch.device("cuda" if is_cuda else "cpu") # GPU가 사용가능하다면 GPU를 사용 아니면 CPU 사용
print(is_cuda, device)

seed = 1010
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 3.2 데이터셋(개의 품종 분류)

# %%
# Jupyter-only command: !ls "drive/MyDrive/Colab Notebooks/Learning-PyTorch-Deeplearning-with-Hands-On-Projects/Part3/3. 개 품종 분류"
# Jupyter-only command: !unzip "drive/MyDrive/Colab Notebooks/Learning-PyTorch-Deeplearning-with-Hands-On-Projects/Part3/3. 개 품종 분류/dog-breed-identification.zip" -d dataset
# Jupyter-only command: !ls dataset

# %%
# Jupyter-only command: !ls dataset/train| wc -l
# Jupyter-only command: !ls dataset/test | wc -l

# %% [markdown]
# #### EDA(탐색적 데이터 분석)

# %%
import pandas as pd
from IPython.display import display

df = pd.read_csv("dataset/labels.csv")
display(df.head())

# %%
df_count = df.groupby(["breed"])["breed"].count()
df_count = df_count.reset_index(name='count')
df_count = df_count.sort_values(['count'], ascending=False)
display(df_count.head())

print("max:", df_count['count'].max())
print("min:", df_count['count'].min())
print("mean:", df_count['count'].mean())
print("median", df_count['count'].median())

df_count.plot(kind='bar', x='breed', y='count', figsize=(20,3))

# %%
# Sample image
import matplotlib.pyplot as plt
from PIL import Image

for i in range(9):
    plt.subplot(3,3,i+1)
    img = Image.open(f"dataset/train/{df.iloc[i].id}.jpg")

    plt.gca().set_title(df.iloc[i].breed)
    plt.imshow(img)
    plt.axis('off')

# %% [markdown]
# ##### 데이터셋

# %%
from os.path import join

from PIL import Image
from torch.utils.data import random_split, dataset

class Dogs(dataset.Dataset):
  def __init__(self, root, train=True, transforms=None):

    self.train = train
    self.transforms = transforms

    # load image path and labels
    df = pd.read_csv(join(root, 'labels.csv'))
    self.class_names = df['breed'].unique().tolist()

    if train:
      dir = 'train'
      self.labels = df['breed'].tolist()
    else:
      dir = 'test'
      df = pd.read_csv(join(root, 'sample_submission.csv'))

    self.images = [join(root, dir, f"{id}.jpg") for id in df['id']]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, i):
    image = Image.open(self.images[i])

    if self.transforms is not None:
      image = self.transforms(image)

    if self.train:
      label = self.class_names.index(self.labels[i])
    else:
      label = -1

    return image, label

# %%
def get_dogs_datasets(
    val=0.2, root='dataset',
    train_transforms=None, test_transforms=None):
  # split
  origin = Dogs(
      root=root,
      train=True,
      transforms=train_transforms
  )
  val_samples = int(len(origin) * val)
  train_samples = len(origin) - val_samples

  # dataset
  trainset, valset = random_split(
      origin,
      (train_samples, val_samples),
  )
  valset.transforms = test_transforms
  testset = Dogs(
      root=root,
      train=False,
      transforms=test_transforms
  )

  return trainset, valset, testset

dataset_list = get_dogs_datasets(root='dataset')

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

