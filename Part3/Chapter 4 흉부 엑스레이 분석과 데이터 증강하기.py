# %%
# Converted from: Chapter 4 흉부 엑스레이 분석과 데이터 증강하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# # 딥러닝의 기본 이미지 분류하기
# ## Chapter 4 흉부 엑스레이 분석과 데이터 증강하기

# %%
import os
from pathlib import Path
import random


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torch import nn,optim
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 1010
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 4.1 데이터셋

# %%
# Jupyter-only command: %cd /content/drive/MyDrive/Colab\ Notebooks/프로젝트로\ 시작하는\ Pytorch/5.4

# %%
# Jupyter-only command: !unzip -qq -n "Chest X-Ray Pneumonia.zip" -d dataset

# %%
# Jupyter-only command: %ls dataset/chest_xray/train

# %%
import os

print("Train")
print("NORMAL:", len(os.listdir("dataset/chest_xray/train/NORMAL")), end=', ')
print("PNEUMONIA:", len(os.listdir("dataset/chest_xray/train/PNEUMONIA")))
print()
print("Validation")
print("NORMAL:", len(os.listdir("dataset/chest_xray/val/NORMAL")), end=', ')
print("PNEUMONIA:", len(os.listdir("dataset/chest_xray/val/PNEUMONIA")))
print()
print("Test")
print("NORMAL:", len(os.listdir("dataset/chest_xray/test/NORMAL")), end=', ')
print("PNEUMONIA:", len(os.listdir("dataset/chest_xray/test/PNEUMONIA")))

# %%
from PIL import Image
import matplotlib.pyplot as plt

root = "dataset/chest_xray/test/"
normal_dir = root + 'NORMAL/'
pneumonia_dir = root + 'PNEUMONIA/'

normal = list(map(lambda x: normal_dir+x, os.listdir(normal_dir)[:5]))
pneumonia = list(map(lambda x: pneumonia_dir+x, os.listdir(pneumonia_dir)[:5]))

samples = pneumonia + normal

# show samples
plt.figure(figsize=(30,10))
for i in range(10):
  plt.subplot(2, 5, i+1)
  img = Image.open(samples[i])

  ax = plt.gca()
  ax.set_title("Pneumonia" if i < 5 else "Normal")
  ax.imshow(img, cmap='gray')
  ax.axis('off')
  ax.set_aspect('auto')
plt.show()

# %% [markdown]
# #### 이미지 폴더(ImageFolder)

# %%
def get_dataset(
    root="dataset/chest_xray", val=0.1,
    train_transforms=None, test_transforms=None
):
  origin = datasets.ImageFolder(
      os.path.join(root, 'train'),
      transform=train_transforms
  )

  val_samples = int(len(origin) * val)
  train_samples = len(origin) - val_samples

  trainset, valset = torch.utils.data.random_split(
    origin,
    (train_samples, val_samples),
  )
  valset.transforms = test_transforms

  testset = datasets.ImageFolder(
      os.path.join(root, 'test'),
      transform=test_transforms
  )
  return trainset, valset, testset

trainset, valset, testset = get_dataset(train_transforms=transforms.ToTensor())
print(len(trainset), len(valset), len(testset))

# %%
class_names = trainset.dataset.classes
print(class_names)
print(trainset.dataset.class_to_idx)

# %%
image, label = trainset[0]

print(image.shape)
print(label)

# %% [markdown]
# ### 4.2 데이터 증강과 Albumentation

# %%
# Jupyter-only command: !pip install -q -U opencv-python
# Jupyter-only command: !pip install -q -U albumentations

# %%
import cv2
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=(-20,+20)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

trainset, _, _ = get_dataset(
  train_transforms=train_transform,
)

train_loader = DataLoader(
  dataset=trainset,
  shuffle=True,
  batch_size=64,
  num_workers=0,
)

for i in tqdm(train_loader):
  pass


# %%
class AlbumentationsDataset(ImageFolder):

  def __getitem__(self, index: int):
    path, target = self.samples[index]
    # Read image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform
    if self.transform is not None:
      augmented = self.transform(image=image)
      image = augmented['image']

    return image, target

# %%
def get_dataset_v2(root="dataset/chest_xray", val=0.1, train_transforms=None, test_transforms=None):
  origin = AlbumentationsDataset(os.path.join(root, 'train'), transform=train_transforms)

  val_samples = int(len(origin) * val)
  train_samples = len(origin) - val_samples

  trainset, valset = torch.utils.data.random_split(
    origin,
    (train_samples, val_samples),
  )
  valset.transforms = test_transforms

  testset = AlbumentationsDataset(os.path.join(root, 'test'), transform=test_transforms)
  return trainset, valset, testset

trainset, valset, testset = get_dataset_v2(train_transforms=None)
print(len(trainset), len(valset), len(testset))

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
        A.Resize(256, 256),
        A.Rotate(limit=(-20, +20)),
        A.RandomCrop(224, 224),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2(),
])

trainset, _, _ = get_dataset_v2(
  train_transforms=train_transform,
)

train_loader = DataLoader(
  dataset=trainset,
  shuffle=True,
  batch_size=64,
  num_workers=0,
)

for i in tqdm(train_loader):
    pass

# %% [markdown]
# #### Albumentation만의 특별한 기능

# %%
test_transform = A.Compose([
        A.Resize(256,256),
        A.OneOf([
            A.HorizontalFlip(p=1),
            A.RandomRotate90(p=1),
            A.VerticalFlip(p=1)
        ], p=1),
        A.CenterCrop(224, 224),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.OpticalDistortion(p=0.4),
            A.GaussNoise(p=0.5)
        ], p=0.5),
        ToTensorV2(),
])

# %%
data_dir = Path("dataset/chest_xray/")
testset = AlbumentationsDataset(data_dir / 'test', transform=test_transform)

num_samples = 5
fig, ax = plt.subplots(1, num_samples, figsize=(25, 5))
for i in range(num_samples):
  ax[i].imshow(transforms.ToPILImage()(testset[0][0]))
  ax[i].axis('off')

# %% [markdown]
# ### 4.3 학습 모델(ResNet18)

# %%
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print(model)

# %%
def load_resnet():
  model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(in_features=512, out_features=1, bias=True)  ###

  return model

# %% [markdown]
# ### 4.4 학습 및 기록(Wandb)

# %%
# Jupyter-only command: !pip install -q wandb

# %%
# Jupyter-only command: !wandb login

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  # train mode
  model.train()

  # 학습 통계
  running_loss = 0
  correct = 0

  with tqdm(dataloader) as pbar:
    for i, (data, targets) in enumerate(pbar):
      data, targets = data.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(data)
      loss = criterion(outputs, targets.unsqueeze(1).float()) #### Change
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

      # Accuracy 계산
      with torch.no_grad():
        predicted = torch.sigmoid(outputs).round() #### Change
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

  with tqdm(dataloader) as pbar:
    with torch.no_grad():
      for i, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets.unsqueeze(1).float())  #### Change

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # Accuracy 계산
        predicted = torch.sigmoid(outputs).round() #### Change
        correct += predicted.eq(targets.view_as(predicted)).sum()

  # Accuracy 계산
  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num
  print(f'[{epoch}/{EPOCH}] valid loss: {running_loss/len(dataloader):.4f} valid acc: {correct}/{data_num} ({acc:.2f}%)\n')

  return running_loss/len(dataloader), acc

# %%
def test(model, dataloader, device):
    # eval 모드
    model.eval()

    # 테스트 통계
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
      for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)    # forward

        # Accuracy 계산
        predicted = torch.sigmoid(outputs).round()  ### Change
        correct += predicted.eq(targets.view_as(predicted)).sum()

        y_true.append(targets)
        y_pred.append(outputs)

    # Accuracy 계산
    data_num = len(dataloader.dataset)
    print(f'Test Accuracy: {correct}/{data_num} ({100. * correct / data_num:.2f}%)')

    return 100. * correct / data_num, torch.cat(y_true), torch.cat(y_pred)

# %%
import wandb

wandb.init(project="Pneumonia", save_code=True)

EPOCH = 10
BATCH_SIZE = 256
NUM_WORKERS = 0
LR = 0.001

wandb.config = {
  "learning_rate": LR,
  "epochs": EPOCH,
  "batch_size": BATCH_SIZE,
  "num_workers": NUM_WORKERS
}

train_transform = A.Compose([
        A.Resize(256, 256),
        A.Rotate(limit=(-20, +20)),
        A.RandomCrop(224, 224),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2(),
])
test_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
        ToTensorV2(),
])

trainset, valset, testset = get_dataset_v2(
  train_transforms=train_transform,
  test_transforms=test_transform
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

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

model = model.to(device)
criterion = criterion.to(device)

max_acc = 0
# Start Training
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  tloss, tacc = train(model, train_loader, criterion, optimizer, epoch, device)
  vloss, vacc = validation(model, val_loader, criterion, epoch, device)

  wandb.log({
      "lr": scheduler.get_last_lr()[0],
      "train_loss": tloss,
      "train_accuracy": tacc,
      "val_loss": vloss,
      "val_acc": vacc
  })
  scheduler.step()

  if vacc > max_acc:
    torch.save(model.state_dict(), "best.pth")

# load best model
model.load_state_dict(torch.load("best.pth"))
artifact = wandb.Artifact('best', type='checkpoint')
artifact.add_file('best.pth')
wandb.log_artifact(artifact)

# Test
tacc, y_true, y_preds = test(model, test_loader, device)
class_names = testset.classes
wandb.log({
  "test_accuracy": tacc,
  "conf_mat": wandb.plot.confusion_matrix(probs=None,
                y_true=y_true.tolist(),
                preds=torch.sigmoid(y_preds).squeeze().round().int().tolist(),
                class_names=class_names)})
wandb.finish()

