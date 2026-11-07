# Converted from: Chapter 5 자연 풍경 이미지 분류하기.ipynb
# Original .ipynb is preserved alongside this file.

# # 딥러닝의 기본 이미지 분류하기
# ## Chapter 5 자연 풍경 이미지 분류하기

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#reading files
import os
import scipy.io as sio
import cv2
from torch.utils.data import Dataset, DataLoader

#plots
import matplotlib as mpl
import matplotlib.pyplot as plt
from random import randrange

#model
import torch
from torch import nn, optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

# metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

is_cuda = torch.cuda.is_available()
# 사용 가능한 GPU가 있으면 CUDA를, 없으면 CPU를 계산 장치로 선택한다.
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 827
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device.type == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# ### 5.1 데이터 분석

# Jupyter-only command: %cd /content/drive/MyDrive/Colab\ Notebooks/프로젝트로\ 시작하는\ Pytorch/5.5
# Jupyter-only command: %ls

# Jupyter-only command: !unzip -qq MIML.zip -d dataset

# Jupyter-only command: ls dataset

# Jupyter-only command: ls dataset/miml-image-data

#base directory
base_dir = 'dataset/miml-image-data'
image_dir = os.path.join(base_dir, 'original')

mat_file_path = os.path.join(base_dir, 'processed/miml data.mat')
# loadmat(): MATLAB .mat 파일을 파이썬 딕셔너리 형태로 읽는다.
mat_file = sio.loadmat(mat_file_path)

#target labels
# 원본 targets는 (클래스, 이미지) 모양이므로 전치해 (이미지, 클래스)로 바꾼다.
# 각 이미지는 5개 클래스별 포함 여부를 [0, 1, 0, ...]처럼 갖는 다중 레이블이다.
target_array = mat_file['targets'].T
target_list = [[j if j==1 else 0 for j in row] for row in target_array]

#class
class_names = [j[0][0] for j in mat_file['class_name']]

print("number of samples:", len(target_list))
print('target: ', target_list[425])
print('class: ', class_names)

num_classes = np.sum(target_list, axis=0)
# axis=0으로 이미지 방향을 합하면 클래스별 등장 횟수가 된다.
plt.bar(class_names, num_classes)
print(num_classes)

counts, num = np.unique(np.sum(target_list, axis=1), return_counts=True)
# 이미지별 레이블 개수를 센 뒤, 레이블이 1개·2개 등인 이미지가 각각 몇 장인지 집계한다.
plt.bar(counts, num)
print(num)

avg_label_pre_image = np.sum(counts * num) / len(target_list)
# 레이블 개수별 빈도를 가중 평균하여 이미지 한 장당 평균 레이블 수를 구한다.
avg_label_pre_image

# ### 5.2 데이터셋

class SceneDataset(Dataset):
  def __init__(self, root, transform=None):
    self.root = root
    self.transform = transform

    mat_file_path = os.path.join(root, 'processed/miml data.mat')
    mat_file = sio.loadmat(mat_file_path)
    target_array = mat_file['targets'].T

    # -1/1 형식인 원본 레이블을 BCE 손실에서 쓸 0/1 형식으로 바꾼다.
    self.targets = [[j if j==1 else 0 for j in row] for row in target_array]
    self.images = [
      os.path.join(root, f'original/{a+1}.jpg')
      for a in range(len(self.targets))
    ]

    self.class_names = [j[0][0] for j in mat_file['class_name']]

  def __getitem__(self, index):
    path, target = self.images[index], self.targets[index]

    # OpenCV는 BGR 순서로 읽으므로 화면 표시와 사전 학습 모델에 맞게 RGB로 바꾼다.
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transform is not None:
      augmented = self.transform(image=image)
      image = augmented["image"]

    # 한 이미지가 여러 클래스에 속할 수 있으므로 target은 길이 5의 벡터이다.
    return image, torch.tensor(target)

  def __len__(self):
    return len(self.targets)

  def target2class(self, target):
    # 값이 1인 위치의 클래스 이름만 골라 사람이 읽을 수 있는 목록으로 만든다.
    return [self.class_names[i] for i, v in enumerate(target) if v == 1]

def get_dataset(
  root, train=0.8, val=0.1,
  train_transforms=None, test_transforms=None,
  random_seed=827
):
  # 같은 이미지 순서를 가진 데이터셋을 변환별로 따로 만든다.
  # 하나만 나눈 뒤 Subset의 transform을 바꾸면 공유 원본의 변환이 함께 바뀔 수 있다.
  train_origin = SceneDataset(root, transform=train_transforms)
  eval_origin = SceneDataset(root, transform=test_transforms)

  train_samples = int(len(train_origin) * train)
  val_samples = int(len(train_origin) * val)
  test_samples = len(train_origin) - train_samples - val_samples

  # random_split으로 재현 가능한 인덱스를 만든 뒤 각 변환용 데이터셋에 적용한다.
  torch.manual_seed(random_seed)
  train_split, val_split, test_split = torch.utils.data.random_split(
    train_origin,
    (train_samples, val_samples, test_samples),
  )

  trainset = torch.utils.data.Subset(train_origin, train_split.indices)
  valset = torch.utils.data.Subset(eval_origin, val_split.indices)
  testset = torch.utils.data.Subset(eval_origin, test_split.indices)

  return trainset, valset, testset

trainset, valset, testset = get_dataset(base_dir)
print(len(trainset), len(valset), len(testset))

plt.figure(figsize=(16,6))
for i, data in enumerate(trainset):
  image, label = data

  plt.subplot(2,5,i+1)
  plt.imshow(image)
  plt.title(trainset.dataset.target2class(label))
  plt.axis('off')

  if i >= 9:
    break

# ### 5.3 ResNet18 모델

# Jupyter-only command: !pip install -q torchinfo

import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights

from torchinfo import summary


def load_resnet(classes):
  # ImageNet으로 사전 학습된 ResNet18의 특징 추출 능력을 가져온다.
  model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
  # 기존 1,000개 분류 출력층을 다중 레이블 5개 점수를 내는 층으로 교체한다.
  # 각 출력은 서로 독립적인 클래스의 logit이므로 Softmax를 사용하지 않는다.
  model.fc = nn.Linear(in_features=512, out_features=classes, bias=True)

  return model
summary(load_resnet(5), (1, 3, 224, 224))

# ### 5.4 모델 학습

def train(model, dataloader, criterion, optimizer, epoch, device):
  # model.train(): Dropout과 BatchNorm을 학습 모드로 전환한다.
  model.train()

  # 학습 통계
  running_loss = 0
  correct = 0

  with tqdm(dataloader) as pbar:
    for i, (data, targets) in enumerate(pbar):
      data, targets = data.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(data)
      # 정답의 5개 0/1 값을 실수형으로 바꿔 각 클래스의 이진 손실을 계산한다.
      loss = criterion(outputs, targets.float())
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

      # Accuracy 계산
      with torch.no_grad():
        # 각 클래스 확률을 독립적으로 0/1로 바꾼다.
        predicted = torch.sigmoid(outputs).round()
        # 한 이미지의 5개 레이블을 모두 맞힌 경우만 정확한 샘플로 센다.
        correct += torch.all(torch.eq(predicted, targets), dim=-1).sum()

    # Accuracy 출력
    data_num = len(dataloader.dataset)
    acc = 100. * correct / data_num
    print(
        f"[{epoch}/{EPOCH}]",
        f"train loss: {running_loss/len(dataloader):.4f}",
        f"train acc: {correct}/{data_num} ({acc:.2f}%)"
    )

  return running_loss/len(dataloader), acc

def validation(model, dataloader, criterion, epoch, device):
  # 검증에서는 가중치를 바꾸지 않고 일반화 성능을 확인한다.
  model.eval()

  # 검증 통계
  correct = 0
  running_loss = 0.

  with tqdm(dataloader) as pbar:
    with torch.no_grad():
      for i, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets.float())

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        # Accuracy 계산
        predicted = torch.sigmoid(outputs).round()
        correct += torch.all(torch.eq(predicted, targets), dim=-1).sum()

  # Accuracy 계산
  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num
  print(f'[{epoch}/{EPOCH}] valid loss: {running_loss/len(dataloader):.4f} valid acc: {correct}/{data_num} ({acc:.2f}%)\n')

  return running_loss/len(dataloader), acc

# Jupyter-only command: !pip install -q wandb

# Jupyter-only command: !wandb login

import wandb

# 이번 학습 실행을 W&B에 만들고 지표·체크포인트를 기록한다.
wandb.init(project="Natural Scene", save_code=True)

EPOCH = 10
BATCH_SIZE = 128
NUM_WORKERS = 1
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

trainset, valset, testset = get_dataset(
  base_dir,
  train_transforms=train_transform,
  test_transforms=test_transform
)

# 학습 데이터만 매 epoch 순서를 섞고 검증·테스트 데이터 순서는 유지한다.
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
model = load_resnet(5)

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
# 5개 클래스 각각에 Sigmoid와 이진 교차 엔트로피를 적용하는 손실 함수이다.
criterion = nn.BCEWithLogitsLoss()
# 3 epoch마다 학습률을 절반으로 줄인다.
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
    # 최고 검증 정확도를 갱신한 모델만 저장한다.
    max_acc = vacc
    torch.save(model.state_dict(), "best.pth")

# load best model
model.load_state_dict(torch.load("best.pth", map_location=device))
artifact = wandb.Artifact('best', type='checkpoint')
artifact.add_file('best.pth')
wandb.log_artifact(artifact)

def get_probs_trues(dataset, model, device):
  # 임계값 탐색과 평가에 사용할 클래스별 확률과 정답을 모은다.
  probs, trues = [], []
  model.eval()
  with torch.no_grad():
    for data, label in dataset:
      # 단일 이미지 (C,H,W)에 배치 차원을 추가해 (1,C,H,W)로 입력한다.
      logit = model(data.unsqueeze(0).to(device))
      probs.append(torch.sigmoid(logit).squeeze(0).cpu())
      trues.append(label.squeeze().cpu())
  return torch.stack(probs), torch.stack(trues)

def get_best_thresholds(y_probs, y_trues):
  # 기본 0.5만 쓰지 않고 0.10~0.85 후보 중 클래스별 F1이 가장 높은 기준을 찾는다.
  threshold = np.arange(0.1, 0.9, 0.05)
  best_threshold = np.zeros(y_probs.shape[1])
  for i in range(y_probs.shape[1]):
    f1 = []
    y_prob = np.array(y_probs[:,i])
    for j in threshold:
      y_pred = [1 if prob >= j else 0 for prob in y_prob]
      f1.append(f1_score(y_trues[:,i], y_pred))
    f1   = np.array(f1)
    index = np.where(f1 == f1.max())
    best_threshold[i] = threshold[index][0]
  return best_threshold

# 검증 데이터로 임계값을 정해야 테스트 데이터가 기준 선택에 섞이지 않는다.
y_probs, y_trues = get_probs_trues(valset, model, device)
class_names = testset.dataset.class_names

best_threshold = get_best_thresholds(y_probs, y_trues)

for i in range(5):
  print(f"{class_names[i]}: {best_threshold[i]:.2f}")

wandb.log({
  "best_threshold": wandb.Table(
    columns=[str(class_names[i]) for i in range(5)],
    data=best_threshold.reshape(1,-1)
  )
})

# ### 5.5 정확도/정밀도/재현율/F1 스코어
# 그리고 혼돈 행렬(Confusion matrix)

# ### 5.6 테스트

def get_pred(y_test, y_proba, threshold):
  # 클래스마다 앞에서 찾은 서로 다른 임계값을 적용해 최종 0/1 예측을 만든다.
  y_pred = np.array([[1 if y_proba[i, j] >= threshold[j] else 0
                      for j in range(y_test.shape[1])]
                      for i in range(y_test.shape[0])])
  return y_pred

# 임계값을 고를 때 사용하지 않은 테스트 데이터로 최종 성능을 측정한다.
y_probs, y_true = get_probs_trues(testset, model, device)
y_pred = get_pred(y_true, y_probs, best_threshold)

# 클래스마다 2x2 혼동 행렬을 하나씩 그린다.
f, axes = plt.subplots(2, 3, figsize=(10, 8))
axes = axes.ravel()
for i in range(5):
  disp = ConfusionMatrixDisplay(confusion_matrix(y_true[:, i],
                                                 y_pred[:, i]),
                                display_labels=["False", "True"])
  disp.plot(ax=axes[i], values_format='.4g')
  disp.ax_.set_title(class_names[i])
  disp.im_.colorbar.remove()
axes[5].axis("off")
plt.subplots_adjust(wspace=0.4, hspace=0.1)

# #### Classification Report

# classification_report는 정답을 먼저, 예측을 두 번째 인자로 받는다.
report = classification_report(y_true, y_pred,
                               target_names=class_names,
                               output_dict=True
                              )
wandb.log({
  "report": wandb.Table(
      dataframe=pd.DataFrame(report).T.reset_index(),
      allow_mixed_types=True
    )
})

# #### Prediction Samples

# 증강된 텐서 대신 원본 RGB 이미지를 표시하기 위해 시각화 동안 변환을 해제한다.
testset.dataset.transform = None

plt.figure(figsize=(16,6))
for i, data in enumerate(testset):
  image, label = data

  truth = testset.dataset.target2class(label)
  pred = testset.dataset.target2class(y_pred[i])

  plt.subplot(2,5,i+1)
  plt.imshow(image)
  plt.title(f'Truth: {truth}\nPred: {pred}')
  plt.axis('off')

  if i >= 9:
    break

wandb.log({"testset prediction samples": plt})
wandb.finish()
