# Converted from: Chapter 3 U-Net 객체 분할로 인물 사진 배경 흐리게 하기.ipynb
# Original .ipynb is preserved alongside this file.

# ## Chapter 3 U-Net 객체 분할로 인물 사진 배경 흐리게 하기

# Jupyter-only command: !pip install -q torchinfo

import random
import numpy as np
# tqdm(): 반복문의 진행률과 남은 시간을 진행 막대로 표시한다.
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#reading files
import os
# cv2는 OpenCV의 이미지 읽기·색상 변환 기능을 제공한다.
import cv2
from PIL import Image
# Dataset은 데이터 한 건을 꺼내는 규칙을, DataLoader는 이를 배치로 묶는 방식을 정의한다.
from torch.utils.data import Dataset, DataLoader
# Albumentations는 이미지와 마스크에 같은 위치 변환을 함께 적용할 수 있는 증강 라이브러리이다.
import albumentations as A
# ToTensorV2(): NumPy 형식의 이미지와 마스크를 PyTorch 텐서로 바꾼다.
from albumentations.pytorch import ToTensorV2

#plots
import matplotlib.pyplot as plt

#model
import torch
from torch import nn, optim
import torchvision
from torchinfo import summary

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 130
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)  # gpu seed

# ### 3.1 데이터 분석(EG1800)

# 업로드한 노트북도 바로 실행할 수 있도록 Colab의 기본 임시 작업 폴더를 사용한다.
# Jupyter-only command: %cd /content
# Jupyter-only command: %ls

# PaddleSeg가 제공하는 공개 EG1800 압축 파일을 내려받아 dataset 폴더에 해제한다.
# Jupyter-only command: !wget -q -O EG1800.zip https://paddleseg.bj.bcebos.com/dataset/EG1800.zip
# Jupyter-only command: !unzip -qo EG1800.zip -d dataset

print(os.listdir("dataset/EG1800"))

print(os.listdir("dataset/EG1800/Images")[:10])

print(os.listdir("dataset/EG1800/Labels")[:10])

base_path = 'dataset/EG1800/'
# os.path.join(): 운영체제에 맞는 경로 구분자를 사용해 폴더 경로를 안전하게 연결한다.
image_path = os.path.join(base_path, 'Images')
label_path = os.path.join(base_path, 'Labels')

# os.listdir(): Images 폴더에 있는 파일 이름 목록을 가져온다.
file_list = os.listdir(image_path)

print('Total {} images.'.format(len(file_list)))

image_name = "00001.png"

# 같은 파일 이름의 원본 이미지와 정답 마스크를 나란히 확인한다.
plt.subplot(1, 2, 1)
plt.imshow(plt.imread(os.path.join(image_path, image_name)))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(plt.imread(os.path.join(label_path, image_name)))
plt.axis("off")

image_name = "00120.png"

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(os.path.join(image_path, image_name)))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(plt.imread(os.path.join(label_path, image_name)))
plt.axis("off")

# ### 3.2 데이터셋

class PortraitDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform=transform

    self.image_dir = os.path.join(self.root_dir, "Images")
    self.label_dir = os.path.join(self.root_dir, "Labels")
    # Images와 Labels의 같은 파일 이름이 각각 입력 사진과 정답 마스크 한 쌍이 된다.
    self.file_list = os.listdir(self.image_dir)

  def __getitem__(self, idx):
    file_name = self.file_list[idx]
    image_path = os.path.join(self.image_dir, file_name)
    label_path = os.path.join(self.label_dir, file_name)

    # cv2.imread()는 컬러 이미지를 BGR 순서로 읽으므로, 모델에서 일반적인 RGB 순서로 바꾼다.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 마스크는 각 픽셀이 사람 영역인지 나타내므로 색상 채널 없이 흑백으로 읽는다.
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if self.transform:
      # image와 mask를 함께 전달해야 뒤집기·크기 변경이 동일한 위치에 적용된다.
      augs = self.transform(image=image, mask=mask)

      image = augs['image']
      mask = augs['mask']

    # 원본 마스크가 0/255여도 BCE 손실의 정답 범위에 맞게 0/1 float 텐서로 바꾼다.
    mask = (mask > 0).float()
    # 흑백 마스크 (H, W)에 채널 축을 추가해 (1, H, W) 모양으로 반환한다.
    return image, mask.unsqueeze(0)

  def __len__(self):
    # Dataset의 전체 샘플 수는 이미지 파일 개수와 같다.
    return len(self.file_list)

train_transform=A.Compose(
  [
      # A.Compose(): 이미지와 마스크에 적용할 변환을 지정한 순서대로 묶는다.
      A.Resize(height=224, width=224),
      # p는 각 변환이 적용될 확률이다. 사진과 마스크는 같은 방향으로 함께 뒤집힌다.
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.1),
      # ImageNet 평균과 표준편차로 컬러 이미지만 정규화하며, 정답 마스크 값에는 적용하지 않는다.
      A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
      ToTensorV2()
  ],
)

test_transform=A.Compose(
  [
    A.Resize(height=224, width=224),
    A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ToTensorV2(),
  ],
)

portrait_dataset = PortraitDataset(base_path, train_transform)

# DataLoader(): Dataset을 batch_size개씩 묶어 가져온다. shuffle=True이면 에포크마다 순서를 섞는다.
data_loader = DataLoader(portrait_dataset, batch_size=1,
                        shuffle=True, num_workers=0)

for image, target in data_loader:
  # PyTorch 이미지 (채널, 높이, 너비)를 화면 출력용 (높이, 너비, 채널) 순서로 바꾼다.
  img = np.transpose(image[0].numpy(), (1, 2, 0))
  # 배치 축과 크기가 1인 마스크 채널 축을 제거해 (높이, 너비)로 만든다.
  target = target[0].squeeze().numpy()

  # display the image
  plt.subplot(1,2,1)
  plt.imshow(img)
  plt.axis('off')
  plt.title("image")
  plt.subplot(1,2,2)
  plt.imshow(target)
  plt.axis('off')
  plt.title("mask")

  break

def get_dataset(
    root, train=0.8, val=0.1,
    train_transforms=None, test_transforms=None,
    random_seed=130
):
  # 학습 세트와 평가 세트가 서로 다른 transform을 사용하도록 Dataset을 따로 만든다.
  train_origin = PortraitDataset(
    root,
    transform=train_transforms
  )
  eval_origin = PortraitDataset(
    root,
    transform=test_transforms
  )

  # 비율을 실제 샘플 개수로 바꾸고, 나머지는 모두 테스트 세트에 배정한다.
  train_samples = int(len(train_origin) * train)
  val_samples = int(len(train_origin) * val)
  test_samples = len(train_origin) - train_samples - val_samples

  # randperm(): 고정된 난수 생성기로 전체 인덱스를 섞어 매번 같은 분할을 재현한다.
  generator = torch.Generator().manual_seed(random_seed)
  indices = torch.randperm(len(train_origin), generator=generator).tolist()
  train_indices = indices[:train_samples]
  val_indices = indices[train_samples:train_samples + val_samples]
  test_indices = indices[train_samples + val_samples:]

  # Subset(): 같은 인덱스 분할에 학습용 또는 평가용 Dataset을 각각 연결한다.
  trainset = torch.utils.data.Subset(train_origin, train_indices)
  valset = torch.utils.data.Subset(eval_origin, val_indices)
  testset = torch.utils.data.Subset(eval_origin, test_indices)

  return trainset, valset, testset

trainset, valset, testset = get_dataset(base_path,
                                        train_transforms=train_transform,
                                        test_transforms=test_transform)
print(len(trainset), len(valset), len(testset))

# ### 3.3 U-Net 모델

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
  # 합성곱 → 배치 정규화 → ReLU를 반복해서 사용하므로 하나의 함수로 묶는다.
  conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, padding=padding)
  # BatchNorm2d(): 채널별 출력 분포를 정규화해 학습을 안정시킨다.
  bn = nn.BatchNorm2d(out_channels)
  relu = nn.ReLU()
  return nn.Sequential(conv, bn, relu)

class EncBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(EncBlock, self).__init__()
    self.conv1 = conv_bn_relu(in_channels, out_channels)
    self.conv2 = conv_bn_relu(out_channels, out_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class DecBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_dims):
    super(DecBlock, self).__init__()
    self.conv1 = conv_bn_relu(in_channels, hidden_dims)
    self.conv2 = conv_bn_relu(hidden_dims, out_channels)

  def forward(self, x1, x2):
    # U-Net의 스킵 연결: 인코더 특징과 같은 크기의 디코더 특징을 채널 방향(dim=1)으로 합친다.
    x = torch.cat((x1, x2), dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.enc1 = EncBlock(3, 64)
    self.enc2 = EncBlock(64, 128)
    self.enc3 = EncBlock(128, 256)
    self.enc4 = EncBlock(256, 512)
    self.enc5 = conv_bn_relu(512, 1024)
    # MaxPool2d(2, 2)는 공간 크기를 절반으로 줄이면서 중요한 특징을 남긴다.
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool(enc1))
    enc3 = self.enc3(self.pool(enc2))
    enc4 = self.enc4(self.pool(enc3))
    enc5 = self.enc5(self.pool(enc4))
    # 가장 압축된 특징부터 원본에 가까운 특징까지 반환해 디코더의 스킵 연결에 사용한다.
    return [enc5, enc4, enc3, enc2, enc1]

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.dec5 = conv_bn_relu(1024, 512)
    self.dec4 = DecBlock(1024, 256, 512)
    self.dec3 = DecBlock(512, 128, 256)
    self.dec2 = DecBlock(256, 64, 128)
    self.dec1 = DecBlock(128, 64, 64)

    # ConvTranspose2d(2, 2)는 특징 맵의 높이와 너비를 두 배씩 복원한다.
    self.unpool4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
    self.unpool3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
    self.unpool2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
    self.unpool1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

  def forward(self, x):
    dec5 = self.dec5(x[0])

    unpool4 = self.unpool4(dec5)
    dec4 = self.dec4(x[1], unpool4)

    unpool3 = self.unpool3(dec4)
    dec3 = self.dec3(x[2], unpool3)

    unpool2 = self.unpool2(dec3)
    dec2 = self.dec2(x[3], unpool2)

    unpool1 = self.unpool1(dec2)
    dec1 = self.dec1(x[4], unpool1)
    return dec1

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    # 1×1 합성곱으로 64개 특징 채널을 사람/배경을 나타내는 1개 출력 채널로 줄인다.
    self.fc = nn.Conv2d(64, 1, kernel_size=1)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    x = self.fc(x)
    return x

model = UNet()
# 배치 1개, RGB 3채널, 224×224 입력을 기준으로 계층별 출력 크기를 확인한다.
summary(model, (1, 3, 224, 224))

# ### 3.4 모델 학습

def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, targets) in enumerate(pbar):
      images = images.to(device)
      # BCE 계열 손실은 정답을 정수 클래스가 아니라 float 형식의 마스크로 받는다.
      targets = targets.type(torch.FloatTensor).to(device)

      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs.squeeze(), targets.squeeze())

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  running_loss = 0.
  with tqdm(dataloader) as pbar:
    # 검증에서는 가중치를 갱신하지 않으므로 기울기 기록을 끈다.
    with torch.no_grad():
      for i, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.type(torch.FloatTensor).to(device)

        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets.squeeze())

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

  print(f'[{epoch}/{EPOCH}] valid loss: {running_loss/len(dataloader):.4f}\n')

  return running_loss/len(dataloader)

EPOCH = 10
BATCH_SIZE = 32
NUM_WORKERS = 2
LR = 0.001

train_transform=A.Compose(
  [
      A.Resize(height=224, width=224),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.1),
      A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
      ToTensorV2()
  ],
)

test_transform=A.Compose(
  [
    A.Resize(height=224, width=224),
    A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ToTensorV2(),
  ],
)

trainset, valset, testset = get_dataset(base_path,
                                        train_transforms=train_transform,
                                        test_transforms=test_transform
                                       )

train_loader = DataLoader(
  dataset=trainset, shuffle=True,
  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
  dataset=valset,
  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
)

# Model
model = UNet()

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
# BCEWithLogitsLoss(): Sigmoid와 이진 교차 엔트로피를 한 번에 계산해 수치적으로 안정적이다.
criterion = nn.BCEWithLogitsLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

model = model.to(device)
criterion = criterion.to(device)

# Start Training
min_loss = 999
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  tloss = train(model, train_loader, criterion, optimizer, epoch, device)
  vloss = validation(model, val_loader, criterion, epoch, device)
  scheduler.step()

  # 검증 손실이 가장 낮은 U-Net 가중치만 best.pth에 저장한다.
  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    # Chapter 2의 best.pth와 충돌하지 않도록 U-Net 전용 이름으로 저장한다.
    torch.save(model.state_dict(), "unet_best.pth")
    min_loss = vloss

model = UNet()
# load best model
model.load_state_dict(torch.load("unet_best.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
  image, target = testset[0]
  print(image.shape, target.shape)

  output = model(image.unsqueeze(0))
  # 모델 출력은 로짓이므로 Sigmoid로 0~1 확률로 바꾼다.
  sigmoid_output = torch.sigmoid(output)

  # 확률이 0.5보다 큰 픽셀을 사람 영역(True)으로 판정한다.
  mask = sigmoid_output > 0.5

  plt.subplot(1, 3, 1)
  plt.imshow(image.permute((1, 2, 0)))
  plt.title("image")
  plt.axis("off")
  plt.subplot(1, 3, 2)
  plt.imshow(target.squeeze())
  plt.title("target")
  plt.axis("off")
  plt.subplot(1, 3, 3)
  plt.imshow(mask.squeeze())
  plt.title("mask")
  plt.axis("off")

# ### 3.5 추론 및 배경 흐리게 처리하기

def inference(image):
  model = UNet()
  # Chapter 3에서 저장한 U-Net 가중치를 불러온다.
  model.load_state_dict(torch.load("unet_best.pth", map_location="cpu"))
  model.eval()

  test_transform=A.Compose(
    [
      A.Resize(height=224, width=224),
      A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
      ToTensorV2(),
    ],
  )
  image = test_transform(image=image)["image"]

  # 한 장의 (C, H, W) 이미지에 배치 축을 추가하고, 추론 중에는 기울기를 기록하지 않는다.
  with torch.no_grad():
    output = model(image.unsqueeze(0))
    sigmoid_output = torch.sigmoid(output)

  mask = sigmoid_output > 0.5
  return mask.squeeze()

image_path = 'dataset/EG1800/Images/00131.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 불리언 마스크를 OpenCV가 처리할 수 있는 uint8의 0과 1로 바꾼다.
mask = inference(image).numpy().astype(np.uint8)
# 모델 입력은 224×224였으므로 마스크를 원본 사진 크기로 되돌린다.
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

radius = 31
blur = image.copy()
for i in range(3):
  # 사람 영역(mask==1)은 원본을 유지하고 배경(mask==0)에만 가우시안 블러를 적용한다.
  blur[:, :, i] = np.where(mask == 0,
                           cv2.GaussianBlur(blur[:, :, i], (radius, radius), 0),
                           blur[:, :, i])


plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blur)
plt.title("blured")
plt.axis("off")

