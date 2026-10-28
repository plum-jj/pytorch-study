# %%
# Converted from: Chapter 3 U-Net 객체 분할로 인물 사진 배경 흐리게 하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 3 U-Net 객체 분할로 인물 사진 배경 흐리게 하기

# %%
# Jupyter-only command: !pip install -q torchinfo

# %%
import random
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#reading files
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

#plots
import matplotlib.pyplot as plt

#model
import torch
from torch import nn, optim
import torchvision
from torchinfo import summary

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 130
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 3.1 데이터 분석(EG1800)

# %%
# Jupyter-only command: %cd /content/drive/MyDrive/Colab\ Notebooks/프로젝트로\ 시작하는\ Pytorch/6.3
# Jupyter-only command: %ls

# %%
# !unzip -qq "EG1800 archive.zip" -d dataset

# %%
# Jupyter-only command: ls dataset/EG1800

# %%
# Jupyter-only command: ls dataset/EG1800/Images | head

# %%
# Jupyter-only command: ls dataset/EG1800/Labels | head

# %%
base_path = 'dataset/EG1800/'
image_path = os.path.join(base_path, 'Images')
label_path = os.path.join(base_path, 'Labels')

file_list = os.listdir(image_path)

print('Total {} images.'.format(len(file_list)))

# %%
image_name = "00001.png"

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(os.path.join(image_path, image_name)))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(plt.imread(os.path.join(label_path, image_name)))
plt.axis("off")

# %%
image_name = "00120.png"

plt.subplot(1, 2, 1)
plt.imshow(plt.imread(os.path.join(image_path, image_name)))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(plt.imread(os.path.join(label_path, image_name)))
plt.axis("off")

# %% [markdown]
# ### 3.2 데이터셋

# %%
class PortraitDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform=transform

    self.image_dir = os.path.join(self.root_dir, "Images")
    self.label_dir = os.path.join(self.root_dir, "Labels")
    self.file_list = os.listdir(self.image_dir)

  def __getitem__(self, idx):
    file_name = self.file_list[idx]
    image_path = os.path.join(self.image_dir, file_name)
    label_path = os.path.join(self.label_dir, file_name)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    if self.transform:
      augs = self.transform(image=image, mask=mask)

      image = augs['image']
      mask = augs['mask']

    return image, mask.unsqueeze(0)

  def __len__(self):
    return len(self.file_list)

# %%
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

# %%
portrait_dataset = PortraitDataset(base_path, train_transform)

data_loader = DataLoader(portrait_dataset, batch_size=1,
                        shuffle=True, num_workers=0)

# %%
for image, target in data_loader:
  img = np.transpose(image[0].numpy(), (1, 2, 0))
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

# %%
def get_dataset(
    root, train=0.8, val=0.1,
    train_transforms=None, test_transforms=None,
    random_seed=130
):
  origin = PortraitDataset(
    root,
    transform=train_transforms
  )

  train_samples = int(len(origin) * train)
  val_samples = int(len(origin) * val)
  test_samples = len(origin) - train_samples - val_samples

  torch.manual_seed(random_seed)
  trainset, valset, testset = torch.utils.data.random_split(
    origin,
    (train_samples, val_samples, test_samples),
  )
  valset.transforms = test_transforms
  testset.transforms = test_transforms

  return trainset, valset, testset

trainset, valset, testset = get_dataset(base_path,
                                        train_transforms=train_transform,
                                        test_transforms=test_transform)
print(len(trainset), len(valset), len(testset))

# %% [markdown]
# ### 3.3 U-Net 모델

# %%
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
  conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, padding=padding)
  bn = nn.BatchNorm2d(out_channels)
  relu = nn.ReLU()
  return nn.Sequential(conv, bn, relu)

# %%
class EncBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(EncBlock, self).__init__()
    self.conv1 = conv_bn_relu(in_channels, out_channels)
    self.conv2 = conv_bn_relu(out_channels, out_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x

# %%
class DecBlock(nn.Module):
  def __init__(self, in_channels, out_channels, hidden_dims):
    super(DecBlock, self).__init__()
    self.conv1 = conv_bn_relu(in_channels, hidden_dims)
    self.conv2 = conv_bn_relu(hidden_dims, out_channels)

  def forward(self, x1, x2):
    x = torch.cat((x1, x2), dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x

# %%
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.enc1 = EncBlock(3, 64)
    self.enc2 = EncBlock(64, 128)
    self.enc3 = EncBlock(128, 256)
    self.enc4 = EncBlock(256, 512)
    self.enc5 = conv_bn_relu(512, 1024)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool(enc1))
    enc3 = self.enc3(self.pool(enc2))
    enc4 = self.enc4(self.pool(enc3))
    enc5 = self.enc5(self.pool(enc4))
    return [enc5, enc4, enc3, enc2, enc1]

# %%
class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.dec5 = conv_bn_relu(1024, 512)
    self.dec4 = DecBlock(1024, 256, 512)
    self.dec3 = DecBlock(512, 128, 256)
    self.dec2 = DecBlock(256, 64, 128)
    self.dec1 = DecBlock(128, 64, 64)

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

# %%
class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.fc = nn.Conv2d(64, 1, kernel_size=1)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    x = self.fc(x)
    return x

# %%
model = UNet()
summary(model, (1, 3, 224, 224))

# %% [markdown]
# ### 3.4 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (images, targets) in enumerate(pbar):
      images = images.to(device)
      targets = targets.type(torch.FloatTensor).to(device)

      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs.squeeze(), targets.squeeze())

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  running_loss = 0.
  with tqdm(dataloader) as pbar:
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

# %%
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

  if vloss < min_loss:
    print(f"Update Best Model: {vloss}")
    torch.save(model.state_dict(), "best.pth")
    min_loss = vloss

# %%
model = UNet()
# load best model
model.load_state_dict(torch.load("best.pth"))
model.eval()

with torch.no_grad():
  image, target = testset[0]
  print(image.shape, target.shape)

  output = model(image.unsqueeze(0))
  sigmoid_output = torch.sigmoid(output)  # Apply sigmoid activation

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

# %% [markdown]
# ### 3.5 추론 및 배경 흐리게 처리하기

# %%
def inference(image):
  model = UNet()
  # load best model
  model.load_state_dict(torch.load("best.pth"))
  model.eval()

  test_transform=A.Compose(
    [
      A.Resize(height=224, width=224),
      A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
      ToTensorV2(),
    ],
  )
  image = test_transform(image=image)["image"]

  output = model(image.unsqueeze(0))
  sigmoid_output = torch.sigmoid(output)  # Apply sigmoid activation

  mask = sigmoid_output > 0.5
  return mask.squeeze()

# %%
image_path = 'dataset/EG1800/Images/00131.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = inference(image).numpy().astype(np.uint8)
mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

radius = 31
blur = image.copy()
for i in range(3):
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

