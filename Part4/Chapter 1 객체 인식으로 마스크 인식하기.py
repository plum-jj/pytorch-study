# %%
# Converted from: Chapter 1 객체 인식으로 마스크 인식하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 1 객체 인식으로 마스크 인식하기

# %%
# Jupyter-only command: !pip install -q torchinfo

# %%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 이미지와 XML 주석 파일을 읽는 데 사용하는 라이브러리이다.
import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET

# Dataset은 데이터 한 개를 가져오는 규칙을 정의하고,
# DataLoader는 Dataset을 배치 단위로 반복해서 가져온다.
from torch.utils.data import Dataset, DataLoader

# Albumentations는 이미지와 객체 위치 상자를 함께 변환할 수 있는 전처리 라이브러리이다.
import albumentations as A

# ToTensorV2는 Albumentations의 결과를 PyTorch 텐서로 변환한다.
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

import torch
from torch import nn, optim
import torchvision

# Faster R-CNN의 기본 분류기를 현재 데이터셋의 클래스 수에 맞게 교체할 때 사용한다.
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# summary는 입력 크기를 기준으로 모델의 계층과 출력 모양을 요약한다.
from torchinfo import summary

# %%
# CUDA GPU 사용 가능 여부를 확인한 뒤 모델과 텐서를 계산할 장치를 정한다.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 322

# 파이썬, NumPy, PyTorch가 같은 난수 흐름을 사용하도록 시드를 고정한다.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device는 torch.device 객체이므로 실제 장치 종류는 device.type으로 확인한다.
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ### 1.2 데이터 분석

# %%
# 업로드한 노트북도 바로 실행할 수 있도록 Colab의 기본 임시 작업 폴더를 사용한다.
# Jupyter-only command: %cd /content
# Jupyter-only command: %ls

# %%
# Kaggle의 공개 Face Mask Detection 데이터셋을 내려받아 dataset 폴더에 압축 해제한다.
# Jupyter-only command: !wget -q -O face-mask-detection.zip https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection
# Jupyter-only command: !unzip -qo face-mask-detection.zip -d dataset

# %%
print(os.listdir("dataset"))

# %%
print(os.listdir("dataset/images")[:10])

# %%
print(os.listdir("dataset/annotations")[:10])

# %%
with open("dataset/annotations/maksssksksss457.xml", encoding="utf-8") as annot_file:
  print(annot_file.read())

# %%
def read_annot(file_name, xml_dir):
  bbox = []
  labels = []

  # 이미지 파일의 확장자를 xml로 바꾸어 같은 이름의 주석 파일 경로를 만든다.
  annot_path = os.path.join(xml_dir, file_name[:-3]+'xml')

  # ET.parse()는 XML 파일을 트리 구조로 읽는다.
  tree = ET.parse(annot_path)
  root = tree.getroot()

  # 한 이미지에 여러 object가 있을 수 있으므로 모든 객체를 반복한다.
  for boxes in root.iter('object'):
    # Pascal VOC 좌표는 1부터 시작하므로 파이썬 이미지 좌표에 맞게 1을 뺀다.
    ymin = int(boxes.find("bndbox/ymin").text)-1
    xmin = int(boxes.find("bndbox/xmin").text)-1
    ymax = int(boxes.find("bndbox/ymax").text)-1
    xmax = int(boxes.find("bndbox/xmax").text)-1
    label = boxes.find('name').text

    bbox.append([xmin,ymin,xmax,ymax])

    # Faster R-CNN에서 0번 클래스는 배경으로 예약한다.
    # 마스크 미착용 계열은 1, 마스크 착용은 2로 변환한다.
    if label == 'with_mask':
      label_idx = 2
    else:
      label_idx = 1
    labels.append(label_idx)

  return bbox, labels

# %%
def draw_boxes(img, boxes, labels, thickness=1):
  # Matplotlib은 RGB, OpenCV의 그리기 함수는 BGR 색상 순서를 사용하므로 변환한다.
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  for box,label in zip(boxes,labels):
    # OpenCV의 rectangle()에 전달할 좌표는 정수여야 한다.
    box = [int(x) for x in box]
    if label == 2:
      color = (0,225,0)
    elif label == 1:
      color = (0,0,225)
    cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]),color,thickness)

  # 화면 출력에 사용할 수 있도록 다시 RGB 순서로 되돌린다.
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
dir_path = 'dataset/images'
xml_path = 'dataset/annotations'

file_list = os.listdir(dir_path)

# os.listdir()가 반환한 파일 이름의 개수로 전체 이미지 수를 확인한다.
print('Total {} images.'.format(len(file_list)))

# %%
image_name = "maksssksksss457.png"
bbox, labels  = read_annot(image_name, xml_path)

# XML에서 읽은 상자를 실제 이미지 위에 그려 주석 위치가 맞는지 확인한다.
img = draw_boxes(plt.imread(os.path.join(dir_path, image_name)), bbox, labels)
plt.imshow(img)
plt.axis('off')

# %% [markdown]
# ### 1.3 데이터셋

# %%
class MaskDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform=transform

    # 이미지와 XML 주석이 저장된 하위 폴더 경로를 만든다.
    self.image_dir = os.path.join(self.root_dir, "images")
    self.xml_dir = os.path.join(self.root_dir, "annotations")
    self.image_list = os.listdir(self.image_dir)

  def __getitem__(self, idx):
    # dataset[idx]를 호출하면 해당 이미지와 Faster R-CNN용 정답을 반환한다.
    image_name = self.image_list[idx]
    image_path = os.path.join(self.image_dir, image_name)

    # cv2.imread()는 BGR로 읽으므로 모델 입력과 시각화에 사용할 RGB로 바꾼다.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes, labels = read_annot(image_name, self.xml_dir)

    if self.transform:
      # 이미지와 경계 상자를 같은 변환에 함께 전달해야 위치가 어긋나지 않는다.
      augs = self.transform(image=image, bboxes=bboxes, labels=labels)

      # ToTensorV2가 이미지를 (채널, 높이, 너비) 텐서로 바꾼다.
      image = augs['image']

      # Faster R-CNN의 boxes는 [xmin, ymin, xmax, ymax] 모양의 실수 텐서이고,
      # labels는 각 객체의 클래스 번호를 담은 정수 텐서이다.
      bboxes = torch.tensor(augs['bboxes'], dtype=torch.float32).reshape(-1, 4)
      labels = torch.tensor(augs['labels'], dtype=torch.int64)

    # 객체 탐지 모델은 정답을 boxes와 labels 키를 가진 딕셔너리로 받는다.
    target = {}
    target['labels'] = labels
    target['boxes'] = bboxes

    return image, target

  def __len__(self):
    # len(dataset)은 전체 이미지 개수를 반환한다.
    return len(self.image_list)

# %%
# pascal_voc 형식은 상자를 [xmin, ymin, xmax, ymax] 순서로 표현한다.
# 너무 작거나 변환 후 거의 보이지 않는 상자는 제거한다.
bbox_params = A.BboxParams(
  format='pascal_voc',
  min_area=5,
  min_visibility=0.9,
  check_each_transform =True,
  label_fields=['labels']
)

# A.Compose()는 이미지 변환을 순서대로 묶으며 bbox_params를 지정하면
# 좌우·상하 반전과 크기 변경이 경계 상자에도 동일하게 적용된다.
train_transform=A.Compose(
  [
    A.Resize(height=448, width=448),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),

    # 사전 학습된 ResNet 백본이 사용한 ImageNet 평균과 표준편차로 정규화한다.
    A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ToTensorV2()
  ],
  bbox_params=bbox_params,
  is_check_shapes=False
)

# 검증과 테스트에서는 결과가 매번 달라지지 않도록 무작위 반전을 제외한다.
test_transform=A.Compose(
  [
    A.Resize(height=448, width=448),
    A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ToTensorV2(),
  ],
  bbox_params=bbox_params,
  is_check_shapes=False
)

def collate_fn(batch):
    # 이미지마다 객체 수와 boxes 크기가 다르므로 기본 DataLoader처럼 하나의
    # 텐서로 쌓지 않고 이미지 묶음과 정답 묶음으로 각각 나누어 반환한다.
    return tuple(zip(*batch))

# %%
mask_dataset = MaskDataset("dataset", train_transform)

mask_loader = DataLoader(mask_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn)

# %%
for image, target in mask_loader:
  # PyTorch 이미지는 (채널, 높이, 너비)이므로 화면 출력용 (높이, 너비, 채널)로 바꾼다.
  img = np.transpose(image[0].numpy(), (1, 2, 0))
  new_img = draw_boxes(img, target[0]['boxes'], target[0]['labels'])

  plt.imshow(new_img)
  plt.axis('off')

  break

# %%
def get_dataset(
    root, train=0.8, val=0.1,
    train_transforms=None, test_transforms=None,
    random_seed=827
):
  # 학습 세트와 평가 세트가 서로 다른 transform을 사용하도록 Dataset을 따로 만든다.
  train_origin = MaskDataset(
    root,
    transform=train_transforms
  )
  eval_origin = MaskDataset(
    root,
    transform=test_transforms
  )

  train_samples = int(len(train_origin) * train)
  val_samples = int(len(train_origin) * val)
  test_samples = len(train_origin) - train_samples - val_samples

  # 고정된 난수 생성기로 인덱스를 섞어 매번 같은 데이터 분할을 재현한다.
  generator = torch.Generator().manual_seed(random_seed)
  indices = torch.randperm(len(train_origin), generator=generator).tolist()
  train_indices = indices[:train_samples]
  val_indices = indices[train_samples:train_samples + val_samples]
  test_indices = indices[train_samples + val_samples:]

  trainset = torch.utils.data.Subset(train_origin, train_indices)
  valset = torch.utils.data.Subset(eval_origin, val_indices)
  testset = torch.utils.data.Subset(eval_origin, test_indices)

  return trainset, valset, testset

trainset, valset, testset = get_dataset('dataset')
print(len(trainset), len(valset), len(testset))

# %% [markdown]
# ### 1.3 FasterRCNN 모델

# %%
def load_model(num_classes=3):
  # COCO 데이터로 사전 학습된 ResNet50-FPN 기반 Faster R-CNN을 불러온다.
  # 사전 학습된 백본은 일반적인 이미지 특징을 이미 학습한 상태이다.
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

  # 기존 분류기가 받는 입력 특징 수는 유지하고 출력 클래스 수만 교체한다.
  # num_classes=3은 배경(0), 미착용 계열(1), 착용(2)을 뜻한다.
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

# %%
model = load_model()

# 객체 탐지 모델은 이미지 한 장도 배치 차원을 포함한 (1, 3, 448, 448)로 확인한다.
summary(model, (1, 3, 448, 448))

# %% [markdown]
# ### 1.5 모델 학습

# %%
def train(model, dataloader, optimizer, epoch, device):
  # train() 모드의 Faster R-CNN은 이미지와 정답을 받으면 예측값 대신
  # 분류·경계 상자·RPN 등에 대한 손실 딕셔너리를 반환한다.
  model.train()

  running_loss = 0

  with tqdm(dataloader) as pbar:
    for i, (data, target) in enumerate(pbar):
      # 객체 탐지 입력은 이미지마다 크기와 객체 수가 다를 수 있어 리스트로 전달한다.
      data = list(image.to(device) for image in data)

      # boxes와 labels를 포함한 각 정답 딕셔너리의 텐서를 계산 장치로 옮긴다.
      target = [{k: v.to(device) for k, v in t.items()} for t in target]

      loss_dict = model(data, target)

      # 여러 종류의 탐지 손실을 모두 더해 한 번의 역전파에 사용한다.
      losses = sum(loss for loss in loss_dict.values())

      # 이전 배치의 기울기를 지우고 현재 손실로 새 기울기를 계산한 뒤 가중치를 갱신한다.
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()

      running_loss += losses
      pbar.set_postfix(loss=losses.item())

    print(
        f"[{epoch}/{EPOCH}]",
        f"train loss: {running_loss/len(dataloader):.4f}"
    )

  return running_loss/len(dataloader)

# %%
def validation(model, dataloader, epoch, device,
               threshold_score=0.5, threshold_iou=0.5):
  # eval()은 BatchNorm과 Dropout을 추론 방식으로 동작시킨다.
  model.eval()

  tps = []
  fps = []
  fns = []

  with tqdm(dataloader) as pbar:
    # 검증에서는 역전파가 필요 없으므로 계산 기록을 만들지 않는다.
    with torch.no_grad():
      for i, (data, target) in enumerate(pbar):
        data = list(image.to(device) for image in data)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        # eval() 모드의 Faster R-CNN은 boxes, labels, scores가 담긴 예측 딕셔너리를 반환한다.
        predictions = model(data)

        # 마지막 배치만이 아니라 DataLoader의 모든 이미지에 대해 평가한다.
        for j in range(len(target)):
          true_boxes = target[j]['boxes'].cpu()
          true_labels = target[j]['labels'].cpu()
          pred_boxes = predictions[j]['boxes'].cpu()
          pred_labels = predictions[j]['labels'].cpu()
          pred_scores = predictions[j]['scores'].cpu()

          # 모델이 일정 점수 이상으로 확신한 예측만 남긴다.
          keep = pred_scores >= threshold_score
          pred_boxes = pred_boxes[keep]
          pred_labels = pred_labels[keep]

          matched_true = set()
          true_positives = 0

          if len(pred_boxes) and len(true_boxes):
            # 각 예측을 같은 클래스의 아직 매칭되지 않은 정답 하나와만 연결한다.
            iou_matrix = torchvision.ops.box_iou(pred_boxes, true_boxes)
            for pred_idx in range(len(pred_boxes)):
              valid = true_labels == pred_labels[pred_idx]
              candidate_ious = iou_matrix[pred_idx].clone()
              candidate_ious[~valid] = 0
              for true_idx in matched_true:
                candidate_ious[true_idx] = 0

              best_iou, best_true_idx = candidate_ious.max(dim=0)
              if best_iou >= threshold_iou:
                true_positives += 1
                matched_true.add(best_true_idx.item())

          false_positives = len(pred_boxes) - true_positives
          false_negatives = len(true_boxes) - true_positives

          tps.append(true_positives)
          fps.append(false_positives)
          fns.append(false_negatives)

  precision = sum(tps) / max(sum(tps) + sum(fps), 1)
  recall = sum(tps) / max(sum(tps) + sum(fns), 1)

  # F1 점수는 정밀도와 재현율의 조화 평균이다.
  f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)

  print(f'[{epoch}/{EPOCH}] precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f} \n')

  return precision, recall, f1

# %%
EPOCH = 10
BATCH_SIZE = 4
NUM_WORKERS = 2
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY=0.0005
LR_SETP=6
LR_GAMMA=0.7

trainset, valset, testset = get_dataset(
  "dataset",
  train_transforms=train_transform,
  test_transforms=test_transform
)

# 학습 데이터는 순서를 섞고, 검증과 테스트 데이터는 정해진 순서로 가져온다.
# 객체 수가 서로 다른 정답을 처리하기 위해 모두 같은 collate_fn을 사용한다.
train_loader = DataLoader(
  dataset=trainset,
  shuffle=True,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  collate_fn=collate_fn
)
val_loader = DataLoader(
  dataset=valset,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  collate_fn=collate_fn
)
test_loader = DataLoader(
  dataset=testset,
  batch_size=BATCH_SIZE,
  num_workers=NUM_WORKERS,
  collate_fn=collate_fn
)

model = load_model()

# SGD는 기울기 방향으로 모델 가중치를 갱신한다.
# momentum은 이전 갱신 방향을 일부 유지하고 weight_decay는 과도한 가중치를 억제한다.
optimizer = optim.SGD(model.parameters(), lr=LR,
                      momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# StepLR은 6 epoch마다 학습률에 0.7을 곱해 점차 작게 만든다.
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=LR_SETP, gamma=LR_GAMMA)

model = model.to(device)

max_f1 = 0
for epoch in range(EPOCH):
  # get_last_lr()는 현재 optimizer에 적용되는 학습률을 리스트로 반환한다.
  print("LR:", scheduler.get_last_lr())

  tloss = train(model, train_loader, optimizer, epoch, device)
  p, r, f1 = validation(model, val_loader, epoch, device)

  scheduler.step()

  # 검증 F1 점수가 지금까지 가장 높을 때만 모델의 가중치 딕셔너리를 저장한다.
  if f1 > max_f1:
    # Chapter 2 오토인코더의 best.pth와 충돌하지 않도록 마스크 모델 전용 이름을 사용한다.
    torch.save(model.state_dict(), "mask_best.pth")
    max_f1 = f1

p, r, f1 = validation(model, test_loader, epoch, device)

# %% [markdown]
# ### 1.6 추론

# %%
model = load_model()

# 저장된 state_dict를 같은 구조의 새 모델에 적용한다.
# GPU에서 저장한 파일도 CPU 환경에서 읽으려면 torch.load()에 map_location을 지정할 수 있다.
model.load_state_dict(torch.load("mask_best.pth", map_location=device))
model = model.to(device)

# %%
eval_transform = A.Compose(
  [
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToTensorV2(),
  ]
)

def inference(model, image, transform,
              threshold_score=0.5, threshold_nms=0.5):
  # Albumentations는 image 키로 이미지를 받아 전처리 결과 딕셔너리를 반환한다.
  t_image = transform(image=image)['image']

  # 한 장의 (채널, 높이, 너비) 이미지 앞에 배치 차원을 추가한다.
  t_image = t_image.unsqueeze(0).to(device)

  model.eval()
  with torch.no_grad():
    pred = model(t_image)

  # nms()는 서로 많이 겹치는 예측 상자 중 점수가 높은 상자만 남긴다.
  # 반환값은 삭제되지 않고 유지된 상자의 인덱스이다.
  keep_boxes = torchvision.ops.nms(
      pred[0]['boxes'].cpu(), pred[0]['scores'].cpu(), threshold_nms)

  # NMS를 통과한 상자 중 신뢰 점수가 threshold_score를 넘는 것만 선택한다.
  score_filter = pred[0]['scores'].cpu().numpy()[keep_boxes] > threshold_score

  # 화면에 그릴 수 있도록 선택한 상자와 클래스 번호를 NumPy 배열로 변환한다.
  test_boxes = pred[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
  test_labels = pred[0]['labels'].cpu().numpy()[keep_boxes][score_filter]

  return test_boxes, test_labels

# %%
img_path = "dataset/images/maksssksksss457.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes, labels = inference(model, image, eval_transform)

new_img = draw_boxes(image, boxes, labels)
plt.imshow(new_img)

