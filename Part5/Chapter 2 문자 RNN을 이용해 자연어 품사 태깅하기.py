# Converted from: Chapter 2 문자 RNN을 이용해 자연어 품사 태깅하기.ipynb
# Original .ipynb is preserved alongside this file.

# ## Chapter 2 문자 RNN을 이용해 자연어 품사 태깅하기

# Jupyter-only command: !pip install konlpy torchinfo

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchinfo import summary

from konlpy.corpus import kolaw
from konlpy.tag import Okt

# CUDA GPU를 사용할 수 있으면 cuda, 아니면 CPU에서 계산하도록 장치를 정한다.
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

seed = 827
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
# CUDA를 사용할 때는 모든 GPU의 난수도 같은 값으로 고정한다.
if device.type == "cuda":
  torch.cuda.manual_seed_all(seed)  # gpu seed

# ### 2.2 품사 태깅 데이터

# KoNLPy에 포함된 대한민국 헌법 말뭉치를 문자열로 읽는다.
with kolaw.open('constitution.txt') as f:
  corpus = f.read()
print(corpus[:40])

# 중복 문자를 제거하고 정렬한 뒤, 패딩에 사용할 빈 문자열을 0번에 추가한다.
vocab = list(sorted(set(corpus)))
vocab.insert(0, '')
# 각 문자를 신경망이 처리할 정수 번호로 바꾸기 위한 사전이다.
char_to_index = {char: i for i, char in enumerate(vocab)}
char_to_index

def text_to_sequence(text):
  # 문자열의 각 문자를 char_to_index에 등록된 정수 번호로 변환한다.
  return [char_to_index[char] for char in text]
text_to_sequence("대한민국")

# Okt는 한국어 문장을 형태소 단위로 나누고 각 형태소의 품사를 반환한다.
okt = Okt()
tagged = okt.pos(corpus)
tagged

# zip(*tagged)는 [(단어, 품사), ...]를 단어 묶음과 품사 묶음으로 분리한다.
words, tags = zip(*tagged)
print(words)
print(tags)

okt.tagset

words_len = [len(word) for word in words]
print(len(words_len))
max_len = max(words_len)
mean_len = sum(words_len) / len(words_len)
print("MAX:", max_len)
print("MEAN:", mean_len)

plt.hist(words_len, bins=9)
plt.show()

train_x = [torch.Tensor(text_to_sequence(word)) for word in words]

# pad_sequence()는 길이가 다른 단어들을 가장 긴 단어 길이에 맞춰 0으로 채운다.
# batch_first=True이므로 결과 모양은 (단어 수, 최대 문자 수)가 된다.
nn.utils.rnn.pad_sequence(train_x, batch_first=True).size()

# 품사 이름을 정렬해 분류 정답 번호와 품사 이름이 항상 같은 순서로 대응하게 한다.
class_names = list(sorted(set(tags)))
class_names

class TokenizerDataset(Dataset):
  def __init__(self):
    corpus =  kolaw.open('constitution.txt').read()
    tokenizer = Okt()

    self.vocab = list(sorted(set(corpus)))
    self.vocab.insert(0, '')
    self.char_to_index = {char: i for i, char in enumerate(self.vocab)}

    # data에는 형태소 문자열, labels에는 그 형태소의 품사 문자열이 들어간다.
    self.data, self.labels = zip(*tokenizer.pos(corpus))
    # Embedding 계층의 입력은 정수 인덱스여야 하므로 LongTensor로 변환한다.
    self.data = [torch.LongTensor(self.text_to_sequence(word)) for word in self.data]
    # 모든 단어를 최대 길이 9에 맞춰 0으로 패딩해 하나의 2차원 텐서로 만든다.
    self.data = nn.utils.rnn.pad_sequence(self.data, batch_first=True)

    self.class_names = list(sorted(set(self.labels)))
    # CrossEntropyLoss의 정답은 문자열이 아니라 0부터 시작하는 클래스 번호여야 한다.
    self.labels = [self.class_names.index(label) for label in self.labels]

  def text_to_sequence(self, text):
    return [self.char_to_index[char] for char in text]

  def sequence_to_text(self, sequence):
    # 텐서의 문자 번호를 vocab에서 다시 찾아 사람이 읽는 문자열로 복원한다.
    if isinstance(sequence, torch.Tensor):
      sequence = [int(i) for i in sequence]
    return ''.join([self.vocab[i] for i in sequence])

  def __getitem__(self, i):
    # DataLoader가 i번째 패딩된 단어와 품사 번호를 한 쌍으로 가져간다.
    return self.data[i], self.labels[i]

  def __len__(self):
    # 전체 형태소 개수를 Dataset 길이로 반환한다.
    return len(self.data)

dataset = TokenizerDataset()

data, label = dataset[0]
print(dataset.sequence_to_text(data), data)
print(dataset.class_names[label], label)

def get_dataset(train=0.8, val=0.1,random_seed=827):
  origin = TokenizerDataset()

  # 전체 데이터의 80%는 학습, 10%는 검증, 나머지 10%는 테스트에 사용한다.
  train_samples = int(len(origin) * train)
  val_samples = int(len(origin) * val)
  test_samples = len(origin) - train_samples - val_samples

  # 같은 seed를 사용하면 실행할 때마다 동일한 샘플 구성이 만들어진다.
  torch.manual_seed(random_seed)
  trainset, valset, testset = torch.utils.data.random_split(
    origin,
    (train_samples, val_samples, test_samples),
  )

  return trainset, valset, testset

trainset, valset, testset = get_dataset()
print(len(trainset), len(valset), len(testset))

# ### 2.3 RNN 분류기 모델

class PosTagger(nn.Module):
  def __init__(self, vocab_size, hidden_dims, layers, seq_len, classes):
    super(PosTagger, self).__init__()
    self.hidden_dims = hidden_dims
    self.seq_len = seq_len

    # Embedding은 각 문자 번호를 hidden_dims 길이의 학습 가능한 벡터로 바꾼다.
    self.embedding = nn.Embedding(vocab_size, hidden_dims)
    # 2층 LSTM이 문자 순서를 읽는다. batch_first=True이면 입력은 (배치, 문자 수, 특성)이다.
    self.lstm = nn.LSTM(hidden_dims, hidden_dims, layers, batch_first=True)
    # 모든 문자 위치의 LSTM 출력을 이어 붙인 뒤 품사별 점수(logit)로 변환한다.
    self.fc = nn.Linear(hidden_dims*seq_len, classes, bias=True)

  def forward(self, x):
    x = self.embedding(x)
    # LSTM의 두 번째 반환값인 마지막 hidden/cell state는 여기서는 사용하지 않는다.
    x, _ = self.lstm(x)
    # (배치, 문자 수, 은닉 크기)를 (배치, 문자 수×은닉 크기)로 펼친다.
    x = x.reshape((-1, self.hidden_dims * self.seq_len))
    x = self.fc(x)
    return x

model = PosTagger(len(dataset.vocab), 128, 2, 9, len(dataset.class_names))
# summary()는 예시 입력을 통과시켜 계층별 출력 모양과 파라미터 수를 보여준다.
summary(model, input_data=torch.randint(0, 380, (2, 9)))

# ### 2.4 모델 학습

def train(model, dataloader, criterion, optimizer, epoch, device):
  # train()은 Dropout·BatchNorm 등이 학습 방식으로 동작하도록 모델을 학습 모드로 전환한다.
  model.train()

  running_loss = 0
  correct = 0

  with tqdm(dataloader) as pbar:
    for i, (data, targets) in enumerate(pbar):
      # 모델과 입력·정답이 같은 CPU 또는 GPU 메모리에 있어야 계산할 수 있다.
      data, targets = data.to(device), targets.to(device)

      # 이전 배치에서 계산된 기울기가 누적되지 않도록 0으로 초기화한다.
      optimizer.zero_grad()
      outputs = model(data)
      loss = criterion(outputs, targets)
      # backward()가 기울기를 계산하고 step()이 그 기울기로 가중치를 갱신한다.
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

      # 정확도 계산은 가중치 학습에 쓰지 않으므로 계산 과정을 기록하지 않는다.
      with torch.no_grad():
        # 각 샘플에서 점수가 가장 큰 품사 번호를 예측값으로 선택한다.
        _, predicted = torch.max(outputs, 1)
        correct += predicted.eq(targets.view_as(predicted)).sum()

    data_num = len(dataloader.dataset)
    acc = 100. * correct / data_num

  return running_loss/len(dataloader), acc

def validation(model, dataloader, criterion, epoch, device):
  # eval()은 모델을 평가 모드로 전환한다.
  model.eval()

  correct = 0
  running_loss = 0.

  with tqdm(dataloader) as pbar:
    # 검증은 가중치를 변경하지 않으므로 기울기 계산과 기록을 끈다.
    with torch.no_grad():
      for i, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        outputs = model(data)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

        _, predicted = torch.max(outputs, 1)
        correct += predicted.eq(targets.view_as(predicted)).sum()

  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num

  return running_loss/len(dataloader), acc

EPOCH = 10
BATCH_SIZE = 128
NUM_WORKERS = 1
LR = 0.001

trainset, valset, testset = get_dataset()

# dataloader
train_loader = DataLoader(
  dataset=trainset,
  # 학습 데이터는 Epoch마다 순서를 섞고 최대 128개씩 한 배치로 묶는다.
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

vocab_len = len(trainset.dataset.vocab)
classes = len(trainset.dataset.class_names)
model = PosTagger(vocab_len, 128, 2, 9, classes)

# Optimizer, Loss, Scheduler
# Adam은 model.parameters()에 포함된 학습 가능한 가중치를 갱신한다.
optimizer = optim.Adam(model.parameters(), lr=LR)
# CrossEntropyLoss는 품사별 점수와 int64 품사 번호를 비교해 분류 손실을 계산한다.
criterion = nn.CrossEntropyLoss()
# StepLR은 3 Epoch마다 학습률에 0.5를 곱해 점차 작게 만든다.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

model = model.to(device)
criterion = criterion.to(device)

max_acc = 0
# Start Training
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  tloss, tacc = train(model, train_loader, criterion, optimizer, epoch, device)
  print(f"[{epoch}/{EPOCH}] train loss: {tloss:.4f} train acc: {tacc:.2f}%")
  vloss, vacc = validation(model, val_loader, criterion, epoch, device)
  print(f"[{epoch}/{EPOCH}] valid loss: {vloss:.4f} valid acc: {vacc:.2f}%\n")

  scheduler.step()

  # 검증 정확도가 이전 최고 기록보다 높을 때만 모델 가중치를 저장한다.
  if vacc.item() > max_acc:
    torch.save(model.state_dict(), "pos_tagger_best.pth")
    max_acc = vacc.item()

# 저장된 최고 성능 가중치를 다시 불러와 별도로 남겨 둔 테스트 데이터에서 평가한다.
model.load_state_dict(torch.load("pos_tagger_best.pth", map_location=device))
tloss, tacc = validation(model, test_loader, criterion, epoch, device)
print(f"\n[TEST] loss: {tloss:.4f} acc: {tacc:.2f}%")

def inference(text, dataset, model):
  input = dataset.text_to_sequence(text)
  # 학습 때 사용한 고정 길이 9에 맞도록 오른쪽을 0으로 채운다.
  # 9자보다 긴 입력은 음수 패딩이 되어 오류가 나므로 이 예제는 9자 이하 단어용이다.
  input = torch.nn.functional.pad(torch.LongTensor(input), (0, 9-len(input)))
  # 배치 차원이 없는 (문자 수,) 입력도 Embedding과 LSTM이 단일 샘플로 처리할 수 있다.
  logit = model(input)
  # 가장 큰 품사 점수의 위치를 클래스 번호로 선택한다.
  output = torch.argmax(logit)
  return dataset.class_names[output.item()]

dataset = TokenizerDataset()

vocab_len = len(dataset.vocab)
classes = len(dataset.class_names)
model = PosTagger(vocab_len, 128, 2, 9, classes)
# map_location="cpu"는 GPU에서 저장한 가중치도 CPU 메모리로 옮겨 불러오게 한다.
model.load_state_dict(torch.load("pos_tagger_best.pth", map_location="cpu"))

print(inference("모든", dataset, model))
print(inference("권력", dataset, model))
print(inference("은", dataset, model))
print(inference("국민", dataset, model))
print(inference("으로부터", dataset, model))
print(inference("나온다", dataset, model))
