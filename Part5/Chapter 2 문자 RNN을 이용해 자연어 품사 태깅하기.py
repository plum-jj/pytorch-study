# %%
# Converted from: Chapter 2 문자 RNN을 이용해 자연어 품사 태깅하기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 2 문자 RNN을 이용해 자연어 품사 태깅하기

# %%
# Jupyter-only command: !pip install konlpy torchinfo

# %%
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

# %%
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
print(is_cuda, device)

# %%
seed = 827
random.seed(seed)         # python seed
np.random.seed(seed)      # numpy seed
torch.manual_seed(seed)   # torch seed
if device == 'cuda':
  torch.cuda.manual_seed_all(seed)  # gpu seed

# %% [markdown]
# ### 2.2 품사 태깅 데이터

# %%
with kolaw.open('constitution.txt') as f:
  corpus = f.read()
print(corpus[:40])

# %%
vocab = list(sorted(set(corpus)))
vocab.insert(0, '')
char_to_index = {char: i for i, char in enumerate(vocab)}
char_to_index

# %%
def text_to_sequence(text):
  return [char_to_index[char] for char in text]
text_to_sequence("대한민국")

# %%
okt = Okt()
tagged = okt.pos(corpus)
tagged

# %%
words, tags = zip(*tagged)
print(words)
print(tags)

# %%
okt.tagset

# %%
words_len = [len(word) for word in words]
print(len(words_len))
max_len = max(words_len)
mean_len = sum(words_len) / len(words_len)
print("MAX:", max_len)
print("MEAN:", mean_len)

# %%
plt.hist(words_len, bins=9)
plt.show()

# %%
train_x = [torch.Tensor(text_to_sequence(word)) for word in words]

# %%
nn.utils.rnn.pad_sequence(train_x, batch_first=True).size()

# %%
class_names = list(sorted(set(tags)))
class_names

# %%
class TokenizerDataset(Dataset):
  def __init__(self):
    corpus =  kolaw.open('constitution.txt').read()
    tokenizer = Okt()

    self.vocab = list(sorted(set(corpus)))
    self.vocab.insert(0, '')
    self.char_to_index = {char: i for i, char in enumerate(self.vocab)}

    self.data, self.labels = zip(*tokenizer.pos(corpus))
    self.data = [torch.LongTensor(self.text_to_sequence(word)) for word in self.data]
    self.data = nn.utils.rnn.pad_sequence(self.data, batch_first=True)

    self.class_names = list(sorted(set(self.labels)))
    self.labels = [self.class_names.index(label) for label in self.labels]

  def text_to_sequence(self, text):
    return [self.char_to_index[char] for char in text]

  def sequence_to_text(self, sequence):
    if isinstance(sequence, torch.Tensor):
      sequence = [int(i) for i in sequence]
    return ''.join([self.vocab[i] for i in sequence])

  def __getitem__(self, i):
    return self.data[i], self.labels[i]

  def __len__(self):
    return len(self.data)

# %%
dataset = TokenizerDataset()

# %%
data, label = dataset[0]
print(dataset.sequence_to_text(data), data)
print(dataset.class_names[label], label)

# %%
def get_dataset(train=0.8, val=0.1,random_seed=827):
  origin = TokenizerDataset()

  train_samples = int(len(origin) * train)
  val_samples = int(len(origin) * val)
  test_samples = len(origin) - train_samples - val_samples

  torch.manual_seed(random_seed)
  trainset, valset, testset = torch.utils.data.random_split(
    origin,
    (train_samples, val_samples, test_samples),
  )

  return trainset, valset, testset

trainset, valset, testset = get_dataset()
print(len(trainset), len(valset), len(testset))

# %% [markdown]
# ### 2.3 RNN 분류기 모델

# %%
class PosTagger(nn.Module):
  def __init__(self, vocab_size, hidden_dims, layers, seq_len, classes):
    super(PosTagger, self).__init__()
    self.hidden_dims = hidden_dims
    self.seq_len = seq_len

    self.embedding = nn.Embedding(vocab_size, hidden_dims)
    self.lstm = nn.LSTM(hidden_dims, hidden_dims, layers, batch_first=True)
    self.fc = nn.Linear(hidden_dims*seq_len, classes, bias=True)

  def forward(self, x):
    x = self.embedding(x)
    x, _ = self.lstm(x)
    x = x.reshape((-1, self.hidden_dims * self.seq_len))
    x = self.fc(x)
    return x

# %%
model = PosTagger(len(dataset.vocab), 128, 2, 9, len(dataset.class_names))
summary(model, input_data=torch.randint(0, 380, (2, 9)))

# %% [markdown]
# ### 2.4 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0
  correct = 0

  with tqdm(dataloader) as pbar:
    for i, (data, targets) in enumerate(pbar):
      data, targets = data.to(device), targets.to(device)

      optimizer.zero_grad()
      outputs = model(data)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

      with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        correct += predicted.eq(targets.view_as(predicted)).sum()

    data_num = len(dataloader.dataset)
    acc = 100. * correct / data_num

  return running_loss/len(dataloader), acc

# %%
def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  correct = 0
  running_loss = 0.

  with tqdm(dataloader) as pbar:
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

# %%
EPOCH = 10
BATCH_SIZE = 128
NUM_WORKERS = 1
LR = 0.001

trainset, valset, testset = get_dataset()

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

vocab_len = len(trainset.dataset.vocab)
classes = len(trainset.dataset.class_names)
model = PosTagger(vocab_len, 128, 2, 9, classes)

# Optimizer, Loss, Scheduler
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
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

  if vacc > max_acc:
    torch.save(model.state_dict(), "best.pth")

# %%
tloss, tacc = validation(model, test_loader, criterion, epoch, device)
print(f"\n[TEST] loss: {tloss:.4f} acc: {tacc:.2f}%")

# %%
def inference(text, dataset, model):
  input = dataset.text_to_sequence(text)
  input = torch.nn.functional.pad(torch.LongTensor(input), (0, 9-len(input)))
  logit = model(input)
  output = torch.argmax(logit)
  return dataset.class_names[output.item()]

# %%
dataset = TokenizerDataset()

vocab_len = len(dataset.vocab)
classes = len(dataset.class_names)
model = PosTagger(vocab_len, 128, 2, 9, classes)
model.load_state_dict(torch.load("best.pth"))

# %%
print(inference("모든", dataset, model))
print(inference("권력", dataset, model))
print(inference("은", dataset, model))
print(inference("국민", dataset, model))
print(inference("으로부터", dataset, model))
print(inference("나온다", dataset, model))

# %%


# %%


# %%


# %%


