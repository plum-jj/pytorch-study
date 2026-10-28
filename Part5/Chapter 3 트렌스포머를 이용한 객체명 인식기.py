# %%
# Converted from: Chapter 3 트렌스포머를 이용한 객체명 인식기.ipynb
# Original .ipynb is preserved alongside this file.

# %% [markdown]
# ## Chapter 3 트렌스포머를 이용한 객체명 인식기

# %%
# Jupyter-only command: !pip install -q torchinfo spacy
# Jupyter-only command: !python -m spacy download ko_core_news_sm

# %%
import os
import random
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchinfo import summary
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

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
# ### 3.2 객체명 인식 데이터
# NAVER NLP 챌린지 데이터셋
# - 14 종류의 Annotated Entities
# - 인물(PER), 학문분야(FLD), 인공물(AFW), 기관 및 단체(ORG), 지역명(LOC), 문명 및 문화 (CVL), 날짜(DAT), 시간(TIM), 숫자(NUM), 사건사고 및 행사(EVT), 동물(ANM), 식물(PLT), 금속/암석/화학물질(MAT), 의학용어/IT관련 용어(TRM)

# %%
# Jupyter-only command: !wget https://raw.githubusercontent.com/naver/nlp-challenge/master/missions/ner/data/train/train_data

# %%
import os

def train_iter(file_path="train_data", train=True):
  sentences = []
  sentence = [[], [], []]
  for line in open(file_path, encoding="utf-8"):
    line = line.strip()
    if line == "":
      sentences.append(sentence)
      sentence = [[], [], []]
    else:
      idx, ejeol, ner_tag = line.split("\t")
      # idx는 0부터 시작하도록
      sentence[0].append(int(idx))
      sentence[1].append(ejeol)
      sentence[2].append(ner_tag)
  return sentences

# %%
dl = train_iter()
dl[3]

# %%
len(dl)

# %%
indices, sentences, labels = zip(*dl)

# %%
print(sentences[0])
print(labels[0])

# %% [markdown]
# 네이버 NLP 챌린지 데이터셋은 BIO시스템을 따르는 데이터셋입니다. _B는 Begin의 약자로 해당 단어가 시작할 때, _I는 Inside의 약자로 단어 중간일 때를 의미합니다.

# %%
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

text_vocab = build_vocab_from_iterator(sentences,
                                       min_freq=1,
                                       specials=special_symbols,
                                       special_first=True)
text_vocab.set_default_index(text_vocab["<unk>"])

ner_vocab = build_vocab_from_iterator(labels,
                                      min_freq=1,
                                      specials=special_symbols,
                                      special_first=True)
ner_vocab.set_default_index(text_vocab["<unk>"])

# %%
print(ner_vocab.get_itos())

# %%
text_vocab(["오늘도", "빠지지", "말고", "프로젝트로", "시작하는", "파이토치", "공부하자","!"])

# %% [markdown]
# ### 3.3 데이터셋

# %%
class NERDataset(Dataset):
  def __init__(self, file_path="train_data"):
    sentences = []
    sentence = [[], []]
    for line in open(file_path, encoding="utf-8"):
      line = line.strip()
      if line == "":
        sentences.append(sentence)
        sentence = [[], []]
      else:
        idx, ejeol, ner_tag = line.split("\t")
        sentence[0].append(ejeol)
        sentence[1].append(ner_tag)

    self.texts, self.labels = zip(*sentences)

  def __getitem__(self, i):
    return self.texts[i], self.labels[i]

  def __len__(self):
    return len(self.texts)

# %%
ner_dataset = NERDataset()
ner_dataset[0]

# %%
max([len(text) for text in ner_dataset.texts])

# %%
def get_dataset(train=0.8, val=0.1, random_seed=827):
  origin = NERDataset()

  torch.manual_seed(random_seed)
  trainset, valset, testset = torch.utils.data.random_split(
    origin,
    (train, val, 1-train-val),
  )

  return trainset, valset, testset

trainset, valset, testset = get_dataset()
print(len(trainset), len(valset), len(testset))

# %%
# 순차적인 작업들을 하나로 묶는 헬퍼 함수
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# BOS/EOS를 추가하고 입력 순서(sequence) 인덱스에 대한 텐서를 생성하는 함수
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# 출발어(src)와 도착어(tgt) 원시 문자열들을 텐서 인덱스로 변환하는 변형(transform)
text_transforms = sequential_transforms(text_vocab, # 수치화(Numericalization)
                                        tensor_transform) # BOS/EOS를 추가하고 텐서를 생성
ner_transforms = sequential_transforms(ner_vocab,   # 수치화(Numericalization)
                                        tensor_transform) # BOS/EOS를 추가하고 텐서를 생성

# 데이터를 텐서로 조합(collate)하는 함수
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transforms(src_sample))
        tgt_batch.append(ner_transforms(tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

# %%
dataloader = DataLoader(ner_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# %%
for labels, texts in dataloader:
  print(labels)
  print(texts)
  break

# %% [markdown]
# ### 3.4 트렌스포머 모델

# %%
# 입력 인덱스의 텐서를 해당하는 토큰 임베딩의 텐서로 변환하기 위한 헬퍼 모듈(Module)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# %%
vocab_size = 10
emb_size = 32
emb = TokenEmbedding(vocab_size, emb_size)

sample = torch.randint(0, vocab_size, (seq_len, batch_size))
print(sample)
print(emb(sample).shape)

# %%
# 단어 순서 개념(notion)을 토큰 임베딩에 도입하기 위한 위치 인코딩(positional encoding)을 위한 헬퍼 모듈(Module)
class PositionalEncoding(nn.Module):
  def __init__(self, emb_size, dropout, maxlen=5000):
    super(PositionalEncoding, self).__init__()
    den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)

    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer('pos_embedding', pos_embedding)

  def forward(self, token_embedding):
    token_embedding += self.pos_embedding[:token_embedding.size(0), :]
    return self.dropout(token_embedding)

# %%
seq_len, batch_size, emb_size, dropout = 10, 4, 32, 0.1
pe = PositionalEncoding(emb_size, dropout)

sample = torch.rand((seq_len, batch_size, emb_size))
x = pe(sample).shape
print("PE:    ", pe.pos_embedding[:sample.size(0), :].shape)
print("RESULT:", x)

# %%
# Seq2Seq 신경망
class Seq2SeqTransformer(nn.Module):
  def __init__(self,
                num_encoder_layers: int,
                num_decoder_layers: int,
                emb_size: int,
                nhead: int,
                src_vocab_size: int,
                tgt_vocab_size: int,
                dim_feedforward: int = 512,
                dropout: float = 0.1):
    super(Seq2SeqTransformer, self).__init__()
    self.transformer = nn.Transformer(d_model=emb_size,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout)
    self.generator = nn.Linear(emb_size, tgt_vocab_size)
    self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
    self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
    self.positional_encoding = PositionalEncoding(
        emb_size, dropout=dropout)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
      src_emb = self.positional_encoding(self.src_tok_emb(src))
      tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
      outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                              src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
      return self.generator(outs)

    def encode(self, src, src_mask):
      return self.transformer.encoder(self.positional_encoding(
                          self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
      return self.transformer.decoder(self.positional_encoding(
                        self.tgt_tok_emb(tgt)), memory,
                        tgt_mask)

# %%
def generate_square_subsequent_mask(sz):
  mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

# %%
generate_square_subsequent_mask(4)

# %%
def create_mask(src, tgt):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# %%
src = torch.Tensor([[5], [5], [5], [1], [1]])
tgt = torch.Tensor([[5], [5], [1], [1], [1]])
create_mask(src, tgt)

# %% [markdown]
# ### 3.5 모델 학습

# %%
def train(model, dataloader, criterion, optimizer, epoch, device):
  model.train()

  running_loss = 0
  correct = 0

  with tqdm(dataloader) as pbar:
    pbar.set_description(f'Epoch - {epoch} TRAIN')
    for i, (data, targets) in enumerate(pbar):
      data, targets = data.to(device), targets.to(device)

      tgt_input = targets[:-1, :]

      src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(data, tgt_input)

      logits = model(data, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

      optimizer.zero_grad()
      tgt_out = targets[1:, :]
      loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
      torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      pbar.set_postfix(loss=loss.item())

    data_num = len(dataloader.dataset)
    acc = 100. * correct / data_num

    final_loss = running_loss/len(dataloader)
    pbar.set_postfix(loss=final_loss)

  return final_loss, acc

# %%
def validation(model, dataloader, criterion, epoch, device):
  model.eval()

  correct = 0
  running_loss = 0.

  with tqdm(dataloader) as pbar:
    pbar.set_description(f'Epoch - {epoch} VALID')
    with torch.no_grad():
      for i, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)

        tgt_input = targets[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(data, tgt_input)

        logits = model(data, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = targets[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

  data_num = len(dataloader.dataset)
  acc = 100. * correct / data_num

  final_loss = running_loss/len(dataloader)
  pbar.set_postfix(loss=final_loss)

  return final_loss, acc

# %%
EPOCH = 7
BATCH_SIZE = 64
NUM_WORKERS = 1
LR = 0.0001

trainset, valset, testset = get_dataset()

# dataloader
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

# model
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NHEAD = 8
SRC_VOCAB_SIZE = len(text_vocab)
TGT_VOCAB_SIZE = len(ner_vocab)
FFN_HID_DIM = 512
DROPOUT = 0.2

model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                           NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Optimizer, Loss, Scheduler
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

model = model.to(device)
criterion = criterion.to(device)

min_loss = 999
# Start Training
for epoch in range(EPOCH):
  print("LR:", scheduler.get_last_lr())

  start_time = timer()
  tloss, tacc = train(model, train_loader, criterion, optimizer, epoch, device)
  end_time = timer()

  start_time = timer()
  vloss, vacc = validation(model, val_loader, criterion, epoch, device)
  end_time = timer()

  scheduler.step()

  if vloss < min_loss:
    min_loss = vloss
    torch.save(model.state_dict(), "best.pth")
    print("save model")

# %% [markdown]
# ### 3.6 추론

# %%
def greedy_decode(model, src, src_mask, max_len, start_symbol):
  src = src.to(device)
  src_mask = src_mask.to(device)

  memory = model.encode(src, src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

  for i in range(max_len-1):
    memory = memory.to(device)
    tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys,
                    torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == EOS_IDX:
      break
  return ys

def translate(model: torch.nn.Module, src_sentence: str):
  model.eval()
  src = text_transforms(src_sentence).view(-1, 1)
  num_tokens = src.shape[0]
  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_decode(
    model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
  return " ".join(
      ner_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))
    ).replace("<bos>", "").replace("<eos>", "")

# %%
model.load_state_dict(torch.load("best.pth"))
tokenizer = get_tokenizer("spacy", "ko_core_news_sm")
print(translate(model, tokenizer("12월 25일 부산에서 아시안게임 개최 논의")))

