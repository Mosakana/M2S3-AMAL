import itertools
import logging

from skimage.color.rgb_colors import crimson
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse, parse_incr

from seq2seq import Seq2Seq


logging.basicConfig(level=logging.INFO)

DATA_PATH = "../data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))



logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu", encoding='UTF-8')
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)
raw_train = [parse(x)[0] for x in data_file if len(x)>1]
data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu", encoding='UTF-8')
raw_dev = [parse(x)[0] for x in data_file if len(x)>1]
data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu", encoding='UTF-8')
raw_test = [parse(x)[0] for x in data_file if len(x)>1]

# train_data = TaggingDataset(raw_train, words, tags, True)
# train_data = TaggingDataset(parse_incr(data_file))
dev_data = TaggingDataset(raw_dev, words, tags, True)
test_data = TaggingDataset(raw_test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)

#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)

criterion = nn.CrossEntropyLoss()

print(len(tags))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(100, 256, 2, 100)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

writer = SummaryWriter('./outputs/tblogs/')
logging.info("Starting training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, len(tags)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)
    logging.info("Epoch %d: loss = %.3f", epoch, total_loss / len(train_loader))

    model.eval()
    total_loss = 0
    for i, (x, y) in enumerate(dev_loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output.view(-1, len(tags)), y.view(-1))
        total_loss += loss.item()
    logging.info("Epoch %d: dev loss = %.3f", epoch, total_loss / len(dev_loader))





