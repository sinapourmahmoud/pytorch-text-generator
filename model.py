
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import numpy as np
import os
import io
import re
from tqdm.notebook import trange, tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from torch.distributions import Categorical

from torchtext.datasets import WikiText2, EnWik9, AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torchtext.data.functional import sentencepiece_tokenizer, load_sp_model
from torchtext.data.functional import generate_sp_model

learning_rate = 1e-4

nepochs = 20

batch_size = 32

max_len = 64

dataset_root = "./datasets"
# this line of code is for downloading the train datasets
# dataset_train = AG_NEWS(root=dataset_root,split="train")
# dataset_test = AG_NEWS(root=dataset_root,split="test")

# data = next(iter(dataset_train))
# data = next(iter(dataset_test))
# print(data)


# this lones are for training and tokenization of the vocab

# with open(os.path.join(dataset_root,"datasets/AG_NEWS/train.csv")) as f1:
#     with open(os.path.join(dataset_root,"datasets/AG_NEWS/data.txt"),"w") as f2:
#         for i,line in enumerate(f1):
#             text_only = "".join(line.split(",")[1:])
#             filtered = re.sub(r'\\|\\n|;', ' ', text_only.replace('"', ' ').replace('\n', ' '))
#             f2.write(filtered.lower() + "\n")

# generate_sp_model(os.path.join(dataset_root,"datasets/AG_NEWS/data.txt"),model_prefix="my_tokenizer",vocab_size=2000)




class AGNews(Dataset):

    def __init__(self,test_train):
        self.df = pd.read_csv(os.path.join(dataset_root,"datasets/AG_NEWS/"+test_train+".csv"),names=["Class", "Title", "Content"])

        self.df.fillna("",inplace=True)

        self.df["Article"] = self.df["Title"] + ":" + self.df["Content"]

        self.df.drop(["Title","Content"],axis=1,inplace=True)

        self.df['Article'] = self.df['Article'].str.replace(r'\\n|\\|\\r|\\r\\n|\n|"', ' ', regex=True)


    def __getitem__(self, index):
        return self.df.iloc[index]["Article"].lower()

    def __len__(self):
        return len(self.df)


dataset_train = AGNews("train")
dataset_test = AGNews("test")


train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=0)

sp_model = load_sp_model("my_tokenizer.model")

tokenizer = sentencepiece_tokenizer(sp_model)

# for testing the tokenizer
# for token in tokenizer(["unbelievable"]):
#     print(token)


def yield_token(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield [line.split("\t")[0]]


vocab = build_vocab_from_iterator(
    yield_token("my_tokenizer.vocab"),
    specials=['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)

vocab.set_default_index(vocab['<unk>'])

class TokenDrop(nn.Module):

    def __init__(self,  prob=0.1, pad_token=0, num_special=4):
        
        super(self,TokenDrop).__init__()

        self.prob = prob
        self.pad_token = pad_token
        self.num_special = num_special


    def __call__(self,sample):

        mask = torch.bernoulli(self.prob * torch.ones_like(sample)).long()
        can_drop = (sample >= self.num_special).long()

        mask = mask * can_drop

        replace_with = (self.pad_token * torch.ones_like(sample)).long()

        sample = (1 - mask) * sample + mask * replace_with

        return sample


train_transform = T.Sequential(
    T.SentencePieceTokenizer("my_tokenizer.model"),
    T.VocabTransform(vocab=vocab),
    T.AddToken(1,begin=True),
    T.Truncate(max_seq_len=max_len),
    T.AddToken(2,begin=False),
    T.ToTensor(padding_value=0)
)

gen_transfor = T.Sequential(
    T.SentencePieceTokenizer("my_tokenizer.model"),
    T.VocabTransform(vocab=vocab),
    T.AddToken(1,begin=True),
    T.ToTensor(padding_value=0)
)

class LSTM(nn.Module):

    def __init__(self, num_emb, num_layers=1,emb_size=128,hidden_size=128):
        super(self,LSTM).__init__()

        self.embdding = nn.Embedding(num_emb,emb_size)

        self.mlp_in = nn.Sequential(
            nn.Linear(emb_size,emb_size),
            nn.LayerNorm(emb_size),
            nn.ELU(),
            nn.Linear(emb_size,emb_size)
        )

        self.lstm = nn.LSTM(input_size=emb_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True,dropout=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_size,hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.ELU(),
            nn.Linear(hidden_size//2,num_emb)
        )


    def forward(self,input_seq,hidd,memo):
        input_embs = self.embdding(input_seq)
        input_embs = self.mlp_in(input_embs)

        output,(hidden_out,memo_out) = self.lstm(input_embs,(hidd,memo))
        return self.mlp_out(output),hidden_out,memo_out



device = ("cuda" if torch.cuda.is_available() else "cpu")

emb_size = 256
hidden_size = 1024

# Define number of layers for the LSTM model
num_layers = 2

lstm_generator = LSTM(num_emb=len(vocab),num_layers=num_layers,emb_size=emb_size,hidden_size=hidden_size).to(device)


optimizer = optim.Adam(lstm_generator.parameters(),lr=learning_rate,weight_decay=1e-4)

loss_fn = nn.CrossEntropyLoss()

t = TokenDrop(prob=0.1)

training_loss_logger = []

entropy_logger = []



# Training loop
for epoch in trange(0, nepochs, leave=False, desc="Epoch"):
    lstm_generator.train()
    steps = 0


    # Iterate over batches in training data loader
    for text in tqdm(train_loader, desc="Training", leave=False):

        text_tokens = train_transform(list(text)).to(device)
        bs = text_tokens.shape[0]

        # Randomly drop input tokens
        input_text = t(text_tokens[:, 0:-1])
        output_text = text_tokens[:, 1:]

        hidden = torch.zeros(num_layers, bs, hidden_size, device=device)
        memory = torch.zeros(num_layers, bs, hidden_size, device=device)

        pred, hidden, memory = lstm_generator(input_text, hidden, memory)

        # Calculate loss
        loss = loss_fn(pred.transpose(1, 2), output_text)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training loss
        training_loss_logger.append(loss.item())

        with torch.no_grad():
            dist = Categorical(logits=pred)
            entropy_logger.append(dist.entropy().mean().item())


