import io

import torch
import torch.nn as nn

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import sentencepiece_tokenizer, load_sp_model


TOKENIZER_MODEL_PATH = "my_tokenizer.model"
TOKENIZER_VOCAB_PATH = "my_tokenizer.vocab"

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

EMB_SIZE = 256
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
MAX_LEN = 64


def yield_token(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield [line.split("\t")[0]]


def load_tokenizer_and_vocab(vocab_path=TOKENIZER_VOCAB_PATH, model_path=TOKENIZER_MODEL_PATH):
    sp_model = load_sp_model(model_path)

    tokenizer = sentencepiece_tokenizer(sp_model)

    vocab = build_vocab_from_iterator(
        yield_token(vocab_path),
        specials=['<pad>', '<sos>', '<eos>', '<unk>'],
        special_first=True
    )

    vocab.set_default_index(vocab['<unk>'])
    return tokenizer,vocab
    




class LSTM(nn.Module):

    def __init__(self, num_emb, num_layers=1,emb_size=128,hidden_size=128):
        super(LSTM,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embdding = nn.Embedding(num_emb,emb_size)

        self.mlp_in = nn.Sequential(
            nn.Linear(emb_size,emb_size),
            nn.LayerNorm(emb_size),
            nn.ELU(),
            nn.Linear(emb_size,emb_size)
        )

        self.lstm = nn.LSTM(input_size=emb_size,hidden_size=hidden_size,
                            num_layers=num_layers,batch_first=True,dropout=0.25)

        self.mlp_out = nn.Sequential(
            nn.Linear(hidden_size,hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.ELU(),
            nn.Linear(hidden_size//2,num_emb)
        )
        
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        memory = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return hidden, memory


    def forward(self,input_seq,hidd,memo):
        input_embs = self.embdding(input_seq)
        input_embs = self.mlp_in(input_embs)

        output,(hidden_out,memo_out) = self.lstm(input_embs,(hidd,memo))
        return self.mlp_out(output),hidden_out,memo_out





class TokenDrop(nn.Module):

    def __init__(self,  prob=0.1, pad_token=0, num_special=4):
        
        super(TokenDrop,self).__init__()

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