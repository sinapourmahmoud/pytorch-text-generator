
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


train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=8,drop_last=True)
test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=8)
    
        
            
