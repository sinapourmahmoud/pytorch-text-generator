import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.distributions import Categorical
import torchtext.transforms as T

from model import LSTM, TokenDrop, load_tokenizer_and_vocab, EMB_SIZE, HIDDEN_SIZE, NUM_LAYERS, MAX_LEN, TOKENIZER_MODEL_PATH

learning_rate = 1e-4
nepochs = 20
batch_size = 32
dataset_root = "./datasets"

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


def main():
    dataset_train = AGNews("train")
    dataset_test = AGNews("test")


    train_loader = DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    test_loader = DataLoader(dataset_test,batch_size=batch_size,shuffle=True,num_workers=0)

    _,vocab = load_tokenizer_and_vocab()




    train_transform = T.Sequential(
        T.SentencePieceTokenizer("my_tokenizer.model"),
        T.VocabTransform(vocab=vocab),
        T.AddToken(1,begin=True),
        T.Truncate(max_seq_len=MAX_LEN),
        T.AddToken(2,begin=False),
        T.ToTensor(padding_value=0)
    )
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    
    lstm_generator = LSTM(num_emb=len(vocab),num_layers=NUM_LAYERS,emb_size=EMB_SIZE,hidden_size=HIDDEN_SIZE).to(device)


    optimizer = optim.Adam(lstm_generator.parameters(),lr=learning_rate,weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    t = TokenDrop(prob=0.1)

    training_loss_logger = []

    entropy_logger = []
    
    
    
    for epoch in trange(0, nepochs, leave=False, desc="Epoch"):
        lstm_generator.train()


        # Iterate over batches in training data loader
        for text in tqdm(train_loader, desc="Training", leave=False):

            text_tokens = train_transform(list(text)).to(device)
            bs = text_tokens.shape[0]

            # Randomly drop input tokens
            input_text = t(text_tokens[:, 0:-1])
            output_text = text_tokens[:, 1:]

            hidden,memory = lstm_generator.init_hidden(bs,device)

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
        torch.save({
        "model_state_dict": lstm_generator.state_dict(),
        "config": {"emb_size": EMB_SIZE, "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS, "num_emb": len(vocab)},
        }, "checkpoint.pt")


        lstm_generator.eval()
        val_loss = 0
        with torch.no_grad():
            for val_text in test_loader:
                val_text_tokens = train_transform(list(val_text)).to(device)
                val_bs = val_text_tokens.shape[0]

                val_input_text = val_text_tokens[:, 0:-1]
                val_output_text = val_text_tokens[:, 1:]

                val_hidden,val_memory = lstm_generator.init_hidden(val_bs,device)
                

                val_pred, val_hidden, val_memory = lstm_generator(val_input_text, val_hidden, val_memory)
                val_loss += loss_fn(val_pred.transpose(1, 2), val_output_text).item()

        print(f"Epoch {epoch + 1}/{nepochs} - val loss: {val_loss / len(test_loader):.4f}")
        lstm_generator.train()
        
        
if __name__ == "__main__":
    main()
