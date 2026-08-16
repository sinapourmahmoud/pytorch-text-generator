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




learning_rate = 1e-4

nepochs = 20

batch_size = 32

max_len = 64

dataset_root = "./datasets"


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




def main():
        
 
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







    gen_transform = T.Sequential(
        T.SentencePieceTokenizer("my_tokenizer.model"),
        T.VocabTransform(vocab=vocab),
        T.AddToken(1,begin=True),
        T.ToTensor(padding_value=0)
    )



   





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
        torch.save({
        "model_state_dict": lstm_generator.state_dict(),
        "config": {"emb_size": emb_size, "hidden_size": hidden_size,
                    "num_layers": num_layers, "num_emb": len(vocab)},
        }, "checkpoint.pt")

        # Validation loss: same loss calculation as training, but on
        # dataset_test (articles the model never learns from), with no
        # TokenDrop noise and no backward()/optimizer.step().
        lstm_generator.eval()
        val_loss = 0
        with torch.no_grad():
            for val_text in test_loader:
                val_text_tokens = train_transform(list(val_text)).to(device)
                val_bs = val_text_tokens.shape[0]

                val_input_text = val_text_tokens[:, 0:-1]
                val_output_text = val_text_tokens[:, 1:]

                val_hidden = torch.zeros(num_layers, val_bs, hidden_size, device=device)
                val_memory = torch.zeros(num_layers, val_bs, hidden_size, device=device)

                val_pred, val_hidden, val_memory = lstm_generator(val_input_text, val_hidden, val_memory)
                val_loss += loss_fn(val_pred.transpose(1, 2), val_output_text).item()

        print(f"Epoch {epoch + 1}/{nepochs} - val loss: {val_loss / len(test_loader):.4f}")
        lstm_generator.train()

    index = 0

    temp = 0.9

    init_prompt = [text[index].split(":")[0] + ":"]

    input_tokens = gen_transform(init_prompt).to(device)
    
    
    # test mode 
    log_tokens = []
    
    lstm_generator.eval()
    
    with torch.no_grad():
        hidden = torch.zeros(num_layers,1,hidden_size,device=device)
        memory = torch.zeros(num_layers,1,hidden_size,device=device)
        
        for i in range(100):

            data_pred, hidden, memory = lstm_generator(input_tokens, hidden, memory)
            
            dist = Categorical(logits=data_pred[:, -1] / temp)
            input_tokens = dist.sample().reshape(1, 1)
            
            log_tokens.append(input_tokens.cpu())
            
            if input_tokens.item() == 2:
                break
    
    generated_ids = torch.cat(log_tokens,dim=1).flatten().tolist()
    
    tokens = vocab.lookup_tokens(generated_ids)
    sentence = "".join(tokens).replace("▁", " ").strip()
    print(sentence)
    


if __name__ == "__main__":
    main()