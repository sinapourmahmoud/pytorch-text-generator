import torch
from torch.distributions import Categorical
import torchtext.transforms as T

from model import LSTM, load_tokenizer_and_vocab, TOKENIZER_MODEL_PATH

CHECKPOINT_PATH = "checkpoint.pt"
WORD_BOUNDARY = "▁"

def load_model(checkpoint_path = CHECKPOINT_PATH, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = LSTM(num_emb=config["num_emb"], num_layers=config["num_layers"],
                 emb_size=config["emb_size"], hidden_size=config["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, vocab = load_tokenizer_and_vocab()

    gen_transform = T.Sequential(
        T.SentencePieceTokenizer(TOKENIZER_MODEL_PATH),
        T.VocabTransform(vocab=vocab),
        T.AddToken(1, begin=True),
        T.ToTensor(padding_value=0)
    )

    return model, vocab, gen_transform, device



def generate(prompt_text, model, vocab, gen_transform, device, num_tokens=100, temp=0.9):
    input_tokens = gen_transform([prompt_text]).to(device)

    log_tokens = []

    with torch.no_grad():
        hidden, memory = model.init_hidden(1, device)

        for i in range(num_tokens):
            data_pred, hidden, memory = model(input_tokens, hidden, memory)

            dist = Categorical(logits=data_pred[:, -1] / temp)
            input_tokens = dist.sample().reshape(1, 1)

            log_tokens.append(input_tokens.cpu())

            if input_tokens.item() == 2:
                break

    generated_ids = torch.cat(log_tokens, dim=1).flatten().tolist()

    tokens = vocab.lookup_tokens(generated_ids)
    sentence = "".join(tokens).replace("▁", " ").strip()
    return sentence





def _sample_one_word(prompt_tokens, model, vocab, device, temp=0.9, top_k=10, max_pieces=6):
    input_tokens = prompt_tokens
    piece_ids = []


    with torch.no_grad():
        hidden,memory = model.init_hidden(1,device)
       
        for i in range(max_pieces):
            predict,hidden,memory = model(input_tokens,hidden,memory)
            
            dist = Categorical(logits=predict[:,-1]/temp)
            
            next_id = dist.sample().item()
            
            if next_id in (0,2):
                break
            
            piece = vocab.lookup_tokens([next_id])[0]
            
            if piece.startswith(WORD_BOUNDARY) and piece_ids:
                break
            
            piece_ids.append(next_id)
            input_tokens = torch.tensor([[next_id]],device=device)
            
            
    if not piece_ids:
        return "" 
    
    tokens = vocab.lookup_tokens(piece_ids)
    return "".join(tokens).replace(WORD_BOUNDARY, "").strip()
        
def suggest_next_words(prompt_text, model, vocab, gen_transform, device, num_suggestions=5, temp=0.9, top_k=10):
    
    input_text = gen_transform([prompt_text]).to(device)
    suggestions =[]
    seen = set()
    attempt = 0
    max_attempt = num_suggestions * 4
    
    while len(suggestions) < num_suggestions and attempt < max_attempt:
        attempt+=1
        
        predicted = _sample_one_word(input_text,model,vocab,device,temp,top_k)
        
        if predicted not in seen and predicted:
            seen.add(predicted)
            suggestions.append(predicted)
            
    return suggestions

           
     




if __name__ == "__main__":
    model, vocab, gen_transform, device = load_model()
    print(generate("wall street stocks rise:", model, vocab, gen_transform, device))

