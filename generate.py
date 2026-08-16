import torch
from torch.distributions import Categorical
import torchtext.transforms as T

from model import LSTM, load_tokenizer_and_vocab, TOKENIZER_MODEL_PATH

CHECKPOINT_PATH = "checkpoint.pt"


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


if __name__ == "__main__":
    model, vocab, gen_transform, device = load_model()
    print(generate("wall street stocks rise:", model, vocab, gen_transform, device))

