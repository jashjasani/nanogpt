import torch
from torch import nn
from torch.nn import functional as F
import sys

n_embd = 4*64
batch_size = 32
block_size = 256
dropout = 0.2
n_head = 4
n_layer = 6
epochs = 5000

device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open("input.txt", "r") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}

encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])





data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train = data[:n]
val = data[n:]





def get_batch(split):
    data = train if split=="train" else val
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0 , float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out 
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 
    

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 
    

class BiGramModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab_size,n_embd)
        self.wpe = torch.nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape 

        token_embed = self.wte(idx)
        pos_embed = self.wpe(torch.arange(T, device=device))
        x = token_embed+pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_tokens):
        
        for _ in range(max_tokens):
            
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            sys.stdout.write(decode(idx_next.tolist()[0]))
            sys.stdout.flush()
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        # return idx
    
model= BiGramModel()
model.load_state_dict(torch.load("./model.pt"))
model.to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)



# for steps in range(epochs):
#     x, y = get_batch("train")
    
#     logits, loss = model(x,y)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#     print("Loss = ", loss.item(), " epoch = ", steps)

# torch.save(model.state_dict(), "./model.pt")

x = torch.zeros((1,1), dtype=torch.long, device=device)
model.generate(x,5000)