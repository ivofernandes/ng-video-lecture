import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from src.configs import *
from src.model_gpt import GPTLanguageModel

torch.manual_seed(1337)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# generate from the model
def run_model(m, withMore=False):
    tokens = 200
    if withMore:
        tokens = 600
    print(f"{datetime.datetime.now()} Start generating {tokens} tokens on iteration {iter}")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = m.generate(context, max_new_tokens=tokens)[0].tolist()

    generatedText = decode(generated)
    print(f"{datetime.datetime.now()} Generated {generatedText}")

    file_name = file.replace('.pth', f'_{iter}.pth')
    torch.save(model.state_dict(), file_name)
    print(f"{datetime.datetime.now()} Saved model to {file_name}")
    if withMore:
        open('../../more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


iter = 0
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
run_model(m)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        run_model(m, withMore=False)

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), file)

# generate from the model
run_model(m, withMore=True)
