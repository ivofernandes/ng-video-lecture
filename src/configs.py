# hyperparameters
import tiktoken
import torch

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'mps'
eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
tokenizer = 'char' # 'tiktoken' for openai bpe tokenizor or 'char' for character tokenizor
file = 'gpt.pth'
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if tokenizer == 'tiktoken':
    vocab_size = 50257 #from https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    bpt_tiktokenizer = tiktoken.get_encoding('gpt2')
    encode = lambda s: [int(bpt_tiktokenizer.encode(c)[0]) for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([bpt_tiktokenizer.decode([i]) for i in l]) # decoder: take a list of integers, output a string
    print('vocab size: ' + str(vocab_size))

elif tokenizer == 'char':
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    print('vocab size: ' + str(vocab_size))