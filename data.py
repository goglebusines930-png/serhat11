import torch

# Basit Türkçe metin verisi (daha büyük veri istersen Shakespeare veya kendi txt dosyanı kullan)
text = """Merhaba bu bir yapay zeka eğitim metnidir. 
Yapay zeka gelecekte çok önemli olacak. 
Biz de kendi modelimizi eğitiyoruz."""

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(batch_size=32, block_size=64):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y
