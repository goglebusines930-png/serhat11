import torch
import os

# Dosyadan veri oku
def load_text_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Hata: {file_path} dosyası bulunamadı!")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Tokenization
chars = None
stoi = None
itos = None
vocab_size = None
train_data = None
val_data = None

def prepare_data(text, train_split=0.9):
    global chars, stoi, itos, vocab_size, train_data, val_data
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Kelime dağarcığı boyutu: {vocab_size}")
    print(f"Metin uzunluğu: {len(text)} karakter")

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def get_batch(batch_size=32, block_size=64):
    if train_data is None:
        raise ValueError("Veri hazırlanmadı! prepare_data() çağır.")
    
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

# Veri hazırlama fonksiyonu
def setup_data(file_path):
    global train_data, val_data
    
    text = load_text_from_file(file_path)
    if text is None:
        return False
    
    prepare_data(text)
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return True
