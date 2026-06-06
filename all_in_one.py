import torch
import torch.nn as nn
import math
import os

# ============ MODEL ============
class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(256, 256)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ============ VERİ ============
chars = None
stoi = None
itos = None
vocab_size = None
train_data = None
val_data = None

def load_text_from_file(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def prepare_data(text):
    global chars, stoi, itos, vocab_size
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def get_batch(batch_size=32, block_size=64):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

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

# ============ MAIN ============
print("\n" + "="*60)
print("🤖 TÜRKÇE YAPAY ZEKA MODELİ - GÜVENLIKLI VERSİYON")
print("="*60)

data_file = "data.txt"
if not os.path.exists(data_file):
    print(f"\n⚠️ {data_file} dosyası bulunamadı!")
    print("Örnek dosya oluşturuluyor...\n")
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write("Merhaba bu bir yapay zeka modelidir. Yapay zeka çok önemlidir. Teknoloji gelecektir. ")
        f.write("Python programlama dilidir. Türkiye güzel bir ülkedir. ")
    print(f"✓ {data_file} oluşturuldu. İçine daha fazla Türkçe metin ekle!\n")

if not setup_data(data_file):
    print(f"Hata: Veri yüklenemedi!")
    exit(1)

print(f"✓ Kelime dağarcığı: {vocab_size} karakter")
print(f"✓ Metin uzunluğu: {len(train_data) + len(val_data)} token\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Cihaz: {device}\n")

model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("="*60)
print("EĞITIM BAŞLANIYOR...")
print("="*60)

for iter in range(5000):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 500 == 0:
        print(f"Adım {iter:4d} | Kayıp: {loss.item():.4f}")

model_path = "model_trained.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✓ Model kaydedildi: {model_path}\n")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("="*60)
print("MODELİ ÜRETİM MODU")
print("="*60)

admin_mode = False
admin_password = "anahtar587801"

while True:
    try:
        user_input = input("\n> ").strip()
        
        if not user_input:
            continue
        
        # Admin kontrolü
        if user_input == admin_password:
            admin_mode = True
            print("✓ Admin modu açıldı! Komutları kullanabilirsin.")
            continue
        
        if admin_mode:
            # Admin komutları
            if user_input.startswith("egit"):
                print("\n🔄 Model yeniden eğitiliyor...")
                for iter in range(1000):
                    xb, yb = get_batch()
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if iter % 200 == 0:
                        print(f"  Adım {iter} | Kayıp: {loss.item():.4f}")
                torch.save(model.state_dict(), model_path)
                print("✓ Eğitim tamamlandı ve model kaydedildi!\n")
            
            elif user_input.startswith("veri_ekle"):
                yeni_veri = input("  Eklenecek metni gir: ").strip()
                if yeni_veri:
                    with open(data_file, 'a', encoding='utf-8') as f:
                        f.write(" " + yeni_veri)
                    print("✓ Veri dosyasına eklendi!\n")
                    setup_data(data_file)
                    print("✓ Veri yeniden yüklendi!\n")
            
            elif user_input.startswith("parametreler"):
                print("\n📊 Model Parametreleri:")
                print(f"  - Kelime dağarcığı: {vocab_size}")
                print(f"  - Embedding boyutu: 64")
                print(f"  - Attention başları: 4")
                print(f"  - Katman sayısı: 4")
                print(f"  - Blok boyutu: 256")
                print(f"  - Eğitim verisi: {len(train_data)}")
                print(f"  - Validation verisi: {len(val_data)}\n")
            
            elif user_input.startswith("save"):
                torch.save(model.state_dict(), model_path)
                print("✓ Model kaydedildi!\n")
            
            elif user_input.startswith("cikis_admin"):
                admin_mode = False
                print("✓ Admin modundan çıkıldı.\n")
            
            elif user_input.startswith("yardim"):
                print("\n📖 Admin Komutları:")
                print("  - egit: Modeli 1000 adım eğit")
                print("  - veri_ekle: Yeni veri ekle")
                print("  - parametreler: Model parametrelerini göster")
                print("  - save: Modeli kaydet")
                print("  - cikis_admin: Admin modundan çık")
                print("  - yardim: Bu mesajı göster\n")
            
            else:
                print("❓ Komut tanınmadı. 'yardim' yazarak komutları görebilirsin.\n")
        
        else:
            # Normal mod - metin üretme
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=200)
            generated_text = decode(generated[0].tolist())
            print(f"\n🤖 Model: {generated_text}\n")
    
    except KeyboardInterrupt:
        print("\n\n✓ Program kapatıldı.")
        break
    except Exception as e:
        print(f"❌ Hata: {e}\n")

print("="*60)
print("TAMAMLANDI!")
print("="*60 + "\n")
