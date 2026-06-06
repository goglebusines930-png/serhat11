import torch
from model import MiniGPT
from data import setup_data, get_batch, vocab_size, decode
import os

# Veri hazırla
data_file = "data.txt"
if not setup_data(data_file):
    print(f"Lütfen '{data_file}' dosyasını oluştur ve Türkçe metin ekle!")
    exit(1)

from data import vocab_size as vs
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nCihaz: {device}")

model = MiniGPT(vs).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\n" + "="*50)
print("EĞITIM BAŞLANIYOR...")
print("="*50)
for iter in range(5000):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vs), yb.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 500 == 0:
        print(f"Adım {iter} | Kayıp: {loss.item():.4f}")

# Modeli kaydet
model_path = "model_trained.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✓ Model kaydedildi: {model_path}")

# Test
print("\n" + "="*50)
print("ÖRNEK METİN ÜRETİLİYOR...")
print("="*50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print("\nÜretilen metin:")
print(decode(generated[0].tolist()))
print("="*50)
