import torch
from model import MiniGPT
from data import get_batch, vocab_size, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(5000):   # Daha uzun eğitmek istersen sayıyı artır
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 500 == 0:
        print(f"Adım {iter} | Loss: {loss.item():.4f}")

# Test
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)  # generate fonksiyonu ekleyeceğiz
print(decode(generated[0].tolist()))
