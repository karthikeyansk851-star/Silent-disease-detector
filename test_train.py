print("start", flush=True)
import torch
print("torch ok", flush=True)
from sklearn.model_selection import train_test_split
print("sklearn ok", flush=True)
device = torch.device('cuda')
print("device ok", flush=True)
x = torch.randn(5, 3)
y = train_test_split(range(5), test_size=0.2)
print("torch + sklearn together ok", flush=True)