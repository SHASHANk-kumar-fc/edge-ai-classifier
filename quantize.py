import torch
import torch.nn as nn
from train import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "cnn_quant.pth")

print("PyTorch quantization complete")
