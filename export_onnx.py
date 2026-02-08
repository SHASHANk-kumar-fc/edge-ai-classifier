import torch
import torch.nn as nn
from train import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
model.eval()

# Apply quantization here (not via state_dict)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

dummy = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    quantized_model,
    dummy,
    "model_quant.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12
)

print("Quantized ONNX export complete")
