import onnxruntime as ort
import numpy as np
import time
import os

session = ort.InferenceSession("model.onnx")

input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)

runs = 100
start = time.time()

for _ in range(runs):
    session.run(None, {"input": input_data})

end = time.time()

print("Average inference time:", (end - start) / runs)
print("Model size (MB):", os.path.getsize("model.onnx") / (1024 * 1024))
