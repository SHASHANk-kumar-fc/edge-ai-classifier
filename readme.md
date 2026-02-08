# Edge AI Image Classifier (ONNX Runtime)

This project demonstrates a deep-learning deployment pipeline for edge AI systems using PyTorch and ONNX Runtime.

The goal is to simulate how neural network models are prepared and executed on CPU-based edge devices.

---

## Features

- CNN image classifier trained on CIFAR-10
- ONNX model export
- CPU inference using ONNX Runtime
- Real-time webcam inference demo
- Inference latency benchmarking

---

## Project Structure

edge-ai-classifier/
│
├── train.py
├── export_onnx.py
├── onnx_infer.py
├── webcam_infer.py
├── quantize.py
└── README.md


---

## Installation

Create virtual environment:



python -m venv venv
venv\Scripts\activate


Install dependencies:



pip install torch torchvision onnx onnxruntime opencv-python numpy


---

## Train Model



python train.py


---

## Export Model to ONNX



python export_onnx.py


---

## Run Inference Benchmark



python onnx_infer.py


---

## Run Webcam Demo



python webcam_infer.py


Press **q** to exit.

---

## Tech Stack

- PyTorch
- ONNX Runtime
- OpenCV
- Python

---

## Author

Shashank
