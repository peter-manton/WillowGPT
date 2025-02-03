# 🚀 Post-Training Quantization for Hugging Face Models

This repository contains a **Post-Training Quantization (PTQ) script** for Hugging Face Transformer models. The script converts large models to **FP16 (Half Precision)** format for **GPU acceleration**, improving inference speed while reducing model size.

---

## 📌 Features
- ✅ **Supports GPU Acceleration (CUDA)**
- ✅ **FP16 Quantization (Half Precision) for Faster Inference**
- ✅ **Reduces Model Size and Improves Speed**
- ✅ **CLI Support - Specify Model Path Dynamically**
- ✅ **Compatible with Hugging Face Transformers**

---

## ⚙️ Installation
Ensure you have the necessary dependencies installed:
```bash
pip install torch torchvision transformers
```

---

## 📜 Usage
Run the quantization script by providing the **model directory path**:
```bash
python post_training_quantization.py /path/to/your/model/directory
```

### **Example Output**
```bash
Loading model from: orig_wind_in_the_willows_model
Original Model Size: 474.75 MB
Quantized Model Size: 237.37 MB
Model Size Reduction: 50.00%
Inference Time (Original Model): 0.0376 sec
Inference Time (Quantized Model): 0.0192 sec
Inference Speed-up: 48.94%
```

---

## 📊 Quantization Overview

### **What is Quantization?**
Quantization reduces the **precision** of model weights from high-precision floating-point numbers (**FP32**) to lower-precision formats (**FP16, INT8, or INT4**), which decreases memory usage and speeds up inference.

There are **two primary types of quantization**:
1. **Post-Training Quantization (PTQ)** 🛠️
2. **Quantization-Aware Training (QAT) - Pre-Training Quantization** 🎯

---

## 🔄 Post-Training Quantization (PTQ)
### ✅ **Pros**
- **No Retraining Required** – Faster and easier to apply.
- **Reduces Inference Latency** – Models run significantly faster on GPUs and CPUs.
- **Lower Memory Usage** – Reduces VRAM and RAM consumption.
- **Improves Deployment Efficiency** – Makes large LLMs deployable on edge devices.

### ❌ **Cons**
- **Slight Accuracy Drop** – Small performance degradation (typically 1-2%).
- **Limited to Weight Compression** – Does not optimize activations like QAT.
- **May Not Work for All Layers** – Some architectures (e.g., `nn.Embedding`) require special handling.

---

## 🎯 Quantization-Aware Training (QAT) – Pre-Training Quantization
### ✅ **Pros**
- **Minimizes Accuracy Loss** – Model is trained while considering quantization.
- **Optimized for INT8 Deployments** – Results in highly optimized INT8 models.
- **Best for Edge Devices** – Useful for mobile inference (TensorRT, ONNX, TFLite).

### ❌ **Cons**
- **Requires Retraining** – Needs access to dataset & compute resources.
- **Takes Longer** – Must retrain the model for multiple epochs.
- **More Complex** – Requires model modifications during training.

---

## ⚖️ **PTQ vs. QAT: When to Use What?**
| Use Case | Best Quantization Method |
|----------|--------------------------|
| Faster inference on GPUs | **PTQ (FP16/INT8)** |
| Smaller model size for deployment | **PTQ (INT8)** |
| Maximum accuracy with quantization | **QAT** |
| Running models on edge devices (mobile) | **QAT (TensorRT, TFLite)** |
| No access to training data | **PTQ** |