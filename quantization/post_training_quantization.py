import torch
import argparse
import os
import time
from transformers import AutoModel, AutoTokenizer

def load_model(model_path, device):
    """Loads the Hugging Face model and tokenizer, moving it to the specified device."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

def save_model_size(model, filename):
    """Saves the model and returns its size in MB."""
    torch.save(model.state_dict(), filename)
    return os.path.getsize(filename) / (1024 * 1024)  # Convert bytes to MB

def apply_fp16_quantization(model):
    """Applies FP16 (Half Precision) Quantization for GPU acceleration."""
    model.half()  # Convert model weights to FP16
    return model

def evaluate_model(model, tokenizer, text, device):
    """Measures inference time for a given model using FP16 precision."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision
        start_time = time.time()
        _ = model(**inputs)
        end_time = time.time()

    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="FP16 Quantization for Hugging Face Models (GPU)")
    parser.add_argument("model_path", type=str, help="Path to the directory containing the model files")
    args = parser.parse_args()

    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model_path = args.model_path

    print(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path, device)

    # Save original model and check size
    original_size = save_model_size(model, "original_model.pth")
    print(f"Original Model Size: {original_size:.2f} MB")

    # Apply FP16 Quantization
    quantized_model = apply_fp16_quantization(model)

    # Save quantized model and check size
    quantized_size = save_model_size(quantized_model, "quantized_model.pth")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")

    # Compute model size reduction
    reduction = ((original_size - quantized_size) / original_size) * 100
    print(f"Model Size Reduction: {reduction:.2f}%")

    # Inference speed test
    sample_text = "Hello, how are you?"
    original_time = evaluate_model(model, tokenizer, sample_text, device)
    quantized_time = evaluate_model(quantized_model, tokenizer, sample_text, device)

    print(f"Inference Time (Original Model): {original_time:.4f} sec")
    print(f"Inference Time (Quantized Model): {quantized_time:.4f} sec")

    # Compute speedup
    speedup = ((original_time - quantized_time) / original_time) * 100
    print(f"Inference Speed-up: {speedup:.2f}%")

if __name__ == "__main__":
    main()