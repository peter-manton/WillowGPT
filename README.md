# WillowGPT - Fine-tuned GPT-2 on "The Wind in the Willows"

This project trains and fine-tunes a GPT-2 language model on **"The Wind in the Willows"** dataset, allowing it to generate text based on the book’s style. The script ensures that training only occurs if necessary, saving time and computational resources.

## 🚀 Features
- **Automatic model loading**: If a trained model exists, it is loaded instead of retraining.
- **Fine-tuned GPT-2**: Uses the GPT-2 model from Hugging Face Transformers.
- **Efficient Training**: Optimized for low VRAM usage (GTX 1050 and similar GPUs).
- **Early Stopping**: Stops training if no improvement is seen in 3 evaluation steps.
- **Text Generation**: Allows users to generate text from the trained model.

---

## 📦 Installation

### 1️⃣ Install Dependencies
Make sure you have Python installed, then install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets transformers
```

If you're running without CUDA, install PyTorch without GPU support:

```bash
pip install torch torchvision torchaudio
```

---

## 🏗️ How It Works

1. The script checks if a previously trained model exists in `./wind_in_the_willows_model/`
2. If found, it loads the model and tokenizer.
3. If not found:
   - Loads the dataset (`wind_in_the_willows.txt`)
   - Tokenizes and processes the dataset.
   - Trains the GPT-2 model.
   - Saves the trained model for future use.
4. Once trained, it can generate text in the style of the dataset.

---

## 🏃 Running the Script

Run the script with:

```bash
python main.py
```

If a trained model already exists, it will **skip training** and go straight to text generation.

---

## 🔥 Training Parameters

- **Epochs:** 3
- **Batch Size:** 1 (with gradient accumulation for efficiency)
- **Precision:** Mixed Precision (fp16)
- **Early Stopping:** Stops after 3 epochs without improvement
- **Optimizer:** AdamW with fused optimizations

---

## ✍️ Generating Text

Once the model is trained, you can generate text with:

```python
from main import generate_text
print(generate_text("Once upon a time, there was a riverbank..."))
```

It will return a passage in the style of *The Wind in the Willows*.

---

## 🗂 File Structure
```
│── wind_in_the_willows.txt  # Training dataset
│── wind_in_the_willows_model/  # Saved trained model
│── results/  # Model checkpoints
│── logs/  # Training logs
│── main.py  # Main script
│── README.md  # This file
```

---

## 📌 Notes

- The dataset file `wind_in_the_willows.txt` must be present in the working directory.
- If training is taking too long, try reducing `num_train_epochs` in `main.py`.
- For low-memory GPUs, you may adjust `max_length`, `batch_size`, and `gradient_accumulation_steps`.
