# 🚀 Train, Upload, and Use GPT-2 Model
This project trains a GPT-2 language model on a custom dataset, uploads it to **Hugging Face Hub**, and allows generating text using the uploaded model.

---

## 📌 Prerequisites
1. Install Python (>= 3.8 recommended)
2. Install dependencies:
   ```sh
   pip install torch transformers datasets huggingface_hub
   ```

---

## 🔥 Step 1: Train the Model (`main.py`)
This script **loads a dataset**, **trains a GPT-2 model**, and **saves it locally**.

### **Run Training**
```sh
python main.py
```

✅ **Trained model will be saved at:** `./wind_in_the_willows_model/`

---

## 🚀 Step 2: Upload Model to Hugging Face (`upload_to_huggingface.py`)
After training, upload the model so others can use it.

### **1️⃣ Install Hugging Face CLI**
```sh
pip install huggingface_hub
huggingface-cli login
```
➡️ Log in using a **Hugging Face Access Token** (Get it from: [Hugging Face Tokens](https://huggingface.co/settings/tokens)).

### **2️⃣ Upload Model to Hugging Face**
Run the following command:
```sh
python upload_to_huggingface.py --username "your-username" \
    --model_name "wind-in-the-willows-gpt2" \
    --local_path "./wind_in_the_willows_model" \
    --hf_token "your-huggingface-token"
```
**Replace:**
- `"your-username"` → Your Hugging Face username
- `"wind-in-the-willows-gpt2"` → The repository name
- `"./wind_in_the_willows_model"` → The local trained model path
- `"your-huggingface-token"` → (Optional) Hugging Face API token

✅ **Your model will be available at:**
🔗 `https://huggingface.co/your-username/wind-in-the-willows-gpt2`

---

## 🌎 Step 3: Use the Model from Hugging Face (`use_huggingface_model.py`)
Once uploaded, you (or others) can use the model to generate text.

### **Run the script to generate text:**
```sh
python use_huggingface_model.py --model_name "your-username/wind-in-the-willows-gpt2" \
    --prompt "Once upon a time in the willows," \
    --max_length 150
```
**Replace:**
- `"your-username/wind-in-the-willows-gpt2"` → The Hugging Face model name.
- `"Once upon a time in the willows,"` → The starting text prompt.
- `150` → Maximum length of generated text.

### **Or, Use in Python Code**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model from Hugging Face Hub
model_name = "your-username/wind-in-the-willows-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Generate text
input_text = "Once upon a time in the willows,"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```