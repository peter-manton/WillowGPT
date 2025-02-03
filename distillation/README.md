# 🚀 GPT-2 Model Distillation for "Wind in The Willows" LLM

This project distills a fine-tuned **GPT-2 model** into a smaller, more efficient version using **knowledge distillation**. Distillation reduces the model size while maintaining its performance, making it faster and more suitable for deployment.

## 📖 What is Knowledge Distillation?
**Knowledge Distillation (KD)** is a model compression technique where a large, pretrained model (**teacher model**) transfers its knowledge to a smaller, more efficient model (**student model**).

### 🔬 **How It Works**
1. **Teacher Model** (fine-tuned GPT-2) generates outputs (logits).
2. **Student Model** (smaller GPT-2) is trained to mimic the **soft labels** (probability distributions) of the teacher.
3. The student learns from the teacher’s predictions instead of just raw labels, allowing it to generalize better.

---

## ✅ **Pros & Cons of Distillation**

### ✔️ **Pros**
- 🔥 **Faster inference** → Smaller model = lower latency.
- 💾 **Lower memory usage** → Suitable for edge devices.
- 🎯 **Maintains accuracy** → Retains most of the original model’s knowledge.
- ⚡ **Efficient deployment** → Can run on CPUs and low-power GPUs.

### ❌ **Cons**
- 🚀 **Extra training required** → Training the student model takes additional compute.
- 🎭 **Potential accuracy loss** → Student may not fully match the teacher’s performance.
- 🎛️ **Hyperparameter tuning needed** → Requires careful optimization for best results.

---

## 🛠️ **Setup Instructions**
### **1️⃣ Install Dependencies**
Ensure you have Python and the required libraries installed:

```bash
pip install torch transformers datasets argparse
```

---

### **2️⃣ Run the Distillation Script**
The script requires three inputs:
1. **Path to the fine-tuned model (`--model_dir`)**
2. **Path to the dataset (`--dataset_path`)**
3. **Path to save the distilled model (`--output_dir`)**

Run the script with:
```bash
python model_distillation.py --model_dir ./wind_in_the_willows_model --dataset_path wind_in_the_willows.txt --output_dir ./wind_in_the_willows_distilled
```

---

## ⚙️ **Script Breakdown**
- **Loads a fine-tuned GPT-2 model (teacher)**
- **Loads a smaller GPT-2 model (student)**
- **Tokenizes the dataset**
- **Trains the student to mimic the teacher's output**
- **Saves the distilled model to the specified directory**

---

## 🎯 **Expected Output**
After training, you should see something like:
```
🔍 Loading teacher model from: ./wind_in_the_willows_model
🔍 Loading student model (distilgpt2)...
📚 Loading dataset from: wind_in_the_willows.txt
⚙️ Tokenizing dataset...
🚀 Starting distillation training...
⚠️ Skipping empty input batch...
✅ Epoch 1: Average Loss 0.5672
✅ Epoch 2: Average Loss 0.4238
✅ Epoch 3: Average Loss 0.3210
💾 Saving distilled student model to: ./wind_in_the_willows_distilled
✅ Distilled model saved successfully!
```

---

## 📜 **References**
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531)