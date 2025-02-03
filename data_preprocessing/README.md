# 📝 Text Preprocessing for LLM Training

This repository contains a **text preprocessing script** (`preprocess_data.py`) that prepares raw text data for training **Large Language Models (LLMs)** like GPT-2.

## 🚀 Why is Preprocessing Important?

Preprocessing is **crucial** before training an LLM because:
- **Removes Noise**: Cleans unwanted characters, extra spaces, and formatting issues.
- **Improves Tokenization**: Ensures proper word spacing and punctuation handling.
- **Reduces Training Time**: Filters unnecessary data, making training more efficient.
- **Enhances Model Output**: Provides structured and high-quality input for better text generation.

---

## 🛠️ Features of `preprocess_data.py`
This script:
- ✅ **Removes non-ASCII characters** (e.g., unwanted symbols)
- ✅ **Ensures proper spacing after punctuation** (avoids merging words)
- ✅ **Cleans excessive whitespace and newlines**
- ✅ **Preserves paragraph structure** (no forced splitting)
- ✅ **Prepares text for tokenization & model training**

---

## 📌 Common Text Preprocessing Methods

Before training an LLM, common preprocessing techniques include:

### **1️⃣ Cleaning Unwanted Characters**
- Remove **special symbols**, emojis, and non-printable characters.
- Example:
  **Raw:** `Hello!! 😊 How are you?` → **Cleaned:** `Hello!! How are you?`

### **2️⃣ Fixing Punctuation & Spacing**
- Ensures correct spaces after punctuation marks:
  - **Incorrect:** `"Hello,world!"`
  - **Corrected:** `"Hello, world!"`

### **3️⃣ Lowercasing (Optional)**
- Convert all text to lowercase to **reduce vocabulary size**.
- Example: `"Hello World!" → "hello world!"`

### **4️⃣ Removing Stop Words (Optional)**
- Stop words (`the, is, and, of, etc.`) may be removed in some NLP tasks.

### **5️⃣ Normalizing Quotes & Apostrophes**
- Standardizes curly (`“”`, `‘’`) and straight (`"`, `'`) quotes.

---

## 🔧 Installation & Usage

### **1️⃣ Install Dependencies**
Ensure Python is installed. No external dependencies are needed.

### **2️⃣ Run the Script**
```bash
python preprocess_data.py
```

### **3️⃣ Input & Output**
- **Input:** `wind_in_the_willows.txt` (raw text)
- **Output:** `processed_wind_in_the_willows.txt` (cleaned text)

---

## 📝 Example Before & After

### **📌 Raw Text**
```
   Hello,world!This is a test.
🙂🙂🙂 This is an emoji-filled sentence.
```

### **✅ Preprocessed Text**
```
Hello, world! This is a test.
This is an emoji-filled sentence.
```