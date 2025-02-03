# Self-Attention in NLP: Explanation & Code Guide

## Overview
This project demonstrates **self-attention** in Natural Language Processing (NLP) using a simple Python script (`self_attention_words.py`). The script implements **word-level self-attention** to show how words in a sentence attend to each other using **pre-trained word embeddings** and a basic attention mechanism.

## What is Self-Attention?
Self-attention is a mechanism that allows models to focus on **important words** in a sequence when making predictions. Instead of processing words in a fixed order (like RNNs), self-attention computes relationships between all words in the input **simultaneously**.

### 🔹 How Self-Attention Works
1. **Convert words into embeddings** (vector representations of words).
2. **Compute Query (Q), Key (K), and Value (V) matrices** using learned weights.
3. **Calculate attention scores** by taking the dot product of Q and K.
4. **Apply softmax** to normalize the attention scores.
5. **Multiply the attention scores by V** to get the final word representation.

## Running the Script
### Prerequisites
Ensure you have the necessary dependencies installed:
```bash
pip install spacy torch numpy seaborn matplotlib
python -m spacy download en_core_web_md
```

### Execution
Run the script:
```bash
python self_attention_words.py
```
It will generate a **heatmap** visualizing the self-attention weights of words in the input sentence.

## Output Interpretation
- **Darker colors** in the heatmap indicate **stronger attention** between words.
- Words like **"fox"** and **"jumps"** will likely have **higher attention** since they are semantically linked.
- Common words like **"the"** may have lower attention weights, as they provide less semantic information.

## Pros & Cons of Self-Attention
### ✅ **Pros**
✔ **Captures Long-Range Dependencies**: Unlike RNNs, which process sequentially, self-attention allows words **far apart** to interact directly.
✔ **Efficient Parallelization**: Since words are processed **simultaneously**, self-attention can be computed much faster on modern GPUs.
✔ **Handles Complex Relationships**: It learns context better than simpler methods like bag-of-words or n-grams.

### ❌ **Cons**
✖ **Computationally Expensive**: The self-attention mechanism scales **quadratically** with input length, making it inefficient for very long documents.
✖ **Positional Information Loss**: Unlike RNNs or CNNs, self-attention does not **inherently** capture word order, requiring **positional encodings**.

## Next Steps
- Extend the script to **handle full sentences or paragraphs**.
- Implement **multi-head attention** for a more realistic transformer-like model.
- Experiment with **different embeddings (e.g., BERT, Word2Vec)** for improved results.

## Conclusion
Self-attention is a fundamental part of modern NLP models, powering architectures like **Transformers (BERT, GPT-4, T5, etc.)**. By understanding how words attend to each other, we can build models that better understand text for tasks like **summarization, translation, and question answering**.
