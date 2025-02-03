# Download the en_core_web_md language model for spacy: python -m spacy download en_core_web_md

import spacy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy's English model with pre-trained word vectors
nlp = spacy.load("en_core_web_md")

# Define a real sentence
sentence = "The quick brown fox jumps over the lazy dog."
words = sentence.split()

# Get word vectors from spaCy
word_vectors = [nlp(word).vector for word in words]  # List of vectors
input_embeddings = torch.tensor(word_vectors)  # Convert to tensor

# Initialize Weight Matrices for Query, Key, and Value
embedding_dim = input_embeddings.shape[1]  # Get vector dimension (usually 300)
W_q = torch.rand(embedding_dim, embedding_dim)
W_k = torch.rand(embedding_dim, embedding_dim)
W_v = torch.rand(embedding_dim, embedding_dim)

# Compute Q, K, V matrices
Q = input_embeddings @ W_q
K = input_embeddings @ W_k
V = input_embeddings @ W_v

# Compute Attention Scores (QK^T) and Scale
d_k = K.shape[1]  # Dimension of keys
attention_scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# Apply Softmax to Normalize Scores
attention_weights = F.softmax(attention_scores, dim=1)

# Compute Weighted Sum of Values
output = attention_weights @ V  # Final self-attended output

# Convert to NumPy for visualization
attention_weights_np = attention_weights.detach().numpy()

# Plot Attention Weights Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights_np, annot=True, cmap="Blues", xticklabels=words, yticklabels=words)
plt.title("Self-Attention Weights (Word-Level)")
plt.xlabel("Attending To")
plt.ylabel("Attention From")
plt.show()