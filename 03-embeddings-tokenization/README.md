# Project 03 — Embeddings & Tokenization

## Goal

Understand how raw text becomes vectors that a neural network can process. Implement a character-level tokenizer and a trainable embedding matrix from scratch.

---

## Why This Matters

LLMs don't see words — they see vectors. The embedding layer is the bridge between discrete tokens and continuous vector space. Understanding this is essential before attention makes sense.

---

## Concepts

- **Tokenization** — splitting text into discrete units (characters, subwords, words)
- **Vocabulary** — the set of all unique tokens
- **Token ID** — integer index for each token
- **Embedding matrix** — a lookup table mapping token IDs to vectors
- **Learned embeddings** — embeddings that improve through training

---

## Tasks

### 1. Character-Level Tokenizer
- Build `encode(text) → list[int]` and `decode(ids) → str`
- Build vocabulary from a text corpus (e.g., a short story or poem)
- Verify round-trip: `decode(encode(text)) == text`

### 2. Embedding Matrix
- Implement `EmbeddingLayer(vocab_size, embed_dim)`
- Forward: given a list of token IDs, return their embedding vectors (matrix lookup)
- Initialize with random values
- Verify shape: `embed([1, 3, 5])` returns shape `(3, embed_dim)`

### 3. Train Embeddings (Next-Char Prediction)
- Build a simple dataset: for each character, predict the next one
- Use a 1-layer network: `embedding → linear → softmax`
- Train with cross-entropy loss
- After training, inspect embeddings — similar characters should cluster

### 4. Visualize Embeddings (Notebook)
- Use PCA or t-SNE to reduce embeddings to 2D
- Plot and label each character
- Observe: do vowels cluster together? Do similar chars end up close?

---

## Correctness Goals

- `decode(encode(text)) == text` for any input string
- Embedding lookup returns correct shape `(seq_len, embed_dim)`
- Training loss decreases over epochs
- Embedding visualization shows some meaningful structure

---

## Format

`.py` for tokenizer and embedding implementation, Jupyter notebook for visualization.

```
03-embeddings-tokenization/
├── README.md
├── tokenizer.py          ← encode/decode, vocabulary
├── embedding.py          ← EmbeddingLayer from scratch
├── train.py              ← next-char prediction training
└── visualize.ipynb       ← PCA/t-SNE embedding plot
```

---

## Next Step

Once you can turn text into a sequence of vectors and train embeddings, move to **Project 04**.
