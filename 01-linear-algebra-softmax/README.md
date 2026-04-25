# Project 01 — Linear Algebra & Softmax

## Goal

Build the mathematical foundation needed to understand attention. By the end of this project you will have implemented the core operations that power every transformer from scratch — using only Python and NumPy.

---

## Why This Matters

Attention is just: compute similarity between vectors → normalize with softmax → use as weights. If matrix multiplication and softmax feel like black boxes, attention will too.

---

## Concepts

- **Matrix multiplication** — the engine of every neural network layer
- **Dot product** — measures similarity between two vectors
- **Softmax** — converts a vector of scores into a probability distribution
- **Weighted sum** — combining vectors using attention weights

---

## Tasks

### 1. Matrix Multiplication
- Implement `matmul(A, B)` without using `np.dot` or `@`
- Verify against NumPy's result
- Understand: what does each output cell represent?

### 2. Dot Product & Cosine Similarity
- Implement `dot_product(a, b)`
- Implement `cosine_similarity(a, b)`
- Demo: given 3 word vectors, find which two are most similar

### 3. Softmax
- Implement `softmax(x)` — handle numerical stability (subtract max)
- Verify: output sums to 1, all values positive
- Plot: show how softmax "sharpens" or "flattens" depending on temperature

### 4. Attention Core Demo
- Create 4 random vectors (simulate word embeddings)
- Compute pairwise dot products → similarity matrix
- Apply softmax to each row
- Compute weighted sum of vectors using softmax weights
- Print and inspect the result

---

## Correctness Goals

- `matmul(A, B)` matches `np.dot(A, B)` for any valid shapes
- `softmax(x)` output always sums to 1.0 (within float tolerance)
- `softmax([1, 2, 3])` gives higher weight to index 2
- Weighted sum output has same shape as input vectors

---

## Format

Plain `.py` files. No notebooks needed here — focus is on clean, testable implementations.

```
01-linear-algebra-softmax/
├── README.md
├── matmul.py
├── softmax.py
└── attention_demo.py
```

---

## Next Step

Once you can explain what each line of `attention_demo.py` does mathematically, move to **Project 02**.
