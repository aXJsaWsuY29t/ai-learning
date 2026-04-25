# AI Learning Path — From Linear Algebra to Mini Transformer

A structured, bottom-up learning path to understand how large language models work by building everything from scratch.

The goal is to reach a solid understanding of `CausalSelfAttention` and transformer architecture before returning to the [hermit-ai](../hermit-ai) project.

---

## Projects Overview

| # | Project | Key Concepts | Format |
|---|---------|-------------|--------|
| 01 | [Linear Algebra & Softmax](./01-linear-algebra-softmax/) | Matrix multiplication, dot product, softmax | `.py` |
| 02 | [Neural Network & Backprop](./02-neural-network-backprop/) | MLP, forward pass, manual backprop, gradient descent | `.py` |
| 03 | [Embeddings & Tokenization](./03-embeddings-tokenization/) | Tokenization, embedding matrix, lookup table | `.py` + notebook |
| 04 | [Attention From Scratch](./04-attention-from-scratch/) | Q/K/V, scaled dot-product attention, causal mask | `.py` |
| 05 | [Attention Visualization](./05-attention-visualization/) | Attention weights, heatmaps, interpretability | Notebook |
| 06 | [Char-Level Language Model](./06-char-level-lm/) | Char tokenization, next-char prediction, training loop | `.py` + notebook |
| 07 | [Mini Transformer](./07-mini-transformer/) | Full transformer block, positional encoding, residuals | `.py` + notebook |

---

## Learning Principles

- **Implement everything from scratch** — no PyTorch or TensorFlow until project 07
- **Understand before using** — each project builds directly on the previous one
- **Test your understanding** — each project has clear correctness goals
- **Each project is independent** — can be cloned and run separately on GitHub

---

## Prerequisites

- Python 3.10+
- `numpy` (projects 01–06)
- `matplotlib` (projects 05–07, for visualization)
- `jupyter` (projects with notebooks)
- `torch` (project 07 only, optional comparison)

---

## Recommended Order

Work through projects **01 → 07 in sequence**. Do not skip backprop (02) — it is the hardest and most important step. Project 06 is the key milestone: if your char-level model learns to predict text, you understand 80% of what a transformer does.
