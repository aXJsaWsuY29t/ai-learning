# Project 07 — Mini Transformer

## Goal

Assemble a complete, working transformer from all the components built in previous projects. Train it on a small dataset and compare it to the char-level model from Project 06.

This is the final project before returning to hermit-ai.

---

## Why This Matters

A transformer is just: embeddings + positional encoding + stacked attention blocks + feed-forward layers + residual connections. You've already built most of these. This project puts them together.

---

## Concepts

- **Positional encoding** — injecting position information into embeddings
- **Transformer block** — attention + feed-forward + residual + layer norm
- **Residual connections** — `output = x + sublayer(x)` — stabilize training
- **Layer normalization** — normalize activations within a layer
- **Stacking blocks** — deeper = more capacity, but harder to train
- **PyTorch (optional)** — compare your implementation to `nn.TransformerEncoderLayer`

---

## Tasks

### 1. Positional Encoding
- Implement sinusoidal positional encoding (as in "Attention Is All You Need")
- Add to embeddings: `x = embed(tokens) + pos_encoding`
- Visualize: plot the encoding matrix as a heatmap

### 2. Feed-Forward Block
- Implement `FeedForward(embed_dim, ff_dim)`:
  - Two linear layers with ReLU in between
  - `x → Linear(embed_dim, ff_dim) → ReLU → Linear(ff_dim, embed_dim)`

### 3. Layer Normalization
- Implement `LayerNorm(embed_dim)` from scratch
- Normalize across the embedding dimension
- Include learnable scale (`gamma`) and shift (`beta`) parameters

### 4. Transformer Block
- Combine: `CausalSelfAttention + LayerNorm + FeedForward + LayerNorm`
- Use residual connections: `x = x + attention(norm(x))`
- This is one transformer block (one "layer")

### 5. Full Mini Transformer
- Stack 2 transformer blocks
- Add final linear projection to vocabulary size
- Full architecture:
  ```
  tokens → Embedding + PositionalEncoding
         → TransformerBlock × 2
         → LayerNorm
         → Linear(embed_dim, vocab_size)
         → softmax
  ```

### 6. Train on Char-Level Data
- Reuse data pipeline from Project 06
- Train mini transformer on same dataset
- Compare loss curve to Project 06 attention model
- Generate text and compare quality

### 7. PyTorch Comparison (optional)
- Reimplement the same architecture using `torch.nn`
- Compare: does your implementation match PyTorch's output?
- This validates your implementation and introduces you to the framework

### 8. Notebook Summary
- Training curves for all three models (baseline, attention-only, full transformer)
- Generated text samples from each
- Parameter count comparison
- Reflection: what did each component add?

---

## Correctness Goals

- Positional encoding shape: `(seq_len, embed_dim)`
- LayerNorm output has mean ≈ 0 and std ≈ 1 per sample
- Residual connections: output shape matches input shape
- Full transformer trains and loss decreases
- Generated text quality is noticeably better than Project 06 baseline

---

## Format

`.py` for all components, Jupyter notebook for experiments.

```
07-mini-transformer/
├── README.md
├── positional_encoding.py     ← sinusoidal encoding
├── feedforward.py             ← FF block
├── layer_norm.py              ← LayerNorm from scratch
├── transformer_block.py       ← full transformer block
├── transformer.py             ← full model
├── train.py                   ← training loop
├── generate.py                ← text generation
└── experiments.ipynb          ← comparison, generated text, analysis
```

---

## After This Project

You now have all the building blocks to understand and extend hermit-ai. Return to the project with:

- A clear mental model of `CausalSelfAttention`
- Understanding of how training works end-to-end
- Experience debugging and tuning small models
- A foundation for reading papers and framework source code
