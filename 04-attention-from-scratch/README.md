# Project 04 — Attention From Scratch

## Goal

Implement scaled dot-product attention and causal (masked) self-attention entirely from scratch using only NumPy. This is the core mechanism of every transformer.

---

## Why This Matters

Attention is what makes transformers powerful. `CausalSelfAttention` — the thing that felt confusing in hermit-ai — is just this mechanism with a mask applied. After this project, it will be completely transparent.

---

## Concepts

- **Query, Key, Value (Q, K, V)** — three projections of the input
- **Scaled dot-product attention** — `softmax(QKᵀ / √d_k) · V`
- **Attention scores** — how much each token "attends to" each other token
- **Causal mask** — prevents a token from attending to future tokens
- **Multi-head attention** — running attention in parallel with different projections

---

## Tasks

### 1. Scaled Dot-Product Attention
- Implement `attention(Q, K, V)`:
  - Compute scores: `Q @ K.T / sqrt(d_k)`
  - Apply softmax to scores
  - Return weighted sum: `softmax(scores) @ V`
- Test with small random matrices, verify output shape

### 2. Causal Mask
- Implement `causal_mask(seq_len)` — upper triangular matrix of `-inf`
- Apply mask before softmax: `scores + mask`
- Verify: after softmax, future positions have weight ≈ 0
- Understand why: `softmax(-inf) = 0`

### 3. Self-Attention Layer
- Implement `SelfAttentionLayer(embed_dim)`:
  - Weight matrices `W_Q`, `W_K`, `W_V` (randomly initialized)
  - `forward(X)` → compute Q, K, V from X, then run attention
- Test: input shape `(seq_len, embed_dim)` → output same shape

### 4. Causal Self-Attention
- Add causal mask to `SelfAttentionLayer`
- Verify: token at position `i` only attends to positions `0..i`
- This is exactly `CausalSelfAttention` from GPT

### 5. Multi-Head Attention (optional but recommended)
- Split embedding into `n_heads` heads
- Run attention independently per head
- Concatenate and project back
- Verify output shape matches input shape

---

## Correctness Goals

- Output shape of attention is `(seq_len, d_v)`
- Causal mask: attention weight from position `i` to position `j > i` is exactly 0
- Self-attention output shape matches input shape
- Multi-head output shape matches input shape

---

## Format

Plain `.py` files. This is pure math — keep it clean and well-commented.

```
04-attention-from-scratch/
├── README.md
├── attention.py           ← scaled dot-product attention
├── causal_mask.py         ← mask implementation and tests
├── self_attention.py      ← SelfAttentionLayer class
└── multihead_attention.py ← optional multi-head extension
```

---

## Next Step

Once you can explain every line of `self_attention.py` and the causal mask makes sense, move to **Project 05**.
