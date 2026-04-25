# Project 05 — Attention Visualization

## Goal

Build intuition for what attention actually does by visualizing attention weights on real text. See which tokens attend to which — and why.

---

## Why This Matters

Attention weights are interpretable. Visualizing them turns an abstract matrix operation into something you can reason about. This project bridges the gap between "I implemented it" and "I understand it."

---

## Concepts

- **Attention matrix** — `(seq_len × seq_len)` matrix of weights
- **Heatmap** — visual representation of attention weights
- **Causal pattern** — lower-triangular structure in causal attention
- **Attention heads** — different heads learn different patterns

---

## Tasks

### 1. Attention Heatmap
- Take a short sentence: e.g. `"the cat sat on the mat"`
- Tokenize (character or word level)
- Run through your attention layer from Project 04
- Plot the attention matrix as a heatmap using `matplotlib`
- Label axes with token names

### 2. Causal Mask Visualization
- Show the attention matrix before and after applying the causal mask
- Visualize: upper triangle should be zero after masking
- Annotate which positions are "allowed" to attend to which

### 3. Compare Random vs Trained Weights
- Run attention with random `W_Q`, `W_K`, `W_V` — plot heatmap
- Train a tiny model on a few sentences (next-token prediction)
- Run attention again — compare: does the pattern change?

### 4. Multi-Head Comparison (if you did optional task in 04)
- Plot attention heatmap for each head separately
- Observe: do different heads focus on different relationships?

---

## Correctness Goals

- Heatmap rows sum to 1.0 (valid probability distribution)
- Causal mask heatmap shows strict lower-triangular pattern
- Visualization is labeled and readable

---

## Format

Jupyter Notebook — this project is entirely about visual exploration.

```
05-attention-visualization/
├── README.md
└── attention_viz.ipynb   ← all tasks in one notebook
```

---

## Next Step

Once you can look at an attention heatmap and explain what it means, move to **Project 06**.
