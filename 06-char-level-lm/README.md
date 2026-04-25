# Project 06 — Character-Level Language Model

## Goal

Build a language model that learns to predict the next character in a sequence. Train it on real text and watch it generate new text. This is the key milestone of the entire learning path.

This project has two phases:
1. A simple baseline model (no transformer)
2. Replace the core with a single attention block

---

## Why This Matters

A char-level LM is a transformer in miniature. It has the same training objective (predict next token), the same data pipeline, and the same evaluation loop. If this works, you understand 80% of GPT.

---

## Concepts

- **Language modeling objective** — predict the next token given all previous tokens
- **Context window** — how many previous characters the model sees
- **Cross-entropy loss** — standard loss for next-token prediction
- **Temperature sampling** — controlling randomness during text generation
- **Perplexity** — standard metric for language model quality

---

## Tasks

### Phase 1 — Baseline Model (No Transformer)

#### 1. Data Pipeline
- Load a text file (e.g., a short story, Shakespeare excerpt, or any plain text)
- Tokenize at character level (reuse Project 03 tokenizer)
- Build `(input, target)` pairs: input = chars `[i:i+ctx]`, target = chars `[i+1:i+ctx+1]`
- Implement a simple batch sampler

#### 2. Simple Context Model
- Implement a model: `embedding → flatten → linear → softmax`
- Input: sequence of `ctx` token IDs
- Output: probability distribution over vocabulary
- This is a "bag of context" model — no sequence awareness yet

#### 3. Training Loop
- Implement training loop with cross-entropy loss
- Log loss every N steps
- Plot loss curve over training
- Checkpoint: save best model weights

#### 4. Text Generation
- Implement `generate(model, seed_text, n_chars, temperature)`
- Start from a seed string, sample next char, append, repeat
- Try different temperatures: 0.5 (focused), 1.0 (normal), 1.5 (creative/chaotic)
- The model won't be great — but it should produce vaguely text-like output

### Phase 2 — Replace Core with Attention

#### 5. Attention-Based Model
- Replace `flatten → linear` with your `SelfAttentionLayer` from Project 04
- New architecture: `embedding → causal self-attention → linear → softmax`
- Keep everything else the same (data pipeline, training loop, generation)

#### 6. Compare Results
- Train both models on the same data for the same number of steps
- Compare loss curves — attention model should converge better
- Compare generated text quality
- Plot both loss curves on the same chart

#### 7. Notebook Summary
- Combine training, generation, and comparison in a notebook
- Show sample generated text from both models
- Discuss: what did attention improve?

---

## Correctness Goals

- Loss decreases during training for both models
- Generated text is not random noise — it should show learned patterns (e.g., spaces in right places, common letter combinations)
- Attention model achieves lower final loss than baseline on same data
- `generate()` produces different output at different temperatures

---

## Milestone Check

If your attention-based char-level LM generates recognizable text patterns after training, you have genuinely understood the core of how GPT works. The rest is engineering.

---

## Format

`.py` for model and training code, Jupyter notebook for experiments and comparison.

```
06-char-level-lm/
├── README.md
├── data/
│   └── input.txt              ← your training text
├── tokenizer.py               ← reuse or adapt from Project 03
├── dataset.py                 ← data pipeline, batch sampler
├── model_baseline.py          ← simple context model
├── model_attention.py         ← attention-based model
├── train.py                   ← training loop
├── generate.py                ← text generation with temperature
└── experiments.ipynb          ← loss curves, generated text, comparison
```

---

## Next Step

Once your attention-based char-level LM trains and generates text, move to **Project 07**.
