# Word-Level MLP Language Model, Built with NumPy from Scratch

A word-level neural language model built entirely in NumPy, with no deep learning frameworks. Predicts the next word in a sequence using a learned embedding table and a multi-layer perceptron trained via manually implemented forward and backward passes.

This was heavily inspired by Andrej Karpathy's `makemore` from the *Neural Networks: Zero to Hero* series, but extended to the word level with a class-based architecture.

---

## What's Implemented

**Architecture (fully from scratch):**
- `InputLayer` — embedding lookup table (`C`) with manual gradient accumulation over context indices
- `LinearLayer` — matrix multiply + bias, Kaiming He initialization, manual `dW` / `db` / `dinput`
- `BatchNorm` — full train/eval mode, running mean/variance tracking, condensed analytic backward pass (derived on paper)
- `Tanh` / `ReLU` — with saturation diagnostics and distribution visualizations
- `SoftMax` — numerically stable (max-subtracted), with cross-entropy loss and analytic backward

**Training infrastructure:**
- Minibatch SGD with two-phase learning rate schedule
- `train()`, `evaluate_loss()`, `sample()`, `plotloss()` on the `MLP` class
- `calc_mean_var()` for true dataset stats vs. EMA running stats

**Dataset:**
- Tiny Shakespeare, word-level context windows (context size = 4), 80/10/10 split with shuffled sentences

---

## Models Trained

| Model | Architecture | Activation | Batch Size | Steps | LR1 | LR 2 |
|-------|-------------|-----------|-----------|-------|-------|-------|
| `model` | Emb → Linear(256) → BN → Tanh → Proj | Tanh | 64 | 10k | 0.2 | 0.01 |
| `model2` | Emb → Linear(512) → BN → Tanh → Linear(512) → BN → Tanh → Proj | Tanh | 128 | 50k | 0.05 | 0.005 |
| `model3` | Same as model2 | ReLU | 128 | 50k | 0.1 | 0.0001 |
| `model4` | Same as model2 | Tanh | 128 | 50k | 0.05 | 0.001 |

LR1 and LR2 are the learning rates during the first and second half of iterations during training.

---

## Key Takeaways

1. **Tanh saturation:** Spent days trying to fix this before printing input distributions throughout training and realizing it wasn't a problem. Saturation at convergence is the model learning low-probability representations.
2. **BatchNorm EMA drift:** Running mean diverged from true dataset mean by ~0.08 despite a true mean range of ~[-0.1, 0.1]. Traced to `bngain` scaling up batch mean range over training. Implemented `calc_mean_var()` as a ground-truth reference and tweaked a bunch of hyperparameters before realizing that some drift is unavoidable.
3. **ReLU dead neurons:** Switched from tanh to try to fix saturation. ~8,500 dead units at step 1, ~7,500 at 10k. Immediately switched back.
4. **Stop token bug:** Every model produced infinite sentences. One missing `Y.append(0)` in `build_dataset()` meant the model never saw a reason to stop. Suspected activations, then batchnorm, then architecture before realizing the issue. Check the data first.
5. **More layers didn't help much.** The gap between Model 1 and Models 2-4 was smaller than expected. A 4-word context window and simple MLP structure imposes a ceiling that more parameters can't fix.


## Evaluating the Model

There was really no clear answer here. Model 2 had the lowest dev loss but also the most overfitting. Model 3's numbers are essentially meaningless due to a learning rate typo that tanked the second half of training. Qualitative evaluation isn't much cleaner: the generated samples across all four models look similar enough that calling a winner from a few sentences is more gut feeling than analysis. Even outsourcing to an LLM evaluator gave me inconsistent results. My best answer is that Model 4 is probably best, with a lot of emphasis on "probably."

---

## Running

Requirements: `numpy`, `matplotlib`. No PyTorch or other ML libraries used.

```bash
pip install numpy matplotlib
jupyter notebook mlp_numpy.ipynb
```

The notebook downloads Tiny Shakespeare automatically on first run. Training models 2-4 takes ~3 hours each on CPU. Model 1 is much faster (~15 minutes) due to smaller size and fewer steps.

```
mlp_numpy.ipynb    # Main notebook with literally everything
```