# Word-Level MLP Language Model, Built with NumPy from Scratch

A word-level neural language model built entirely in NumPy, with no deep learning frameworks. The model predicts the next word in a sequence using a learned embedding table and a configurable multi-layer perceptron trained via manually implemented forward and backward passes.

This was heavily inspired by Andrej Karpathy's `makemore` progression from the *Neural Networks: Zero to Hero* series. This notebook extends that work to the **word level** (vs. character level) and uses a class-based architecture designed for modularity and experimentation.

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
- Adjustable models through `parent.layers.append(self)` so you can add/remove layers without touching training logic
- `train()`, `evaluate_loss()`, `sample()`, `plotloss()` all on the `MLP` class
- `calc_mean_var()` for computing true dataset statistics vs. EMA-tracked running stats (though it's significantly more inefficient than running stats)

**Dataset:**
- Tiny Shakespeare text, processed into word-level context windows (context size = 4 but easily adjustable). the same MLP class can be trained on any dataset by changing the data loading and preprocessing steps.
- 80/10/10 train/dev/test split with shuffled sentences

---

## Models Trained

| Model | Architecture | Activation | Batch Size | Steps | LR1 | LR 2 |
|-------|-------------|-----------|-----------|-------|-------|
| `model` | Emb → Linear(256) → BN → Tanh → Proj | Tanh | 64 | 10k | 0.2 | 0.01 |
| `model2` | Emb → Linear(512) → BN → Tanh → Linear(512) → BN → Tanh → Proj | Tanh | 128 | 50k | 0.05 | 0.005 |
| `model3` | Same as model2 | ReLU | 128 | 50k | 0.1 | 0.0001 |
| `model4` | Same as model2 | Tanh | 128 | 50k | 0.05 | 0.001 |

LR1 and LR2 are the learning rates during the first and second half of iterations during training.

---

## Key Engineering Decisions & Debugging Notes

This notebook documents the full iterative process, not just the final working version. Notable investigations:

**Tanh saturation:** Early models showed heavy pre-activation mass at ±saturated values (or so I thought). Traced to embedding scale (`C` initialized without `* 0.1`), output projection weight scale, and batchnorm gain drift. Fixed by scaling `C` by 0.1 and using `w_scale = 0.01` for the final linear layer. Saturation at ±1 post-training was ultimately determined to be the model learning low-probability representations, not a training pathology.

**BatchNorm EMA drift:** Running mean diverged significantly from true dataset mean (avg absolute difference ~0.08 despite true mean range of ~[-0.1, 0.1]). Investigated EMA momentum values (0.1 → 0.01 → 0.1), confirmed that batch mean range itself grows over training as `bngain` scales up, causing the drift. Full-pass `calc_mean_var()` was implemented as a ground-truth reference. After many efforts, the practical takeaway was that some drift is inevitable and not necessarily a problem.

**ReLU dead neurons*:** In efforts to reduce tanh saturation, I switched to ReLU; ~8,500 dead units at iteration 1, ~7,500 at 10k. Rejected ReLU in favor of tanh.

**Stop token bug:** Model produced non-terminating samples (infinite sentences). Root cause: stop token (`<S>` / index 0) was not appended to training sequences, so the model never learned to predict sentence boundaries. Fixed by adding `Y.append(0)` at the end of each sentence in `build_dataset()`. Stop token probability rose from ~0 to ~0.013 post-fix. Stupid mistakes can have huge consequences.

---

## What I Learned / Why I Built This

The goal was to develop genuine mechanistic understanding of what a training loop actually does rather than just importing `torch.nn` and calling `.backward()`. I definitely deepened my understanding of which/how gradients flow where and why they have the shapes they do.

Specific things this forced me to understand deeply:
- Why Kaiming initialization matters per-layer (and why the output projection needs a different scale)
- What batchnorm and cross entropy backward pass actually computes and where Bessel's correction appears
- Why EMA running stats can diverge from dataset statistics, and under what conditions
- The difference between tanh saturation as a *training problem* vs. a *representation choice*
- How data preprocessing effects training deeply (the stop token bug was a great example of this))

This is one of the first steps in a broader self-study sequence on deep learning where I aim to develop myself as a "deep learning engineer:" someone who can build and train models from scratch, understand the math and mechanics at a fundamental level, and apply that understanding to new problems. I thoroughly enjoyed making this, which affirms that this is the right path for me.

---

## Running the Notebook

Requirements: `numpy`, `matplotlib`. No PyTorch or other ML libraries used.

```bash
pip install numpy matplotlib
jupyter notebook mlp_numpy.ipynb
```

The notebook downloads Tiny Shakespeare automatically on first run. Training models 2-4 takes ~3 hours each on CPU. Model 1 is much faster (~15 minutes) due to smaller size and fewer steps.

---

## Major Takeaways

**Jumping to model complexity as a fix for problems is not always the move.** It sounds obvious, but it's still a lesson I had to learn the hard way. I retrained the models ridiculous amounts of times, adding layers or neurons or iterations in attempts to minimize loss. To some extent, it worked. But, the difference between my simple model (Model 1) and the other, more complex ones was significantly smaller than I anticipated. Unfortunately, a 4-word context window from one text might impose a ceiling on the model that complexity can't bypass. More parameters can't extract information that isn't in the input. 

**Stupid mistakes will cost you.** The biggest (and most embarassing) failure in this project was not appending the stop token to training sequences. That one missing line in `build_dataset()` caused every model to produce non-terminating samples. And, it took me a ridiculously long time to catch. The debugging path made it worse: the first suspect was the activation function, then the batchnorm statistics, then the model architecture. It took printing `np.unique(Ytr)` and realize index 0 didn't appear to finally trace it back to data. I need to be more careful, meticulous, and (most of all) learn to recognize when a bug is a typo versus an actual training issue.

**"Perfection" isn't the goal.** I spent hours (aross many days) trying to "fix" my tanh saturation issue before realizing it wasn't an issue in the first place. I zeroed biases, scaled down output projections, added layers, switched to ReLU (which immediately failed), and removed Bessel's correction. None of it "solved" saturation. The actual turning point was when I started printing the tanh input distribution thrhoughout training and realized that saturation was emerging *over time* rather than present from the start. The model was just doing it's job. Saturation at convergence reflected the model learning low-probability representations, not a broken training loop.

**A training run is high-stakes.** Each run is a multi-hour commitment, not to mention the moment you change the MLP class itself you have to rerun all your models with the updated class. That changes how you have to think. Catch bugs early and be extremely thoughtful in the models you make. That means make sure all the stat tracking, diagnostic hooks, and features you want are baked into the class *before* training. The practical workaround here was making all model state public so functions could be added post-hoc (like `calc_mean_var()` was), but that's a band-aid. The ReLU switch is a good example of what happens when you don't: an entire 50k-step run wasted to confirm that 8,500 neurons died at iteration 1 and never recovered. A 3-hour run that produces a broken model isn't just a failured run, it's an expensive one. This was a small NumPy model on CPU, and the bigger the model, the higher the stakes.

**This was a tiny model, and it was already hard to reason about.** One embedding table, two linear layers, two batchnorm layers, and a softmax, and tracking distributions, running statistics, gradient flow, and saturation was already a lot to keep track of. The complexity of monitoring what's happening inside the network after every matmul, activation, and norm scales exponentially. Finding useful visualizations of data and stat tracking is probably crucial.

**Evaluating a model is harder than training one.** Loss numbers are easy to compute but open to interpretation — Model 2 had the lowest dev loss but also the most overfitting, while Model 3's numbers are essentially meaningless due to a learning rate typo that tanked the second half of training. Qualitative evaluation isn't much cleaner: the generated samples across all four models look similar enough that calling a winner from a few sentences is more gut feeling than analysis. Even outsourcing to an LLM evaluator only gets you so far. The honest answer after all of this is that Model 4 is probably best, but "probably" is doing a lot of work.

**At the risk of looking stupid, I made sure to document every little hiccup I ran into because through them, I learned the most. Having to dig into the model to look at input distributions and try different scaling factors at initialization, I started to gain a real understanding of how the model worked behind the scenes. And, I will never forget a stop token again lol.**

---


```
mlp_numpy.ipynb    # Main notebook with literally everything
```