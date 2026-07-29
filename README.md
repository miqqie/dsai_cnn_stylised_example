# Convolutional Neural Networks Explained: Detecting a Vertical Edge

This guide walks through the complete lifecycle of a CNN's training process using a concrete, fully-worked example: a single 3×3 kernel sliding across a 6×6 grayscale image patch to detect a vertical edge, then feeding a max-pooled feature into a one-neuron classifier. Starting from the raw pixels, it traces every step through the model — convolution, ReLU, max-pooling, backpropagation, and gradient descent — with exact numbers at each stage so nothing is left to "the network just figured it out."

The core takeaway: CNNs are not magic. They are a sequence of arithmetic operations — multiply, add, compare to zero — repeated across many small windows and many layers. Every number the model produces, from the first feature map to the final weight update, can be traced back to a human choice (kernel size, stride, padding, learning rate) or a straightforward calculation. The guide also bridges the walkthrough to real-world practice with diagnostics, hyperparameter tuning heuristics, and a ready-to-use PyTorch template.

**🔗 [Try the interactive HTML demo](https://huggingface.co/spaces/Miqqie/Convolutional_Neural_Networks)**

---

## Table of Contents
1. [The Dataset & Kernel — a 6×6 patch and a 3×3 edge detector](#1-the-dataset--kernel)
2. [Convolution — sliding one shared kernel across the image](#2-convolution)
3. [ReLU — discarding weak or absent signals](#3-relu)
4. [Max-Pooling, Flatten & the Output Neuron](#4-max-pooling-flatten--the-output-neuron)
5. [Backpropagation — Output Layer](#5-backpropagation--output-layer)
6. [Backpropagation — the Kernel](#6-backpropagation--the-kernel)
7. [Gradient Descent — Updating the Kernel](#7-gradient-descent--updating-the-kernel)
8. [Verifying the Improvement](#8-verifying-the-improvement)
9. [Hyperparameters — what you choose vs. what the network learns](#9-hyperparameters)
10. [From Walkthrough to Production — a PyTorch Template & Rules of Thumb](#10-from-walkthrough-to-production--a-pytorch-template--rules-of-thumb)

---

## 1. The Dataset & Kernel

**Objective:** define the raw input and the one set of weights that will slide across all of it.

Our 6×6 grayscale patch has a vertical brightness transition running down the middle — dark on the left, bright on the right — identical in every row:

```
0.1  0.1  0.5  0.9  0.9  0.9
0.1  0.1  0.5  0.9  0.9  0.9
0.1  0.1  0.5  0.9  0.9  0.9
0.1  0.1  0.5  0.9  0.9  0.9
0.1  0.1  0.5  0.9  0.9  0.9
0.1  0.1  0.5  0.9  0.9  0.9
```

The kernel is a 3×3 **vertical edge detector**, fixed for this walkthrough:

| −1 | 0 | 1 |
|---|---|---|
| −1 | 0 | 1 |
| −1 | 0 | 1 |

Bias = **−1.5**. The same 9 weights and 1 bias are reused at every window position — this **weight sharing** is what makes a kernel cheap (10 parameters total, regardless of image size) and translation-aware (the same edge is detected the same way wherever it appears).

**Why not a separate weight per pixel (a fully-connected layer)?** An FC layer on a 6×6 patch would need a separate weight for every one of the 36 pixels feeding every output — no weight sharing, no notion that neighbouring pixels matter, and no ability to recognise the same edge one pixel to the left without learning it from scratch as a completely different pattern.

[⬆ Back to top](#table-of-contents)

---

## 2. Convolution

**Objective:** slide the kernel across the image and compute one number per window — a measure of "how much does this window look like a vertical edge?"

The kernel visits every valid 3×3 window. Because every row of the input is identical, the four column-offset windows (call them **A**, **B**, **C**, **D**, at columns 0–2, 1–3, 2–4, 3–5) fully determine all 16 output positions — each window's answer just repeats down all 4 output rows.

**Window B (columns 1–3 → [0.1, 0.5, 0.9], straddling the true edge):**
```
Row dot product: (−1×0.1) + (0×0.5) + (1×0.9) = 0.8
Sum over 3 identical rows: 0.8 × 3 = 2.4
+ bias: 2.4 + (−1.5) = 0.9
```

| Window | Columns | Pixels | Pre-activation |
|---|---|---|---|
| A | 0–2 | [0.1, 0.1, 0.5] | −0.3 |
| **B** | **1–3** | **[0.1, 0.5, 0.9]** | **0.9** |
| C | 2–4 | [0.5, 0.9, 0.9] | −0.3 |
| D | 3–5 | [0.9, 0.9, 0.9] | −1.5 |

Only Window B — the one straddling the actual transition — produces a strong positive number. A, C, and D each sit entirely on one side of the edge (or entirely past it), and the kernel correctly finds nothing edge-like there.

**Why not just check if left < right?** A raw threshold check can't be learned or generalized — the kernel's weights are what get tuned by gradient descent (Section 6) to detect *this* pattern; a hardcoded rule can't adapt to a different kind of edge, a diagonal line, or a texture without a human rewriting it.

[⬆ Back to top](#table-of-contents)

---

## 3. ReLU

**Objective:** discard windows that don't match the pattern, keeping the feature map sparse and interpretable.

```
ReLU(A) = max(0, −0.3) = 0        (dead — no edge here)
ReLU(B) = max(0,  0.9) = 0.9      (survives — this is the edge)
ReLU(C) = max(0, −0.3) = 0        (dead)
ReLU(D) = max(0, −1.5) = 0        (dead — deep inside the bright region)
```

**Implication — dead cells:** 12 of the 16 feature-map cells (every column except B, across all 4 rows) are now exactly zero. This isn't wasted information — a zero *is* the answer "no edge here," and it's what makes max-pooling (Section 4) trivial: only Window B's signal can possibly survive the pool.

**Why not skip the activation and pass raw values forward?** Without a non-linearity, stacking any number of convolution layers collapses mathematically into one single linear operation — depth would add no representational power at all. ReLU's `max(0, z)` is what breaks that collapse and lets later layers combine *pattern-dependent* outputs from this layer instead of a fixed linear mix of raw pixels.

[⬆ Back to top](#table-of-contents)

---

## 4. Max-Pooling, Flatten & the Output Neuron

**Objective:** shrink the feature map to its strongest signals, then combine them into one prediction.

Max-pooling with a 2×2 window over the top-left region ([0, 0.9, 0, 0.9]) keeps only the strongest value:
```
max(0, 0.9, 0, 0.9) = 0.9
```
Repeated over the other three 2×2 regions of the 4×4 feature map, then flattened into a length-4 vector and fed to the output neuron (weight 0.8 on the relevant entry, bias −0.3):
```
z_out = (0.9 × 0.8) + (−0.3) = 0.42
ŷ = sigmoid(0.42) ≈ 0.6035   (60.35% confidence: "edge present")
```
True label **y = 1** (an edge really is present) — the network is under-confident and needs correcting.

**Why max instead of average-pooling here?** Max-pooling answers "is this pattern present anywhere in the window?" — exactly the question this task needs. Average-pooling would dilute Window B's strong 0.9 with three zero-valued neighbours, cutting the signal by more than half before it ever reaches the output neuron (see the Hyperparameters section for the worked-out comparison).

[⬆ Back to top](#table-of-contents)

---

## 5. Backpropagation — Output Layer

**Objective:** turn the loss into a signed, weight-specific instruction — not just "how wrong," but "which direction, and by how much."

### Loss
```
Loss = −ln(ŷ) = −ln(0.6035) ≈ 0.505
```

### Output error signal
```
δ_out = ŷ − y = 0.6035 − 1 = −0.3965
```
This single negative number is both magnitude (how wrong) and direction (predict higher) in one value — the loss alone (a positive scalar) can't tell you which way to move a weight; δ_out can.

### Gradients
```
∂L/∂w_out = δ_out × pooled_value = −0.3965 × 0.9 = −0.357
∂L/∂b_out = δ_out × 1 = −0.3965
```

**Why not use the loss directly as the gradient?** The loss is one number describing the *whole* network's error; a gradient has to describe *each weight's individual contribution* to that error, which is exactly what the chain rule (multiplying δ_out by whatever fed into that weight) computes.

[⬆ Back to top](#table-of-contents)

---

## 6. Backpropagation — the Kernel

**Objective:** push the blame for the output error back through pooling and ReLU into the one kernel that produced the surviving signal.

Blame flows backward through the same path the max-pool winner came from:
```
δ_pooled = δ_out × w_out = −0.3965 × 0.8 = −0.3172
```
Because ReLU's gate is "pass-through if positive, zero if not," and only Window B survived, **only Window B receives any gradient at all** — Windows A, C, and D get exactly zero, regardless of δ_pooled's size.

Window B's row was [0.1, 0.5, 0.9], so each kernel weight's gradient is δ_pooled × the pixel it multiplied:

| Weight | Pixel | Gradient |
|---|---|---|
| k[·][0] | 0.1 | −0.3172 × 0.1 = **−0.0317** |
| k[·][1] | 0.5 | −0.3172 × 0.5 = **−0.1586** |
| k[·][2] | 0.9 | −0.3172 × 0.9 = **−0.2855** |
| bias | (always 1) | **−0.3172** |

The same gradient row applies to all 3 kernel rows, since the 3 rows saw identical inputs — these 9 numbers (3 rows × 3 columns) are what Section 7 subtracts from the kernel.

**Implication — dead windows, frozen weights:** because A, C, and D contributed zero gradient, this step teaches the kernel nothing about what those windows looked like. If every window in a batch happened to be non-edge regions, the kernel would receive no learning signal at all that step — a real, if usually transient, cause of slow training.

[⬆ Back to top](#table-of-contents)

---

## 7. Gradient Descent — Updating the Kernel

**Objective:** nudge every weight a small step opposite its gradient.

With learning rate η = 0.1:
```
new_weight = old_weight − (η × gradient)

k[·][2]:  1     − (0.1 × −0.2855) = 1.0286   ↑
k[·][1]:  0     − (0.1 × −0.1586) = 0.0159   ↑
k[·][0]: −1     − (0.1 × −0.0317) = −0.9968  ↑
bias:    −1.5   − (0.1 × −0.3172) = −1.4683  ↑
w_out:    0.8   − (0.1 × −0.357)  = 0.8357   ↑
b_out:   −0.3   − (0.1 × −0.3965) = −0.2604  ↑
```
Every weight nudges the same direction (up) because every gradient in this step was negative — the network under-predicted, so every contributing weight increases to push the next prediction higher.

**Why not just subtract the raw gradient directly?** Without η scaling the step down, a single large gradient (common early in training, or on an outlier example) could overshoot the correct value entirely and destabilize training — η is what turns "the right direction" into "a safe-sized step in the right direction." See Section 9 for what happens when η is chosen poorly.

[⬆ Back to top](#table-of-contents)

---

## 8. Verifying the Improvement

**Objective:** confirm the update actually helped, using the network's own updated numbers — not by assumption.

| Quantity | Before | After one step |
|---|---|---|
| Window B pre-activation | 0.9000 | **1.0335** ↑ |
| Pooled value | 0.9000 | **1.0335** ↑ |
| ŷ (true label = 1) | 0.6035 | **0.6464** ↑ |
| Loss | 0.505 | **0.436** ↓ |

The prediction moved closer to the true label (1) and the loss dropped — a confirmed, if small, improvement from a single training step on a single image. Windows A, C, and D remain dead (still negative pre-activation) — the update sharpened the detector's confidence at the true edge without resurrecting the non-edge windows.

[⬆ Back to top](#table-of-contents)

---

## 9. Hyperparameters

**Objective:** separate what a human chooses before training from what the network learns during it.

### Kernel size, stride & padding — Architecture

| Setting | Walkthrough | Implication of increasing it |
|---|---|---|
| Kernel size | 3×3 | More context and parameters per window, but a smaller feature map and higher overfitting risk |
| Stride | 1 | A smaller, coarser feature map for less compute — downsampling without a separate pooling layer |
| Padding | 0 | (Increasing padding) edge pixels get covered as often as interior ones, keeping the map closer to the input's size |

**When it matters:** a single 3×3 kernel here has 10 parameters; stacking several small kernels (rather than using one large one) is how real CNNs build up a large receptive field cheaply — see the interactive demo's Q&A on choosing depth.

### Depth — how many conv/pooling layers? — Architecture

This walkthrough uses the smallest possible stack (one conv, one pool) to keep the arithmetic hand-checkable. Real CNNs repeat that block several times:

| Depth | Blocks | Best for |
|---|---|---|
| Shallow | 2–3 conv+pool blocks | Good starting point for small/simple datasets |
| Typical | 3–5 conv+pool blocks | Most small-to-medium image tasks |
| Very deep | Dozens of blocks | Only pays off with skip connections (ResNet) to stop gradients vanishing across that many layers |

**Why stack small kernels instead of using one big one?** Two stacked 3×3 kernels "see" a 5×5 region of the original input; three stacked see 7×7 — depth is how a network built from small kernels still learns to recognise large, whole-object patterns, at a fraction of one giant kernel's parameter cost. This is also the mechanism behind **higher-level features**: layer 1 detects edges, layer 2 combines edges into corners/textures, layer 3+ combines those into object parts and eventually whole objects — no single layer is told to detect "wheel" or "face"; the concept emerges purely from stacking simple detectors and letting each layer recombine what the last one found.

### Pooling type — Architecture

| Type | Formula on [0, 0.9, 0, 0.9] | Result | Best for |
|---|---|---|---|
| **Max** (used above) | max(cells) | 0.9 | "Is this pattern present anywhere?" |
| Average | mean(cells) | 0.45 | Overall regional intensity (e.g. global-average-pooling) |
| L2-norm | √(Σ cell²) | 1.273 | Rewarding a pattern appearing more than once in the window |
| Adaptive | window size derived from a fixed *output* size | varies | Accepting variable input sizes (e.g. ResNet's final layer) |

**When to worry:** average-pooling on this example dilutes 0.9 down to 0.45, moving z_out to 0.06 and ŷ to ≈0.515 — a far less confident prediction than max-pool's 0.6035, even though the edge is exactly as real.

### Learning rate & optimiser — Optimisation

| Optimiser | Update rule | Key trait | Best for |
|---|---|---|---|
| **Plain SGD** (used above) | `W = W − η·grad` | Simple, cheap, sensitive to η | Small nets; full manual control |
| SGD + Momentum | `v = μv − η·grad; W = W + v` | Smooths noisy gradients | Vision tasks; default μ=0.9 |
| Adam | momentum + adaptive per-weight rate | Fast, forgiving of η | Default for most real CNNs |
| RMSprop | adaptive rate, no momentum | Similar to Adam minus momentum | RNNs, sequential tasks |
| Adagrad | adaptive, accumulator only grows | Rate shrinks over time | Sparse gradients (embeddings) |

**When η is too small / too large / just right:**

| Symptom | Meaning | Fix |
|---|---|---|
| 🟡 Loss barely moves | η too small | Raise η, or switch to Adam |
| 🔴 Loss bounces between batches | η too large — overshooting | Lower η, increase batch size |
| 🟢 Loss decreases steadily | Healthy | Keep going |

**Beyond this walkthrough — one rate for the whole network isn't always right:** with more than one conv layer, earlier layers receive systematically *smaller* gradients than later ones (each layer's gradient passes through one more chain-rule multiplication — see Section 6). The fix is a **higher** η for early layers, not lower — since their raw gradient is already shrunk, they need a *larger* step just to move by a similar amount per update as later layers. In PyTorch this is a per-parameter-group setting on one optimiser (`optim.SGD([{'params': early.parameters(),'lr':0.01}, {'params': late.parameters(),'lr':0.1}])`), not a different optimiser type per layer — mixing optimiser *types* across layers is rare and adds complexity with no established benefit over just varying η.

[⬆ Back to top](#table-of-contents)

---

## 10. From Walkthrough to Production — a PyTorch Template & Rules of Thumb

| Code section | Maps to |
|---|---|
| Architecture (`nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear`) | Sections 2–4 (forward pass structure) |
| Loss & optimiser (`BCELoss`, `SGD`) | Section 5 |
| Capacity check (`total_params` vs. dataset size) | Diagnostic: is the model too large for the data? |
| Training loop (`loss.backward()`, `optimizer.step()`) | Sections 5–7, repeated per image/batch |
| Validation (forward-only, no `backward()`) | Diagnostic: is the model overfitting? |

```python
model = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=3),   # 1 kernel, matches Sections 2-3
    nn.ReLU(),
    nn.MaxPool2d(2),                  # matches Section 4
    nn.Flatten(),
    nn.Linear(4, 1),                  # 2×2 pooled → 1 output
    nn.Sigmoid()
)
criterion = nn.BCELoss()                            # matches Section 5's -log(ŷ)
optimizer = optim.SGD(model.parameters(), lr=0.1)   # tune: switch to Adam for faster convergence

# Capacity check — run once, before training starts
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# Our example: 15 parameters (9 kernel weights + 1 conv bias + 4 output weights + 1 output
# bias) vs. 50,000 training images → 0.03%. Safe from overfitting.

for epoch in range(20):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)               # forward pass (Sections 2-4)
        loss = criterion(y_pred, y_batch)     # -log(ŷ), same as Section 5
        loss.backward()                       # backprop (Sections 5-6)
        optimizer.step()                      # gradient descent (Section 7)
```

**A tensor is more than an array.** It's what makes the four lines above work: PyTorch records every operation on a tensor in a computation graph, and `loss.backward()` replays that graph in reverse to compute every gradient automatically — the exact δ and gradient values hand-calculated in Sections 5–6, just for millions of pixels instead of nine.

**Dropout & weight decay** (not used in this walkthrough — a single 3×3 kernel has no real capacity to overfit, but both matter once a network is much bigger):

| Technique | What it does | When to use |
|---|---|---|
| Dropout (`nn.Dropout(p=0.3–0.5)`) | Randomly zeroes a fraction of units during training only, forcing patterns to spread across many kernels instead of a few fragile ones | Once a network has many more parameters than training images |
| Weight decay (`weight_decay=1e-4`) | Penalises large weights in the loss, nudging the optimiser toward smaller, more general weights | Same signal, or when validation loss rises while training loss keeps falling |

### Diagnostics & Rules of Thumb

| If you observe... | It usually means... | Try this |
|---|---|---|
| 🔴 Loss bouncing between batches | η too large, overshooting | Lower η, or increase batch size |
| 🟡 Loss decreasing very slowly | η too small, or plain SGD on a complex problem | Raise η, or switch to Adam |
| 🟢 Loss decreasing steadily | Healthy training | Keep going, watch validation loss |
| 🔴 High % of dead feature-map cells | Kernel producing mostly-negative activations | Switch to Leaky ReLU/ELU, or lower η |
| 🔴 Validation loss rising while training loss falls | Overfitting | Dropout, weight decay, early stopping, more data |
| 🟡 Parameters > training examples | Model too large for the dataset | Reduce architecture size, add regularisation |
| 🔵 Dataset is small (<10,000) | A single split may be unreliable | Use k-fold cross-validation |
| 🔵 Dataset is large (100,000+) | k-fold not worth the compute | Single fixed train/val/test split |

**The single throughline of this entire walkthrough:** every one of the network's "decisions" — kernel size, stride, padding, pooling type, learning rate — traces back to a number a human chose or a calculation you can redo by hand. Nothing here is magic; it's arithmetic, repeated at scale.

[⬆ Back to top](#table-of-contents)

---

**Note:** This example was adapted from standard convolutional neural network teaching material on vertical edge detection kernels.
