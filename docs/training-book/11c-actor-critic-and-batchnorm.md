> **[📚 Training Book Index](./README.md)** | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | [← 11B: Milestone Runs (TF-B5 to TF-B13)](./11b-milestone-runs-tf-b5-to-b13.md) | [11D: Monte Carlo Tree Search (MCTS) →](./11d-mcts-theory-and-implementation.md)

---

### Phase 3: Value Head (Actor-Critic) — Integrated into the 2-Head ConvNet

**Architectural Motivation:**
In pure Policy Gradient (Stage 1), the neural network only outputs move probabilities $\pi_\theta(a|s)$ ("which move looks promising?"), without any internal representation of state value $V_\theta(s)$ ("who is winning from this board configuration?"). Consequently, when a game is lost on Ply 170 due to a single late-game blunder, pure Policy Gradient has no way to separate the 38 solid positional moves that preceded the blunder from the blunder itself.

**How the Actor-Critic Architecture Works (`src/server/ConvNetwork.cjs`):**
We integrated the **Value Head (Critic)** directly alongside the **Policy Head (Actor)** on top of the shared 3-layer convolutional trunk (`Conv1 → Conv2 → Conv3`). Because canonical 18-channel slot encoding, spatial convolutions, and state-value estimation are deeply interdependent, building them as a unified 2-head architecture ensured that the shared convolutional filters learned spatial features useful for both move selection and position evaluation:

- **The Actor (Policy Head):** Applies a $1\times 1$ bottleneck convolution (`64 → 2` channels), flattens the `[10, 10, 2]` feature map into `200` features, and projects to `400` action logits trained via **Legal-Move Masked Softmax Cross-Entropy** against MCTS visit distributions $\pi_{\text{MCTS}}$.
- **The Critic (Value Head):** Applies a $1\times 1$ bottleneck convolution (`64 → 1` channel), flattens the `[10, 10, 1]` territory map into `100` cell scores, passes through a `64`-unit `ReLU` hidden layer, and outputs a scalar win assessment $V_\theta(s) \in [-1, +1]$ via `tanh`, trained via **Mean Squared Error (MSE)** against the true game outcome $z \in \{-1, 0, +1\}$.

<details>
<summary>📐 Diagram: Actor-Critic Architecture (click to expand ASCII)</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                    Actor-Critic Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Shared trunk (conv layers)                               │
│                      ↓                                      │
│            ┌────────────┴────────────┐                      │
│            ↓                         ↓                      │
│      Actor (Policy)             Critic (Value)               │
│   Input: Position              Input: Position               │
│   Output: Move probs            Output: Win prob [−1, +1]     │
│                                                             │
│   Training signal:              Training signal:             │
│   - Final outcome × policy     - TD(λ) from game outcomes   │
│   - MCTS improved policy       - Bootstrap from value net  │
│                                                             │
│   Benefit: Value head provides dense learning signal.      │
│   Even when you lose, you learn which positions were bad.   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

**Empirical Impact Across Stages (`TF-B5` → `TF-B15`):**
- **Preventing `tanh` Saturation (Bug 11):** As detailed in Chapter 7, pairing a `tanh` Value Head with an aggressive `momentum(0.05, 0.9)` optimizer while the Policy Head suffered from $1/400$ `.mean()` dilution initially drove the Value Head into permanent saturation (`-1.0000`). Switching to `.neg().sum(-1).mean()` for Policy Cross-Entropy and a persistent `tf.train.adam(0.001)` optimizer kept $V_\theta(s)$ well-calibrated across $[-1, +1]$.
- **Unlocking Lookahead Search:** Once calibrated, the Value Head enabled MCTS (`TF-B9`/`TF-B10`) and Quiescence MCTS (`TF-B12`–`TF-B15`) to evaluate leaf positions 5–10 plies ahead without running slow random rollouts to the end of the game—driving our climb from `50%` to **`100%` vs Random** and **`99%` Raw / `100%` +Safety vs Heuristic**.

**System Component:** Actor-Critic Value Head (integrated directly into the 2-Head Convolutional Network, `src/server/ConvNetwork.cjs`)

---

### Phase 4: Batch Normalization Across the Convolutional Trunk

**Architectural Motivation:**
Stacking multiple convolutional layers (`18 → 32 → 64 → 64` channels) followed by dense Policy and Value heads introduces **internal covariate shift**: as weights in `Conv1` update during backpropagation, the activation distributions feeding `Conv2`, `Conv3`, and the `tanh` Value Head shift continuously, risking vanishing or exploding gradients during long self-play runs.

**How Batch Normalization Stabilizes Self-Play (`src/server/ConvNetwork.cjs`):**
We placed a `tf.layers.batchNormalization()` layer immediately after every `Conv2D` operation (before the `ReLU` nonlinearity) across both the shared trunk (`Conv1`, `Conv2`, `Conv3`) and the $1\times 1$ Policy and Value bottleneck convolutions:
$$\hat{z}^{(l)} = \gamma^{(l)} \frac{z^{(l)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta^{(l)}$$

<details>
<summary>📐 Diagram: Deep Residual Network with BatchNorm (click to expand ASCII)</summary>

```
┌─────────────────────────────────────────────────────────────┐
│              Deep Residual Network with BatchNorm            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU    │
│                      ↓                                      │
│                Residual connection (add input)              │
│                      ↓                                      │
│   (Repeat 4-6 times)                                        │
│                      ↓                                      │
│            Policy Head + Value Head                         │
│                                                             │
│   BatchNorm benefits:                                       │
│   • 10x faster convergence                                  │
│   • Allows deeper networks without gradient issues          │
│   • More stable training (fewer hyperparameter tweaks)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

**Empirical Impact & Hardware Synergy:**
- **Zero Gradient Explosions Across 50+ Self-Play Cycles:** Across `TF-B5` through `TF-B15` (over `500,000` gradient steps), `BatchNormalization` maintained well-behaved activation variances with zero `NaN` losses or gradient explosions, even when warming up at `LR = 0.05` under cosine annealing.
- **Large Mini-Batch Synergy (`batchSize = 256`):** Because Batch Normalization computes batch statistics $(\mu_B, \sigma_B^2)$ over the active mini-batch, running `batchSize = 256` on the Mac Studio M2 Ultra provides both ~98% AMX matrix utilization and low-noise batch statistics during training (`training: true`), while seamlessly switching to moving-average population statistics during MCTS leaf evaluation (`training: false`).

**System Component:** Batch-Normalized Convolutional Trunk (integrated directly into the 2-Head Convolutional Network, `src/server/ConvNetwork.cjs`)

---

> [← 11B: Milestone Runs (TF-B5 to TF-B13)](./11b-milestone-runs-tf-b5-to-b13.md) | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | **[📚 Training Book Index](./README.md)** | [11D: Monte Carlo Tree Search (MCTS) →](./11d-mcts-theory-and-implementation.md)
