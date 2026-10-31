> **[📚 Training Book Index](./README.md)** | [← Chapter 4: Hardware & Matrix Sizing](./04-hardware-m2-ultra.md) | [Chapter 6: Training Strategies →](./06-training-strategies.md)

---

## 5. Parameters Explored {#5-parameters}

### Parameter Overview

Reinforcement learning pipelines expose four interacting families of knobs inside the training orchestrator (`src/server/train_tfjs.cjs` and `src/server/train_headless.js`):

1. **Optimization Hyperparameters (`learningRate`, `lrSchedule`, `iterations`, `batchSize`, `momentum`):** Govern how aggressively and smoothly neural network weights adjust to new self-play batches.
2. **Exploration & Tree Search Controls (`exploration`, `mctsSims`, `cPuct`, `dirichletAlpha`, `temperature`):** Control the trade-off between testing novel board lines and exploiting the strongest known moves.
3. **Self-Play & Replay Memory (`games`, `cycles`, `opponentMix`, `replayBufferSize`, `replayRatio`):** Determine the volume, opponent curriculum, and historical window of training positions.
4. **Reward & Loss Formulation (`gradientIsolation`, `enableClusterReward`, `quiescenceMode`, `tacticalSafetyFilter`):** Define the mathematical objective that defines a "good" move.

<details>
<summary>📐 Diagram: Parameter Categories (click to expand ASCII)</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                   Parameter Categories                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Hyperparameters                                   │
│  ──────────────────────────                                 │
│  • learningRate (α) — Step size for gradient descent        │
│  • iterations — Training epochs per cycle                   │
│  • batchSize — Samples per gradient update                 │
│  • momentum — Optimizer dampening                           │
│                                                             │
│  Exploration Parameters                                     │
│  ──────────────────────────                                 │
│  • exploration (ε) — Random move probability               │
│  • explorationDecay — Reduction over time                  │
│                                                             │
│  Self-Play Configuration                                    │
│  ──────────────────────────                                 │
│  • games — Games per training cycle                         │
│  • cycles — Number of training iterations                   │
│  • opponentMix — Distribution of opponent types             │
│  • roleAlternation — Whether AI plays both sides           │
│                                                             │
│  Reward Engineering                                        │
│  ─────────────────────                                       │
│  • clusterRewardFactor — Bonus for territory clustering     │
│  • clusterLambda — TD(λ) decay rate                         │
│  • clusterGamma — Discount factor for future rewards        │
│                                                             │
│  Technical Options                                          │
│  ─────────────────────                                       │
│  • gradientIsolation — Isolate gradient to chosen move    │
│  • enableClusterReward — Use TD(λ) rewards                  │
│  • startFrom — Initialize from existing network            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Learning Rate (α)

**What it does:** Controls how much the network weights change with each gradient update.

<details>
<summary>📐 Diagram: High Learning Rate (0.1+) (click to expand ASCII)</summary>

```
                    High Learning Rate (0.1+)
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │  UNSTABLE - oscillates, diverges   │
         │  Weights bounce around, never settle│
         └─────────────────────────────────────┘
         
         ┌─────────────────────────────────────┐
         │  SWEET SPOT: 0.01 - 0.05           │
         │  - Smooth convergence              │
         │  - Good exploration                 │
         │  - Stable learning                  │
         └─────────────────────────────────────┘
         
         ┌─────────────────────────────────────┐
         │  TOO SLOW: 0.001 - 0.005           │
         │  - Takes forever to converge       │
         │  - Gets stuck in local optima       │
         │  - Never reaches peak performance   │
         └─────────────────────────────────────┘
                            ▲
                    Low Learning Rate (0.001-)
```

</details>

| Value | Observed Behavior | Best For |
|-------|-------------------|----------|
| 0.001 | Very slow convergence, risk of local minima | Fine-tuning trained models |
| **0.01** | Smooth learning, stable convergence | **Default choice** |
| 0.05 | Faster learning, slight oscillation | Fresh training |
| 0.1+ | Diverges, unstable, weights explode | Avoid |
**Key insight:** Learning rate affects *convergence speed*, not final performance. A good architecture at lr=0.01 eventually reaches the same peak as at lr=0.05.

**Momentum & Cosine Annealing (`TF-B14` & `TF-B15`):**
In early experiments we used SGD with `momentum = 0.9` to dampen gradient oscillations:

```
Momentum accumulates past gradients:

velocity = momentum × velocity + gradient
weight = weight - learningRate × velocity

Effect: Dampens oscillations, accelerates convergence
```

In our champion AlphaZero runs (`TF-B14` and `TF-B15`), we combined the Adam optimizer (`tf.train.adam`) with a **10-cycle Cosine Annealing Learning Rate Schedule (`0.05 → 0.005`)**:
$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})\left(1 + \cos\left(\frac{t - 1}{T - 1}\pi\right)\right)$$
Starting at $\alpha_1 = 0.05$ allows rapid reorganization of the convolutional filters during early self-play cycles when the replay buffer is first populated, while smoothly decaying to $\alpha_{10} = 0.005$ prevents late-cycle weight updates from overwriting subtle endgame and border-wrapping defenses.

### Replay Window Sizing & Opponent Curriculum Parameters

Two additional parameter discoveries proved decisive in Stages 4 and 5:
- **Sliding-Window Replay Buffer (`--replay-buffer-size 45000` vs. `500000`):** An unbounded `500,000`-sample buffer retains Cycle 1 amateur self-play blunders for the entire run, anchoring the Policy Head to weak early habits ("replay buffer pollution"). Constraining the buffer to a **45,000-sample sliding window** (~3.5 cycles of fresh MCTS self-play) evicts obsolete games automatically as the agent improves.
- **Exploration Decay vs. MCTS Temperature (`--mcts-temp 1.0 → 0.1`):** Rather than injecting uniform random blunders throughout the game (`exploration = 0.3`), sampling moves proportionally to MCTS visit counts ($N(s,a)^{1/\tau}$) with $\tau = 1.0$ for the first 15 plies and $\tau = 0.1$ thereafter preserves high opening diversity (`5.6+ bits`) without sabotaging mid-game tactical precision.

---

> [← Chapter 4: Hardware & Matrix Sizing](./04-hardware-m2-ultra.md) | **[📚 Training Book Index](./README.md)** | [Chapter 6: Training Strategies →](./06-training-strategies.md)
