> **[📚 Training Book Index](./README.md)** | [← Chapter 4: Hardware & Matrix Sizing](./04-hardware-m2-ultra.md) | [Chapter 6: Training Strategies →](./06-training-strategies.md)

---

## 5. Parameters Explored {#5-parameters}

### Parameter Overview

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

**Momentum:** We use momentum=0.9 to smooth gradient updates.

```
Momentum accumulates past gradients:

velocity = momentum × velocity + gradient
weight = weight - learningRate × velocity

Effect: Dampens oscillations, accelerates convergence
```

---

---

> [← Chapter 4: Hardware & Matrix Sizing](./04-hardware-m2-ultra.md) | **[📚 Training Book Index](./README.md)** | [Chapter 6: Training Strategies →](./06-training-strategies.md)
