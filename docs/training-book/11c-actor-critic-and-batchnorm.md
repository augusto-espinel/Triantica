> **[📚 Training Book Index](./README.md)** | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | [← 11B: Milestone Runs (TF-B5 to TF-B13)](./11b-milestone-runs-tf-b5-to-b13.md) | [11D: Monte Carlo Tree Search (MCTS) →](./11d-mcts-theory-and-implementation.md)

---

### Phase 3: Value Head (Actor-Critic) — MERGED INTO PHASE 2

**Status:** ✅ Merged — Value head added as step 2.5 in the combined Phase 2 plan. Canonical slot encoding + conv layers + value head are being implemented together since they're deeply interdependent. A value head on top of the wrong encoding would give misleading signal; the canonical encoding makes the value head's job tractable.

**Original Problem:** Network only learns "what moves are good" (policy). It doesn't know "how good is my position" (value).

**Legacy Phase 1 Baseline:** Pure policy gradient — only learns from game outcomes.

**Implemented Actor-Critic Architecture (`ConvNetwork.cjs`):** Learn both policy ($\pi_\theta$) and value ($V_\theta \in [-1, +1]$) simultaneously on top of the shared convolutional trunk.

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

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 3.1 | Add value head to conv network (flatten → dense → tanh) | 45 min |
| 3.2 | Implement value loss: MSE(predicted, actual) | 30 min |
| 3.3 | Combine policy loss + value loss (weighted) | 30 min |
| 3.4 | Implement TD(λ) returns for value targets | 1 hour |
| 3.5 | Modify training loop: compute both losses | 45 min |
| 3.6 | Add value prediction to evaluation logging | 20 min |
| 3.7 | Test: Verify value predictions are in [−1, +1] | 15 min |
| 3.8 | Run experiment TF-B6 (actor-critic, 20 cycles) | 50 min |
| 3.9 | Analyze: correlation between value and outcome | 30 min |
| 3.10 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria (Verified across `TF-B5` → `TF-B15`):**
- [x] Value predictions correlate with outcomes in $[-1, +1]$ once single-perspective targets and unsaturated initialization (`Bug 6` & `Bug 11`) are applied
- [x] Training converges smoothly with combined Policy Cross-Entropy + Value MSE loss
- [x] vs Random improves beyond 65%+ when Value Head guides MCTS (`79%` in `TF-B9`, `87%` in `TF-B10`, **`100%`** in `TF-B12`–`B15`)
- [x] vs Heuristic breaks the 0% wall when Value Head evaluates quiescent post-cascade leaves (`63%` → **`100%`**)

**M2 Ultra Optimization:**
- Larger batches for value loss (32-64)
- Use `tf.losses.meanSquaredError` with reduction=SUM
- Value loss weight: 0.5 (tune after initial results)

**System Component:** Actor-Critic Value Head (integrated directly into the 2-Head Convolutional Network, `src/server/ConvNetwork.cjs`)

---

### Phase 4: Batch Normalization + Deeper Network

**Priority:** Medium (stability for deeper networks)

**Problem:** Deeper networks are harder to train. Gradients vanish or explode.

**Solution:** Batch normalization after each conv layer.

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

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 4.1 | Add `tf.batchNormalization()` after each conv | 30 min |
| 4.2 | Implement residual block (conv → BN → ReLU → conv → BN → add skip) | 1 hour |
| 4.3 | Stack 4 residual blocks | 20 min |
| 4.4 | Test: Verify gradients don't explode (add gradient clipping) | 30 min |
| 4.5 | Run experiment TF-B7 (deep network, 20 cycles) | 60 min |
| 4.6 | Compare training curves vs Phase 3 | 20 min |
| 4.7 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria (Verified — Integrated into `ConvNetwork.cjs`):**
- [x] Training reaches lower, stable loss with `BatchNormalization` after every `Conv2D` layer
- [x] No NaN losses (gradient stability verified across 40+ MCTS cycles in `TF-B12`–`TF-B15`)
- [x] vs Random reaches 70%+ (`79%` in `TF-B9` → **`100%`** in `TF-B12`–`TF-B15`)
- [x] vs Heuristic exceeds 3–5% target (`63%` in `TF-B12` → **`99%` raw / `100%` +Safety** in `TF-B15`)

**M2 Ultra Optimization:**
- BatchNorm works better with larger batches (64+)
- Use `training: true` in BatchNorm during training
- Lower learning rate (0.001 → 0.0005) for stability

**System Component:** Batch-Normalized Convolutional Trunk (integrated directly into the 2-Head Convolutional Network, `src/server/ConvNetwork.cjs`)

---

---

> [← 11B: Milestone Runs (TF-B5 to TF-B13)](./11b-milestone-runs-tf-b5-to-b13.md) | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | **[📚 Training Book Index](./README.md)** | [11D: Monte Carlo Tree Search (MCTS) →](./11d-mcts-theory-and-implementation.md)
