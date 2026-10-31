> **[📚 Training Book Index](./README.md)** | [← Chapter 5: Parameters Explored](./05-parameters-explored.md) | [Chapter 7: Critical Bugs & Forensic Audits →](./07-critical-bugs-and-audits.md)

---

## 6. Training Strategies {#6-training-strategies}

### Strategy 1: Pure Self-Play

```
┌─────────────────────────────────────────────────────────────┐
│                    Pure Self-Play                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Network plays against itself                             │
│   ─────────────────────────────                             │
│   P1: Network (exploration=0.3)                             │
│   P2: Network (exploration=0.3)                             │
│                                                             │
│   Pros: No external dependency, improves together            │
│   Cons: Can converge to weird equilibria, no strategic      │
│         pressure from competent opponents                   │
│                                                             │
│   Result: 37-49% vs Random, 0% vs Heuristic                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiments TF-B1_v3, TF-B1_pure:** Pure self-play baselines
- Final vs Random: 37-49%
- Final vs Heuristic: 0%

The network learned basic legality but no strategic depth.

### Strategy 2: Mixed Opponents ("Learn from the Master" vs Curriculum Learning)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           WHY OPPONENT MIX FAILED IN STAGE 1 BUT WORKED IN STAGE 4           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Attempt A: "Learn from the Master" on Feedforward Net (exp21 & TF-B2)       │
│  ─────────────────────────────────────────────────────────────────────       │
│  • Config: 40%–70% Heuristic, 20%–30% Network, 10%–30% Random                │
│  • Hypothesis: Playing against the Heuristic master will expose the AI       │
│                to strong territorial play and force it to adapt.             │
│  • What Actually Happened:                                                   │
│      - AI lost 100% of games vs Heuristic (avg score: AI = 4, Heur = 170).   │
│      - In Policy Gradient, a 100% loss rate means EVERY move made against    │
│        Heuristic receives a negative gradient (-1)!                          │
│      - Worse: exp21's vs-Random win rate COLLAPSED from 91% down to 49%!     │
│        Constant negative punishment erased the network's existing intuition. │
│                                                                              │
│  Attempt B: 50/50 Mixed Opponents + Quiescence MCTS (TF-B10 & TF-B12)        │
│  ────────────────────────────────────────────────────────────────────        │
│  • Config: 50% QuiescenceMCTS-50 Self-Play + 50% Heuristic Opponent          │
│  • Why It Succeeded Here (0% ──► 63% vs Heuristic in TF-B12!):               │
│      1. Quiescence MCTS lookahead allowed the AI to start winning ~20–60%    │
│         of training games against Heuristic, providing POSITIVE target       │
│         trajectories alongside losses.                                       │
│      2. Even when the AI lost a game against Heuristic, MCTS policy targets  │
│         π_MCTS trained the Policy Head toward the locally best defensive     │
│         replies discovered by the search tree, while only the Value Head V   │
│         absorbed the -1 outcome!                                             │
│                                                                              │
│  Attempt C: Pure Dual-Side Quiescence MCTS Self-Play (TF-B14 & TF-B15)       │
│  ─────────────────────────────────────────────────────────────────────       │
│  • Config: 100% Dual-Side QuiescenceMCTS-50 (Zero Heuristic in training!)    │
│  • Result: 90% (TF-B14) ──► 99% Raw / 100% +Safety (TF-B15) vs Heuristic!    │
│  • Insight: Once the search tree itself models cascade dynamics via          │
│    Quiescence rollouts, pure self-play surpasses mixed-opponent training     │
│    because BOTH sides of every training game generate high-entropy,          │
│    50-simulation tactical search targets!                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Experiments `exp21` (`brain.js`) & `TF-B2` (`TF.js`):** Mixed opponent feedforward training
- `exp21` ("Learn from the Master"): Degraded from `91%` to **`49%` vs Random**, **`0%` vs Heuristic**
- `TF-B2_v3`, `TF-B2_mixed`: **`43–51%` vs Random**, **`0%` vs Heuristic**

**Key Systems Lesson:** In pure Policy Gradient RL, **an opponent that beats you 100% of the time teaches you nothing**—it only injects uniform negative gradients that destroy existing policy structure. Curriculum opponents (`Heuristic`) only become useful once your search architecture (`QuiescenceMCTS`) is strong enough to survive their tactical traps and separate move quality ($\pi_{\text{MCTS}}$) from game outcome ($z$).

### Strategy 3: Cluster Reward Shaping ($TD(\lambda)$) vs True Value Critics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 CLUSTER REWARD SHAPING (TD-λ) vs VALUE HEAD                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Traditional Policy Gradient: Reward only at game end (Ply 170)             │
│   TD(λ) Shaped Reward: Blend immediate cluster growth with final outcome     │
│                                                                              │
│   Shaped Reward Formula (exp17, exp22, TF-B3):                               │
│   ────────────────────────────────────────────                               │
│   • r_immediate = 0.3 × enclosed_count + 0.1 × max_cluster_delta             │
│   • TD(λ) trace: λ = 0.9, γ = 0.99                                           │
│                                                                              │
│   Results Across Engines:                                                    │
│   ───────────────────────                                                    │
│   • exp17b / exp22 (brain.js): +5% boost vs Random (70% → 75%), 0% vs Heur   │
│   • TF-B3 (TF.js feedforward): 43–52% vs Random, 0% vs Heuristic             │
│                                                                              │
│   Why Hand-Crafted Reward Shaping Hit a Ceiling:                             │
│   ──────────────────────────────────────────────                             │
│   • Rewarding immediate cluster growth (+0.1 per triangle) encourages        │
│     greedy outward expansion ("building long thin walls").                   │
│   • Against HeuristicPlayer, building long unclosed walls is suicidal:       │
│     Heuristic waits until the wall is 90% complete, steps in to close the    │
│     last gap, and steals the entire cavity + bonus turns!                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Finding:** Hand-crafted intermediate rewards ($TD(\lambda)$ cluster bonuses) help beat naive `Random` players by encouraging piece connectivity, but **backfire against tactical opponents** because they reward *greedy local expansion* over *defensive cavity prevention*. Replacing hand-crafted reward shaping with a learned **Value Head ($V(s) \in [-1, +1]$) trained on pure terminal game outcomes ($z \in \{-1, 0, +1\}$) + Quiescence MCTS** eliminated this bias completely.

### Strategy 4: Optimized Hyperparameters

```
┌─────────────────────────────────────────────────────────────┐
│                 Optimized Training                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Key changes from baseline:                               │
│   ──────────────────────────                               │
│   • learningRate: 0.05 → 0.01 (more stable)               │
│   • iterations: 10 → 100 (more thorough)                   │
│   • batchSize: 256 → 32 (better gradients)                 │
│   • gradientIsolation: true (cleaner updates)             │
│                                                             │
│   Training time: ~35 min for 10 cycles                     │
│                                                             │
│   Result: 47% vs Random (best TF.js result)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiment TF-B1_optimized:** Optimized hyperparameters
- Final vs Random: 47%
- Final vs Heuristic: 0%

**Key lesson:** Hyperparameter tuning improves performance but doesn't change the fundamental capability ceiling.

---

---

> [← Chapter 5: Parameters Explored](./05-parameters-explored.md) | **[📚 Training Book Index](./README.md)** | [Chapter 7: Critical Bugs & Forensic Audits →](./07-critical-bugs-and-audits.md)
