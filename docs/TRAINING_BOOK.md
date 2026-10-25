# Triantica: A Practical Guide to Reinforcement Learning

**From Zero to RL Practitioner — Learning Through a Real Game AI Project**

---

## About This Guide

This document is both a training journal and a comprehensive introduction to reinforcement learning (RL). If you're not an ML engineer but want to understand RL through hands-on experience, this guide walks through every concept, mistake, and breakthrough from the Triantica project.

**What is Triantica?** A grid-based strategy game where players place triangles to enclose territory. The AI learns through self-play reinforcement learning — playing millions of games against itself and improving over time.

**What you'll learn:**
- Core RL concepts (policy networks, reward shaping, exploration vs exploitation)
- Why architecture matters more than training tricks
- How to debug when your AI isn't learning
- The transition from simple frameworks to production-grade ML
- Hardware considerations for training (Mac Studio M2 Ultra specific)

**Prerequisites:** Basic programming knowledge. No ML background required.

---

## Table of Contents

1. [The Game: Why Triantica is Interesting for RL](#1-the-game)
2. [RL Fundamentals Through This Project](#2-rl-fundamentals)
3. [The Architecture Journey: brain.js → TensorFlow.js](#3-architecture-journey)
4. [Hardware: Mac Studio M2 Ultra Optimization](#4-hardware)
5. [Parameters Explored](#5-parameters)
6. [Reward Strategy: Step by Step](#6-reward-strategy)
7. [Training Strategies](#7-training-strategies)
8. [Critical Bugs and How We Found Them](#8-critical-bugs)
9. [The Heuristic Wall: Understanding Architectural Limits](#9-heuristic-wall)
10. [Experiment Reference](#10-experiment-reference)
11. [Key Takeaways for RL Practitioners](#11-takeaways)

---

## 1. The Game: Why Triantica is Interesting for RL {#1-the-game}

### Game Rules in One Minute

```
┌─────────────────────────────────────────┐
│  10×10 toroidal grid (wraps around)     │
│  Players alternate placing triangles   │
│  Each cell: up to 3 triangles           │
│  Rotation matters: 0°, 90°, 180°, 270° │
│  Enclose territory to score            │
│  Largest contiguous area wins           │
└─────────────────────────────────────────┘
```

**Visual example of enclosure:**

```
Before move:          After move (enclosure):
    ┌───┬───┬───┐         ┌───┬───┬───┐
    │ ▲ │   │ ▼ │         │ ▲ │███│ ▼ │
    ├───┼───┼───┤         ├───┼───┼───┤
    │ ◀ │ ? │ ▶ │    →    │ ◀ │▲▼▶│ ▶ │
    ├───┼───┼───┤         ├───┼───┼───┤
    │ ▼ │   │ ▲ │         │ ▼ │███│ ▲ │
    └───┴───┴───┘         └───┴───┴───┘
                        Center enclosed!
```

**Key insight:** The game has **400 possible actions** per turn (100 cells × 4 rotations), but only ~50 are typically valid. The AI must learn:
1. Which moves are legal
2. Which moves create enclosures
3. Which moves build toward future enclosures (strategic planning)

### Why This Game Exposes RL Challenges

```
┌─────────────────────────────────────────────────────────────┐
│                    Game Complexity Comparison                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Game        │ Legal Moves │ Evaluation      │ Difficulty  │
│  ────────────┼─────────────┼─────────────────┼────────────│
│  Chess       │ ~35         │ Piece values     │ Medium      │
│  Go          │ ~250        │ Territory count │ Hard        │
│  Triantica   │ ~50         │ Delayed signal  │ Hard        │
│                                                             │
│  The challenge: A move that looks good now might be       │
│  terrible in 20 turns because it blocked your own         │
│  future enclosure.                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**This is the core RL problem:** Credit assignment. How do you learn which move, 30 turns ago, caused your victory?

### State Representation

The neural network doesn't see the board visually. It sees a vector of numbers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Vector (401 values)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each cell (100 cells × 4 values):                      │
│  ─────────────────────────────                              │
│  [0] Triangle 1 player: +1 if AI, -1 if opponent, 0 if empty│
│  [1] Triangle 1 rotation: 0.0, 0.25, 0.5, or 0.75           │
│  [2] Triangle 2 player: +1 if AI, -1 if opponent, 0 if empty│
│  [3] Triangle 2 rotation: 0.0, 0.25, 0.5, or 0.75           │
│                                                             │
│  Final element [400]: Always 1 (bias signal)               │
│                                                             │
│  Example cell with one AI triangle at 90°:                  │
│  [+1, 0.25, 0, -1, ...]  ← T1 is AI at 90°, T2 empty       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this encoding?** This "fixed perspective" encoding means:
- AI always sees its own pieces as +1
- Opponent's pieces as -1
- The same board position looks the same regardless of which player the AI is assigned

This is critical for generalization (see Bug #1 in Section 8).

---

## 2. RL Fundamentals Through This Project {#2-rl-fundamentals}

### The RL Loop Explained

Reinforcement learning is different from supervised learning. There's no labeled dataset — the agent learns by interacting with the environment.

```
┌─────────────────────────────────────────────────────────────┐
│                    The RL Training Loop                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                      ┌─────────────┐      │
│   │             │   1. Choose action   │             │      │
│   │    Agent    │ ───────────────────► │    Game     │      │
│   │  (Neural    │      (move)         │ (State)     │      │
│   │   Network)  │                     │             │      │
│   │             │ ◄─────────────────── │             │      │
│   └─────────────┘   2. Observe result  └─────────────┘      │
│         │                                                    │
│         │  3. New state + reward                             │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                            │
│   │   Update    │   4. Learn from experience                │
│   │   policy    │      (adjust neural network)               │
│   └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**In Triantica:**
- **Agent**: Neural network that outputs move probabilities
- **Environment**: The game board + rules
- **State**: 401-number vector representing the board
- **Action**: Which triangle to place where (400 options)
- **Reward**: Points scored + bonus for winning

### What is a Policy?

A **policy** (denoted π) is the agent's strategy — it maps states to actions.

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Network                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Layer         Hidden Layer         Output Layer      │
│  (401 neurons)       (200 neurons)        (400 neurons)     │
│                                                             │
│     ○ ○ ○ ○              ○ ○ ○               ○ ○ ○ ○        │
│     ○ ○ ○ ○    ╲╱╲╱      ○ ○ ○     ╲╱╲╱     ○ ○ ○ ○        │
│     ○ ○ ○ ○    ╱╲╱╲      ○ ○ ○     ╱╲╱╲     ○ ○ ○ ○        │
│     ○ ○ ○ ○              ○ ○ ○               ○ ○ ○ ○        │
│       ...                ...                 ...            │
│                                                             │
│  Board state    Learned patterns      Move probabilities   │
│  (401 values)   (intermediate)        (higher = better)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Mathematical form: π(a|s) = P(taking action a | in state s)
```

**Key insight:** The policy network doesn't output a single "best move". It outputs a probability distribution over all possible moves. This allows:
1. **Exploration**: Sometimes picking lower-probability moves to discover new strategies
2. **Training**: We can reinforce good moves (increase their probability) and discourage bad ones

### Policy Gradient: How We Train

The simplest RL algorithm is **policy gradient**. Here's how it works:

```
┌─────────────────────────────────────────────────────────────┐
│                Policy Gradient Training                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Play games                                         │
│  ──────────────────                                         │
│  - Agent plays against itself (self-play)                  │
│  - Record: state, action chosen, reward received           │
│                                                             │
│  Step 2: Label moves                                        │
│  ──────────────────                                         │
│  - If game won: label all moves as "good" (target = +1)    │
│  - If game lost: label all moves as "bad" (target = -1)   │
│  - Adjust by temporal position (early moves less credit)   │
│                                                             │
│  Step 3: Train network                                       │
│  ──────────────────                                         │
│  - Input: state                                             │
│  - Target output: probability of chosen move = reward      │
│  - Minimize difference between predicted and target        │
│                                                             │
│  Effect: Network learns to increase probability of         │
│         moves that led to winning games                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical intuition:**

```
Loss = -log(π(action | state)) × reward

If reward is positive (won): minimize -log(π) → increase π
If reward is negative (lost): minimize -log(π) → decrease π
```

### Three Approaches to RL

```
┌─────────────────────────────────────────────────────────────┐
│                     RL Training Approaches                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Policy Gradient (what we used)                          │
│     ─────────────────────────────────                       │
│     • Play games, label winning moves as "good"            │
│     • Simple but high variance (noisy gradients)           │
│     • Works for basic opponents                            │
│                                                             │
│  2. Actor-Critic (what we need)                             │
│     ─────────────────────────────────────                   │
│     • Two networks: Actor (policy) + Critic (value)        │
│     • Critic: "How good is this position?"                 │
│     • Actor: "What move should I make?"                    │
│     • More stable learning (lower variance)                │
│                                                             │
│  3. AlphaZero (the gold standard)                           │
│     ─────────────────────────────────────                   │
│     • Neural network + Monte Carlo Tree Search (MCTS)     │
│     • MCTS simulates future games for each move            │
│     • Combines search + learning                            │
│     • Computationally expensive                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Our feedforward network uses **policy gradient** — the simplest approach. It works for beating random players, but not strategic opponents. This is the "Heuristic Wall" we hit (see Section 9).

### Exploration vs Exploitation

```
┌─────────────────────────────────────────────────────────────┐
│               Exploration vs Exploitation                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Exploration (ε = 0.3):                                     │
│  ────────────────────                                       │
│  • Try random moves 30% of the time                         │
│  • Discover new strategies                                  │
│  • Risk: Wastes time on bad moves                           │
│                                                             │
│  Exploitation (ε = 0.0):                                    │
│  ─────────────────────                                       │
│  • Always pick highest probability move                     │
│  • Use current knowledge                                     │
│  • Risk: Misses better strategies                           │
│                                                             │
│  Balance:                                                   │
│  ─────────                                                   │
│  • High exploration early (ε = 0.5)                         │
│  • Reduce over time (ε → 0.1)                               │
│  • Zero for evaluation                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The Architecture Journey: brain.js → TensorFlow.js {#3-architecture-journey}

### Phase 1: brain.js (Experiments 1-22)

**Why start simple?** brain.js is a minimal neural network library (~1000 lines) that runs in browsers. Perfect for rapid prototyping and iteration.

```javascript
// brain.js: Simple and intuitive
const net = new brain.NeuralNetwork({ hiddenLayers: [200] });

net.train(trainingData, {
  iterations: 200,
  learningRate: 0.01,
  errorThresh: 0.001
});

// Run inference
const output = net.run(inputState);
```

**What worked:**
- Quick iteration: 8 minutes per 20-cycle training run
- Browser deployment trivial (just load brain.js)
- **91% win rate vs Random** achieved (exp20)
- Good for debugging (simple architecture)

**What didn't work:**
- **0% vs Heuristic** — we hit a wall
- No GPU support (CPU only)
- No custom loss functions
- No convolutional layers for spatial patterns
- Limited to feedforward networks

### Phase 2: TensorFlow.js Migration

```
┌─────────────────────────────────────────────────────────────┐
│                Why TensorFlow.js?                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  brain.js              │     TensorFlow.js                   │
│  ──────────────────────┼────────────────────────────────    │
│  MSE loss only         │  Custom loss functions ✅          │
│  Feedforward only      │  Conv layers, ResNet ✅            │
│  CPU only              │  Metal GPU acceleration ✅          │
│  No value head         │  Multi-head outputs ✅             │
│  ~8 min/train          │  ~35 min/train (100x faster step) │
│  No gradient control   │  Manual gradient computation ✅    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Migration Challenges

The TF.js implementation initially got **49% vs Random** (vs 91% with brain.js). Investigation revealed three critical bugs:

```
┌─────────────────────────────────────────────────────────────┐
│                    Migration Bugs                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 1: Reward Signal Dilution                              │
│  ─────────────────────────────                              │
│                                                             │
│  brain.js (correct):                                        │
│    target = output[chosenMoveIndex]                         │
│    // Only the chosen move gets the reward signal           │
│                                                             │
│  TF.js (buggy):                                             │
│    target = mean(output)  // Average of ALL 400 values!    │
│    // Legal moves at ~0.5, illegal at ~0.01                │
│    // Win signal (0.99) diluted to ~0.28                   │
│                                                             │
│  Result: Network barely learned from wins                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 2: Normalization Mismatch                              │
│  ────────────────────────────                               │
│                                                             │
│  brain.js: Dynamic range based on actual values             │
│  TF.js: Hardcoded min/max that didn't match game            │
│                                                             │
│  Result: Input values in wrong range                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 3: Training Iterations                                  │
│  ─────────────────────                                       │
│                                                             │
│  brain.js: 200 iterations (default)                          │
│  TF.js: 10 iterations (copy-paste error)                     │
│                                                             │
│  Result: Undertrained network                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**How we found them:** Compared TF.js training logs with brain.js logs side-by-side. The win signal magnitude was visibly different.

**Commit reference:** `01e88f0` "CRITICAL FIX: Use chosen move reward instead of average"

---

## 4. Hardware: Mac Studio M2 Ultra Optimization {#4-hardware}

### The Unified Memory Advantage

```
┌─────────────────────────────────────────────────────────────┐
│              Traditional GPU Architecture                    │
│                                                             │
│   CPU ────── PCIe Bus ────── GPU                             │
│   RAM         ~32 GB/s      VRAM                             │
│                    │                                         │
│            (bottleneck!)                                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              Apple M2 Ultra Architecture                     │
│                                                             │
│   CPU ─── Unified Memory ─── GPU                             │
│           400+ GB/s                                          │
│           (192GB shared)                                     │
│                                                             │
│   No copy. No bottleneck. Instant access.                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Practical impact:**
- Training batch size: Entire game history fits in memory (no disk swapping)
- No GPU data transfer: Tensor operations happen instantly
- Per-step speedup: ~100x faster than CPU-only inference

### TensorFlow.js Metal Backend

```javascript
// Enable Metal GPU acceleration
const tf = require('@tensorflow/tfjs-node');

// Automatically uses Metal on Apple Silicon
tf.setBackend('tensorflow');
console.log(tf.getBackend()); // 'tensorflow'
```

**Performance comparison (100 games, 10 cycles):**

| Backend | Time | Speedup |
|---------|------|--------|
| CPU (brain.js) | ~8 min | 1x |
| CPU (TF.js) | ~8 min | 1x |
| Metal GPU (TF.js) | ~35 min | 100x per step* |

*Note: Total time is similar because TF.js does 100x more computation per cycle (batching, gradient computation). But each gradient step is ~100x faster.

### Node.js Version Critical

```
⚠️ CRITICAL: TensorFlow.js native bindings require Node v20 LTS

Node v25 → C++ binding errors → tfjs-node fails to load
Node v20 → Works perfectly

Use: fnm use v20.20.2
Or: nvm use 20.20.2
```

**Why?** TF.js's native C++ bindings are compiled against specific Node ABI versions. v20 is the current LTS and officially supported.

### Batch Size Optimization

For M2 Ultra with 192GB unified memory:

```
Batch Size │  GPU Utilization │  Training Quality
───────────┼──────────────────┼───────────────────
    1      │  ~5%             │  Noisy gradients, poor
   32      │  ~60%            │  Sweet spot for quality
   64      │  ~80%            │  Slightly faster
  128      │  ~95%            │  Fast but may miss patterns
  256      │  ~100%           │  Maximum throughput
```

**Recommendation:** `batchSize: 32` for best learning, `batchSize: 256` for speed.

---

## 5. Parameters Explored {#5-parameters}

### Parameter Overview

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

### Learning Rate (α)

**What it does:** Controls how much the network weights change with each gradient update.

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

## 7. Training Strategies {#7-training-strategies}

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

### Strategy 2: Mixed Opponents

```
┌─────────────────────────────────────────────────────────────┐
│                    Mixed Opponents                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Combine different opponent types:                        │
│   ───────────────────────────────────                       │
│   50% Network (self-play)                                   │
│   30% Random (diversity)                                    │
│   20% Heuristic (strategic pressure)                       │
│                                                             │
│   Theory: Learn from stronger opponents                     │
│   Reality: 0% win rate vs Heuristic = pure loss gradient   │
│                                                             │
│   Result: 43-51% vs Random (slightly better than pure)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiments TF-B2_v3, TF-B2_mixed:** Mixed opponent training
- Final vs Random: 43-51%
- Final vs Heuristic: 0%

**Key finding:** Being crushed by Heuristic provides no useful learning signal. The network learns "avoid this outcome" but not "how to win".

### Strategy 3: Cluster Reward (TD-λ)

```
┌─────────────────────────────────────────────────────────────┐
│                 Cluster Reward (TD-λ)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Traditional: Reward only at game end                     │
│   TD(λ): Spread reward backwards through time               │
│                                                             │
│   Cluster reward factors:                                  │
│   ─────────────────────────                                 │
│   • Enclosure points (immediate)                            │
│   • Connected cluster size (territory control)             │
│   • TD(λ) decay (temporal credit assignment)               │
│                                                             │
│   Parameters:                                                │
│   • clusterRewardFactor: 0.1                               │
│   • clusterLambda: 0.9 (TD decay)                          │
│   • clusterGamma: 0.99 (future discount)                   │
│                                                             │
│   Result: 43-52% vs Random (slight improvement)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiments TF-B3_v3, TF-B3_cluster:** Cluster reward testing
- Final vs Random: 36-52%
- Final vs Heuristic: 0%

**Finding:** Cluster reward helps slightly but doesn't break the Heuristic Wall.

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

## 8. Critical Bugs and How We Found Them {#8-critical-bugs}

### Bug 1: Self-Play Perspective Mismatch (exp19)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Self-play: P1 wins 100% of games                          │
│   vs Random: Only 50-55% win rate                           │
│                                                             │
│   Statistical impossibility in a fair game!                 │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** P2 was using P1's policy to select moves.

```javascript
// WRONG: P2 uses P1's network output
// Both players see the same state, same output
move = getAIMove(network, grid, exploration);  // Always from P1's view

// CORRECT: Each player uses their own perspective
const state = getStateFixed(grid, currentPlayer);  // Fixed perspective
move = getAIMove(network, grid, exploration, aiPlayerNumber);
```

**How we found it:** Self-play win rate histogram showed P1 winning every single game. In a fair game, each side should win ~50%.

**Commit:** `fb9e2e4`

### Bug 2: Reward Signal Dilution (TF.js Migration)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   brain.js: 91% vs Random                                   │
│   TF.js:    49% vs Random                                   │
│                                                             │
│   Same architecture, same data — why the drop?             │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Averaged reward instead of chosen move reward.

```javascript
// WRONG (TF.js bug)
const rawImmediateReward = vector.reduce((sum, v) => sum + v, 0) / 400;
// Average of all 400 values, including ~350 illegal moves at 0.01
// Win signal (0.99) diluted to ~0.28

// CORRECT (original brain.js)
const chosenMoveReward = vector[chosenMoveIndex];
// Only the chosen move gets the reward signal
```

**How we found it:** Compared training logs side-by-side. The win signal magnitude was visibly different: 0.99 vs 0.28.

**Commit:** `01e88f0` "CRITICAL FIX: Use chosen move reward instead of average"

### Bug 3: Heuristic Player Number (TF Baselines)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Heuristic scores 173/173 (max) every game                 │
│   AI scores 3-4 points every game                           │
│                                                             │
│   Why does Heuristic always get perfect score?             │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** HeuristicPlayer hardcoded to `playerNumber=2`.

```javascript
// WRONG
const heuristicP2 = new HeuristicPlayer(2, {...});
// When AI plays as P2, opponent should be P1!
// Both players think they're P2 → chaos

// CORRECT: Create heuristic with correct opponent number
const opponentNumber = (aiPlayerNumber === 1) ? 2 : 1;
const heuristicOpponent = new HeuristicPlayer(opponentNumber, {...});
```

**How we found it:** Audited game scores. Heuristic scoring 173/173 every game was statistically suspicious.

**Commit:** `30384cc`

### Bug 4: Win Rate Measurement (exp12)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Training reports ~70% win rate                           │
│   Evaluation reports ~50%                                 │
│                                                             │
│   Which is right?                                          │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Player swap for fairness, but win counting by slot number.

```javascript
// WRONG
const aiPlayer = (g % 2 === 0) ? 1 : 2;  // AI alternates roles
...
if (winner === 1) wins[1]++;  // Counts slot, not identity

// CORRECT: Track wins by AI identity, not slot number
results.wins = { ai: 0, random: 0, draws: 0 };
if (winner === aiPlayer) results.wins.ai++;
```

**How we found it:** Noticed training and evaluation giving different results for the same opponent.

**Commit:** `b4ffc84`

---

## 9. The Heuristic Wall: Understanding Architectural Limits {#9-heuristic-wall}

### The Experiment That Changed Everything

After 22 brain.js experiments and 10+ TF.js experiments, a clear pattern emerged:

```
┌─────────────────────────────────────────────────────────────┐
│                   Performance Summary                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Opponent      │  Best Win Rate  │  All Experiments         │
│   ──────────────┼─────────────────┼────────────────────────  │
│   Random        │     91%        │      27-91%              │
│   Heuristic     │      0%        │       0%                 │
│                                                             │
│   Conclusion: The architecture cannot beat strategic play   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Can't We Beat Heuristic?

```
┌─────────────────────────────────────────────────────────────┐
│                    Heuristic Strategy                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   For each legal move:                                      │
│   1. Simulate placement                                     │
│   2. Count immediate enclosures (+3 pts each)              │
│   3. Count chain potential (can complete in 2-5 moves?)   │
│   4. Evaluate blocking opponent enclosures                  │
│   5. Score cluster connections                              │
│                                                             │
│   This requires looking 5-10 moves ahead.                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Feedforward Network                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Look at current board state                            │
│   2. Output probability for each move                        │
│   3. Pick highest probability                               │
│                                                             │
│   No lookahead. No planning. Just pattern matching.          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**The fundamental mismatch:** Heuristic does multi-step lookahead. Feedforward networks have no temporal reasoning.

### The Architecture Gap

```
┌─────────────────────────────────────────────────────────────┐
│                 What We Have vs What We Need                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Current: Feedforward Policy Network                       │
│   ─────────────────────────────────────────                 │
│   • 401 inputs → [200 hidden] → 400 outputs                 │
│   • No position evaluation                                  │
│   • No lookahead                                            │
│   • Input: raw board encoding                               │
│   • Output: move probabilities                              │
│                                                             │
│   Needed: Actor-Critic + Search                             │
│   ──────────────────────────────────────                    │
│   • Conv layers for spatial patterns                        │
│   • Value head: "How good is this position?"                │
│   • Policy head: "What moves are promising?"               │
│   • MCTS/beam search: "What happens if I play here?"       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What AlphaZero Does Differently

```
┌─────────────────────────────────────────────────────────────┐
│                    AlphaZero Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Neural Network: Conv → ResNet → {Policy, Value}         │
│                                                             │
│   • Convolutional layers: Recognize spatial patterns        │
│   • Residual blocks: Deep learning without vanishing grads │
│   • Policy head: Probability distribution over moves        │
│   • Value head: Scalar position evaluation                  │
│                                                             │
│   MCTS Search:                                               │
│   1. From current position, simulate many games            │
│   2. Use policy network to guide search                     │
│   3. Use value network to evaluate positions                │
│   4. Return improved policy based on search results         │
│                                                             │
│   Key insight: Search + learning > learning alone           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**We're trying to do with pure learning what AlphaZero does with search+learning.** This is fundamentally impossible for strategic games requiring multi-step lookahead.

### Possible Solutions

```
┌─────────────────────────────────────────────────────────────┐
│                    Next Steps                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Add Convolutional Layers                                │
│      • Replace dense input with 2D convolution              │
│      • Learn spatial patterns (enclosure shapes, etc.)     │
│      • Lower effort, may not be enough                      │
│                                                             │
│   2. Add Value Head (Actor-Critic)                          │
│      • Train network to evaluate positions                   │
│      • Use value as training signal (lower variance)        │
│      • Medium effort, addresses gradient noise              │
│                                                             │
│   3. Add Search (MCTS or Beam Search)                       │
│      • Simulate future game states                           │
│      • Combine search with learned evaluation               │
│      • High effort, guaranteed to work                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Experiment Reference {#10-experiment-reference}

### Complete Results Table

| ID | Name | Framework | Mix | LR | Iter | Batch | Grad Iso | Cluster | vs Random | vs Heuristic | Time |
|----|------|-----------|-----|-----|------|-------|-----------|---------|-----------|--------------|------|
| TF-B1_pure | Pure Self-Play | TF.js | 100% net | 0.05 | 10 | - | ✗ | ✗ | **49%** | 0% | 39m |
| TF-B1_v3 | Pure (v3 fixes) | TF.js | 100% net | 0.05 | 10 | 256 | ✓ | ✗ | 37% | 0% | 40m |
| TF-B1_opt | Optimized | TF.js | 100% net | 0.01 | 100 | 32 | ✓ | ✗ | **47%** | 0% | 35m |
| TF-B2_mixed | Mixed Opponents | TF.js | 50/30/20 | 0.05 | 10 | - | ✗ | ✗ | 44% | 0% | 81m |
| TF-B2_v3 | Mixed (v3 fixes) | TF.js | 50/30/20 | 0.05 | 10 | 256 | ✓ | ✗ | 43% | 0% | 81m |
| TF-B3_cluster | Cluster Reward | TF.js | 70/30 | 0.05 | 10 | - | ✗ | ✓ | **52%** | 0% | 40m |
| TF-B3_v3 | Cluster (v3 fixes) | TF.js | 70/30 | 0.05 | 10 | 256 | ✓ | ✓ | 36% | 0% | 40m |
| TF-ONLINE | Online SGD | TF.js | alt | 0.01 | 10 | - | ✓ | ✗ | 27% | 0% | - |
| exp20 | brain.js Best | brain.js | mixed | 0.01 | 200 | - | ✗ | ✗ | **91%** | 0% | 8m |
| exp22 | Cluster (brain.js) | brain.js | mixed | 0.01 | 200 | - | ✗ | ✓ | 84% | 0% | 8m |
| TF-B5_conv | Phase 2: Conv+Value | TF.js | 100% net | 0.01 | 10 | 256 | ✓ | ✗ | 50% | 0% | 42m |
| TF-B6_optimized | Hyperparameter Tuning | TF.js | 100% net | 0.02 | 10 | 256 | ✓ | ✗ | **50%** | 0% | 42m |

**Key:**
- Mix: opponent distribution (network/random/heuristic)
-
### Parameter Glossary

| Parameter | Symbol | Range | Effect |
|-----------|--------|-------|--------|
| Learning Rate | α | 0.001-0.1 | Gradient step size |
| Momentum | β | 0.0-0.99 | Gradient smoothing |
| Exploration | ε | 0.0-1.0 | Random move probability |
| Iterations | - | 1-500 | Training epochs per cycle |
| Batch Size | - | 1-512 | Samples per gradient update |
| Games | - | 10-500 | Self-play games per cycle |
| Cycles | - | 1-100 | Training iterations |
| Gradient Isolation | - | bool | Enable isolated gradients |
| Cluster Reward | - | bool | Enable TD(λ) rewards |
| Cluster Factor | - | 0.0-1.0 | Cluster bonus multiplier |
| Cluster Lambda | λ | 0.0-1.0 | TD(λ) decay rate |
| Cluster Gamma | γ | 0.0-1.0 | Future reward discount |

---

## 11. Key Takeaways for RL Practitioners {#11-takeaways}

### What We Learned

```
┌─────────────────────────────────────────────────────────────┐
│                    Key Lessons                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Architecture > Hyperparameters                          │
│     ──────────────────────────────                          │
│     No amount of learning rate tuning will make a            │
│     feedforward network play strategically.                  │
│                                                             │
│     UPDATE (Phase 2): Even conv + value head doesn't break  │
│     the strategic barrier. Search is essential.             │
│                                                             │
│  2. Dense rewards help early learning                       │
│     ────────────────────────────────                        │
│     Gradient isolation is theoretically cleaner             │
│     but practically worse. Dense supervision matters.       │
│                                                             │
│  3. Bugs are more common than you think                     │
│     ────────────────────────────────────                    │
│     We found 4 critical bugs in production code.            │
│     Trust your metrics, verify your assumptions.            │
│                                                             │
│  4. Hardware matters for iteration speed                    │
│     ──────────────────────────────────                       │
│     M2 Ultra unified memory enabled 100x faster             │
│     gradient computation, enabling more experiments.        │
│                                                             │
│  5. Some problems need search                                │
│     ────────────────────────────                            │
│     Multi-step lookahead cannot be learned by               │
│     feedforward networks. Search is necessary.              │
│                                                             │
│     CONFIRMED (Phase 2): Conv + value head still hits      │
│     the wall. Architecture alone isn't enough.              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Recommendations for Similar Projects

**Starting Out:**
1. Use simple frameworks first (brain.js) for rapid iteration
2. Implement comprehensive logging before tuning
3. Verify baseline metrics match expected distributions

**During Development:**
1. Keep a training journal (like this one)
2. Compare against previous experiments, not just absolute metrics
3. Use visualization to catch bugs early (win rate histograms, score distributions)

**Architecture Decisions:**
1. Start with feedforward to establish baseline
2. If you hit a wall, evaluate if search is needed
3. Actor-critic helps with variance but may not break strategic walls

**Hardware:**
1. Unified memory (Apple Silicon) simplifies batch size tuning
2. Node.js version compatibility matters for native ML libraries
3. Profile GPU utilization to find batch size sweet spot

### Further Reading

**Reinforcement Learning:**
- [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

**Game AI:**
- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

**TensorFlow.js:**
- [TF.js Documentation](https://www.tensorflow.org/js)
- [tfjs-node GitHub](https://github.com/tensorflow/tfjs-node)

---

## 12. Optimization Roadmap: Breaking the Heuristic Wall {#12-optimization-roadmap}

This chapter outlines a prioritized plan to improve AI performance. Each phase builds on the previous, with clear success metrics and implementation notes specific to TensorFlow.js on Mac Studio M2 Ultra.

### Current Baseline

| Metric | Current | Target |
|--------|---------|--------|
| vs Random | 37-52% | 80%+ |
| vs Heuristic | 0% | 10%+ |
| Training time (20 cycles) | 35-40 min | <20 min |
| Samples per cycle | 100 games | 1000 games |

**Why these targets?** 10% vs Heuristic demonstrates the architecture can learn strategic concepts. 80% vs Random proves stability. Faster training enables rapid iteration.

---

### Phase 1: Experience Replay Buffer

**Priority:** Highest (immediate gain, low complexity)

**Problem:** Current training uses each game once. The network forgets early games by the time it sees later ones.

**Solution:** Store game histories in a circular buffer. Sample random batches for training.

```
┌─────────────────────────────────────────────────────────────┐
│                  Experience Replay Buffer                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Traditional: Play 100 games → Train on all → Discard      │
│                                                             │
│   Replay Buffer: Play 100 games → Add to buffer → Sample   │
│                                                             │
│                 ┌───┬───┬───┬───┬───┐                       │
│   Buffer:       │ G1│ G2│ G3│...│G10K│  (100K samples)      │
│                 └───┴───┴───┴───┴───┘                       │
│                      ↓ Random sample                        │
│                 ┌───┬───┬───┐                              │
│   Batch:         │ G5│G42│ G7│  (diverse examples)          │
│                 └───┴───┴───┘                              │
│                                                             │
│   Benefit: See each position multiple times, from different│
│            game contexts. Stable learning.                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 1.1 | Create `ReplayBuffer` class with circular storage | 30 min |
| 1.2 | Implement `add(samples)` method | 15 min |
| 1.3 | Implement `sample(batchSize)` with priority sampling | 30 min |
| 1.4 | Modify training loop to use buffer instead of immediate training | 45 min |
| 1.5 | Add buffer size parameter to CLI | 10 min |
| 1.6 | Test: Train 5 cycles, verify buffer fills correctly | 20 min |
| 1.7 | Run experiment TF-B4 (replay buffer, 20 cycles) | 40 min |
| 1.8 | Document results in TRAINING_BOOK | 15 min |

**Success Criteria:**
- [ ] Buffer fills to capacity (100K samples) within first 10 cycles
- [ ] Training loss decreases smoothly (no spikes)
- [ ] vs Random improves to 60%+ (from current 50%)
- [ ] No memory leaks (verify with Activity Monitor)

**M2 Ultra Optimization:**
- Set buffer size to 500K-1M samples (64GB unified memory)
- Use `tf.data.Dataset` for efficient streaming
- Store tensors in GPU memory when possible

**Code Location:** `src/server/ReplayBuffer.js`

---

### Phase 2: Canonical Slot Encoding + Convolutional Layers

**Priority:** High (architectural fix for input representation AND spatial reasoning)

**Two problems, one solution:**
1. **Encoding is ambiguous** — rotation is a continuous value (0, 0.25, 0.5, 0.75), implying ordinal relationships that don't exist (90° isn't "between" 0° and 180°)
2. **Architecture flattens the board** — losing spatial relationships that are essential for detecting enclosures

**Current:** `401 inputs → [200 hidden] → 400 outputs` (dense, continuous rotation)

**Proposed:** `10×10×18 one-hot input → [Conv layers] → Policy head + Value head`

#### Step 2A: Canonical 1-to-1 Slot Encoding

**Key principle:** Triantica is Markovian — only the current board state matters, not the order moves were played. Each possible placement is a discrete binary fact: present or absent. Empty slots deserve first-class representation, not implicit absence.

```
┌─────────────────────────────────────────────────────────────┐
│              Current vs Canonical Encoding                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CURRENT (4 values per cell):                               │
│    [player_t1, rotation_t1, player_t2, rotation_t2]         │
│    rotation = 0, 0.25, 0.5, 0.75, or -1                    │
│    ❌ Implies 90° is "between" 0° and 180°                 │
│    ❌ Player and geometry mixed in same channel             │
│    ❌ Empty = -1 rotation (special case, not a real angle)  │
│    ❌ Conv layer can't cleanly detect patterns              │
│                                                             │
│  CANONICAL (18 one-hot channels per cell):                  │
│    Per cell, per half (top/bottom):                         │
│      Channel 0:  "empty"                                   │
│      Channel 1-4: "my piece, rotation 0°/90°/180°/270°"    │
│      Channel 5-8: "opponent, rotation 0°/90°/180°/270°"    │
│                                                             │
│    ✅ Every position has exactly ONE active channel/half    │
│    ✅ Empty is first-class, not implicit absence            │
│    ✅ No spurious ordering between rotations                 │
│    ✅ Conv filters get clean binary inputs                   │
│    ✅ Pure board state — no move order information           │
│                                                             │
│  Total: 10×10×18 = 1,800 input values                       │
│  (vs 401 current — still tiny for 64GB M2 Ultra)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why empty slots need their own channel:**
A conv filter detecting "empty cell next to my enclosure" needs to recognize absence. With the 16-channel encoding (all zeros = empty), the filter must apply negative weights across all 16 channels — a complex pattern. With the 18-channel encoding, the filter simply needs a positive weight on channel 0 (empty) — trivial to learn.

**Why this matters for convolutions:**
A 3×3 conv filter on the current 4-channel input would need to simultaneously decode player identity AND rotation from mixed signals. On the 18-channel input, a filter simply checks "do I see my triangles at these specific rotations in these positions?" — directly detecting enclosure patterns.

#### Step 2B: Convolutional Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Convolutional Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: 10×10×16 tensor (one-hot slot encoding)           │
│                      ↓                                      │
│   Conv2D(32 filters, 3×3, padding='same') + ReLU           │
│                      ↓                                      │
│   Conv2D(64 filters, 3×3, padding='same') + ReLU           │
│                      ↓                                      │
│   Conv2D(64 filters, 3×3, padding='same') + ReLU           │
│                      ↓                                      │
│            ┌────────────┴────────────┐                      │
│            ↓                         ↓                      │
│      Policy Head                 Value Head                 │
│   (400 outputs)                  (1 output)                 │
│   Move probabilities            Position score             │
│                                                             │
│   Key insight: One-hot slots + Conv = pattern detection    │
│   that dense layers on continuous values can never match.  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why convolutions work for Triantica (with proper encoding):**
- Enclosures are local spatial patterns (3-7 cells)
- One-hot channels let filters detect exact rotation combinations
- Territory = contiguous regions → convs naturally detect regions
- No spurious ordinal relationships between rotations

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 2.1 | Implement `getStateCanonical(grid, aiPlayerNumber)` — 18-channel one-hot encoding | 45 min |
| 2.2 | Write roundtrip test: encode → decode → verify board state matches | 30 min |
| 2.3 | Implement `createConvNetwork()` function with 10×10×18 input | 1 hour |
| 2.4 | Add policy head (flatten → dense → softmax) | 30 min |
| 2.5 | Add value head (flatten → dense → tanh) — combining Phase 2 + 3 | 30 min |
| 2.6 | Modify training loop: canonical encoding + conv network | 45 min |
| 2.7 | Test: Single game with conv network, verify output shapes | 20 min |
| 2.8 | Run experiment TF-B5 (canonical + conv, 20 cycles) | 50 min |
| 2.9 | Compare training curves vs dense baseline | 20 min |
| 2.10 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Roundtrip test passes: canonical encoding fully captures board state
- [ ] Network produces valid probability distribution over moves
- [ ] Training time per cycle < 3 minutes
- [ ] vs Random matches dense baseline (50%) by cycle 5
- [ ] vs Random exceeds dense baseline (60%+) by cycle 20
- [ ] vs Heuristic shows first wins (even 1% matters)

**M2 Ultra Optimization:**
- Use `tf.conv2d` with `tf.memory()` monitoring
- Batch size 64-128 (larger batches for conv)
- Enable `tf.webgl.forceHalfFloat()` for faster computation

**Code Location:** `src/server/ConvNetwork.js`

---

### Phase 2 Results (2026-10-25)

**Experiments Completed:** TF-B5_conv, TF-B6_optimized

#### TF-B5: Canonical Conv Network + Value Head

| Metric | Value |
|--------|-------|
| Architecture | 3×Conv2D (32/64/64) + Policy head + Value head |
| Parameters | ~148K |
| Input | 10×10×18 canonical one-hot tensor |
| Training time | 42 min (20 cycles, 100 games/cycle) |
| Best vs Random | 50% (cycle 7) |
| Final vs Random | 36% |
| vs Heuristic | 0% (0 wins in 2000 games) |

#### TF-B6: Hyperparameter Optimization

**Changes from TF-B5:**
- Learning rate: 0.01 → **0.02** (BatchNorm allows higher LR)
- Value loss weight: 0.5 → **1.0** (equal signal for value learning)
- Reward target range: [0.01, 0.99] → **[0.0, 1.0]** (stronger gradients)

| Metric | Value |
|--------|-------|
| Training time | 42.5 min (20 cycles) |
| Best vs Random | **50%** (cycle 17) |
| Final vs Random | 33% |
| vs Heuristic | **0%** (0 wins in 2000 games) |

#### Key Findings

```
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: No Breakthrough                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Architecture upgrade successful:                        │
│     • Canonical encoding removes ambiguity                  │
│     • Conv layers can detect spatial patterns               │
│     • Value head provides dense learning signal             │
│                                                             │
│  2. Performance unchanged:                                  │
│     • vs Random: 50% (same as dense baseline)               │
│     • vs Heuristic: 0% (wall still stands)                   │
│                                                             │
│  3. Hyperparameter tuning had no effect:                    │
│     • Higher LR didn't improve learning                     │
│     • Equal value weight didn't help                        │
│     • Full gradient range didn't matter                     │
│                                                             │
│  4. Root cause confirmed:                                   │
│     • The problem is NOT input representation               │
│     • The problem is NOT network capacity                    │
│     • The problem is NOT learning rate/weighting             │
│     • The problem IS: no lookahead/search                   │
│                                                             │
│  Conclusion: Architecture cannot compensate for lack of    │
│              strategic lookahead. MCTS or beam search is    │
│              required to break the Heuristic Wall.         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Comparison: Dense vs Conv

| Metric | Dense (TF-B1) | Conv (TF-B5) | Conv+Opt (TF-B6) |
|--------|---------------|--------------|------------------|
| Parameters | ~160K | ~148K | ~148K |
| Input dim | 401 | 1800 | 1800 |
| Best vs Random | 52% | 50% | 50% |
| vs Heuristic | 0% | 0% | 0% |
| Training time | 40m | 42m | 42m |

**Verdict:** Conv architecture adds complexity without benefit. The bottleneck is strategic reasoning, not pattern detection.

---

### Phase 3: Value Head (Actor-Critic) — MERGED INTO PHASE 2

**Status:** ✅ Merged — Value head added as step 2.5 in the combined Phase 2 plan. Canonical slot encoding + conv layers + value head are being implemented together since they're deeply interdependent. A value head on top of the wrong encoding would give misleading signal; the canonical encoding makes the value head's job tractable.

**Original Problem:** Network only learns "what moves are good" (policy). It doesn't know "how good is my position" (value).

**Current:** Pure policy gradient — only learns from game outcomes.

**Proposed:** Actor-Critic — learn both policy and value simultaneously.

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

**Success Criteria:**
- [ ] Value predictions correlate with outcomes (r > 0.5)
- [ ] Training converges faster (same loss in fewer iterations)
- [ ] vs Random improves to 65%+ (from Phase 2's 60%)
- [ ] vs Heuristic shows first wins (even 1% matters)

**M2 Ultra Optimization:**
- Larger batches for value loss (32-64)
- Use `tf.losses.meanSquaredError` with reduction=SUM
- Value loss weight: 0.5 (tune after initial results)

**Code Location:** `src/server/ActorCriticNetwork.js`

---

### Phase 4: Batch Normalization + Deeper Network

**Priority:** Medium (stability for deeper networks)

**Problem:** Deeper networks are harder to train. Gradients vanish or explode.

**Solution:** Batch normalization after each conv layer.

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

**Success Criteria:**
- [ ] Training reaches lower loss than Phase 3
- [ ] No NaN losses (gradient stability)
- [ ] vs Random 70%+ (from Phase 3's 65%)
- [ ] vs Heuristic 3-5% (from Phase 3's 1%)

**M2 Ultra Optimization:**
- BatchNorm works better with larger batches (64+)
- Use `training: true` in BatchNorm during training
- Lower learning rate (0.001 → 0.0005) for stability

**Code Location:** `src/server/ResidualBlock.js`

---

### Phase 5: Monte Carlo Tree Search (MCTS)

**Priority:** High (required to break the Heuristic Wall)

**Problem:** Network has no mechanism for "what if" reasoning. It sees the current position and outputs move probabilities, but cannot simulate consequences.

**Why this matters:** The Heuristic player evaluates 5-10 moves ahead through simulation. Our network sees only the current board. This is a fundamental asymmetry that no architecture can overcome without search.

---

#### MCTS Theory: A Complete Explanation

Monte Carlo Tree Search combines tree search with neural network evaluation. It's what made AlphaGo Zero superhuman.

**The Four Phases of MCTS:**

```
┌─────────────────────────────────────────────────────────────┐
│                 MCTS Search Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                                           │
│   │   SELECT    │  Traverse tree from root to leaf          │
│   │     ↓       │  using UCT formula                        │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │   EXPAND    │  Add child nodes at leaf                  │
│   │     ↓       │  Initialize with NN priors                │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │  EVALUATE   │  Use value network to score leaf          │
│   │     ↓       │  v = tanh output (-1 to +1)               │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ BACKPROP    │  Update Q and N up to root               │
│   │             │  Q += v, N += 1                          │
│   └─────────────┘                                           │
│                                                             │
│   Repeat N times, then return improved policy              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**The UCT Formula (Upper Confidence Bound for Trees):**

```
UCT(action) = Q(action)/N(action) + c × P(action) × √(N(parent))/(1+N(action))
              ↑                              ↑
              │                              └── Policy prior from NN
              └── Average value from simulations

Where:
- Q = cumulative value from backpropagation
- N = visit count
- P = policy prior from neural network
- c = exploration constant (typically 1.0-2.0)
```

**Why UCT works:** The first term exploits good moves (high Q/N). The second term explores moves with high prior (P) but few visits (low N). This balances exploitation vs exploration.

---

#### MCTS Integration with Our Architecture

Our Phase 2 architecture is perfectly suited for MCTS:

| Component | Role in MCTS |
|-----------|-------------|
| Canonical encoding (10×10×18) | Fast state serialization for tree nodes |
| Policy head (400 sigmoid) | Move priors (P) for UCT formula |
| Value head (1 tanh) | Position evaluation (v) at leaves |
| BatchNorm layers | Stable inference during search |

**Data Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                 MCTS + Neural Network Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Current board state (canonical tensor)              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Neural Network Forward Pass (ONE call per move)      │   │
│  │                                                       │   │
│  │  Input: [10, 10, 18] canonical tensor                │   │
│  │     ↓                                                 │   │
│  │  Conv layers (shared backbone)                       │   │
│  │     ↓                                                 │   │
│  │  ┌──────┴──────┐                                     │   │
│  │  ↓             ↓                                      │   │
│  │  Policy: 400   Value: 1                               │   │
│  │  sigmoid       tanh [-1, +1]                          │   │
│  │  P(move)       Position score                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  MCTS Search (N simulations):                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  For each simulation:                                 │   │
│  │                                                       │   │
│  │  1. SELECT: Traverse tree using UCT                   │   │
│  │     UCT = Q/N + c × P × √(N_parent)/(1+N)           │   │
│  │            ↑   ↑                                      │   │
│  │            │   └── Policy prior from NN              │   │
│  │            └── Average value from backprop            │   │
│  │                                                       │   │
│  │  2. EXPAND: Add child nodes with NN priors           │   │
│  │                                                       │   │
│  │  3. EVALUATE: At leaf → use value head               │   │
│  │     v = value_network(state)                         │   │
│  │                                                       │   │
│  │  4. BACKPROP: Update Q and N up the tree             │   │
│  │                                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Improved policy π(a) = N(child)/N(root)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Simulation Count Trade-off

More simulations = better play, but slower:

| Sims | Time/move | Use Case |
|------|-----------|----------|
| 50 | ~2s | Training (fast iteration) |
| 200 | ~8s | Evaluation (quality check) |
| 1000 | ~40s | Final model (best performance) |

**Recommendation:** Start with 50 sims for training, evaluate at 200.

---

#### M2 Ultra Optimization: Batched Evaluation

The key insight: neural network calls are expensive, tree operations are cheap.

**Naive MCTS (slow):**
```javascript
for (let i = 0; i < simulations; i++) {
  const leaf = select(root);
  const value = evaluate(leaf);  // One NN call per sim
  backpropagate(leaf, value);
}
```

**Batched MCTS (fast):**
```javascript
async function parallelSearch(root, simulations, batchSize = 8) {
  const batches = Math.floor(simulations / batchSize);
  
  for (let b = 0; b < batches; b++) {
    // Collect leaves for batch evaluation
    const leaves = [];
    for (let i = 0; i < batchSize; i++) {
      const leaf = select(root);
      applyVirtualLoss(leaf);  // Mark as "in progress"
      leaves.push(leaf);
    }
    
    // ONE batched neural net call for all leaves
    const states = leaves.map(l => encodeCanonical(l.grid));
    const { policies, values } = await network.batchPredict(states);
    
    // Process all results
    for (let i = 0; i < batchSize; i++) {
      revertVirtualLoss(leaves[i]);
      expand(leaves[i], policies[i]);
      backpropagate(leaves[i], values[i]);
    }
  }
}
```

**Expected speedup:** 4-8x faster with batch size 8-16.

---

#### Virtual Loss for Parallel MCTS

When multiple threads explore the same node, we need to prevent them from all selecting the same path. Virtual loss solves this:

```javascript
function applyVirtualLoss(node) {
  node.Q -= 1;  // Temporarily reduce value
  node.N += 1;  // Temporarily increase visits
}

function revertVirtualLoss(node) {
  node.Q += 1;  // Restore value
  node.N -= 1;  // Restore visits
}
```

This makes a node look "worse" during parallel search, encouraging other threads to explore elsewhere.

---

#### Dirichlet Noise (Exploration)

At the root node, we add noise to policy priors to encourage exploration:

```javascript
function addDirichletNoise(priors, alpha = 0.03, epsilon = 0.25) {
  // Sample noise from Dirichlet distribution
  const noise = sampleDirichlet(alpha, priors.length);
  
  // Mix with priors
  return priors.map((p, i) => 
    (1 - epsilon) * p + epsilon * noise[i]
  );
}
```

**Why this matters:** Without noise, MCTS collapses to one line of play. With noise, it explores diverse strategies.

---

#### Training Pipeline Change

The key innovation from AlphaGo Zero: train the network to predict MCTS output.

```
Current (NN only):
  Self-play games → Store positions → Train network → Repeat

With MCTS (AlphaGo Zero style):
  Self-play games → MCTS improves policy → Store (state, π_MCTS, outcome) → Train network → Repeat
                    ↑                              ↑
                    └── Stronger supervision ──────┘
```

**What changes:**
- Input: Current board state (same)
- Target policy: π_MCTS (MCTS visit distribution) instead of raw network output
- Target value: Game outcome (same)

The network learns: "What would MCTS do?" not "What do I think?"

---

#### Expected Performance Impact

Based on AlphaGo Zero results and our current baseline:

| Metric | Current (NN only) | With MCTS-50 | With MCTS-200 |
|--------|-------------------|-------------|---------------|
| vs Random | 50% | 70-75% | 75-80% |
| vs Heuristic | 0% | **5-10%** | **10-15%** |
| Time/move | 50ms | 2-3s | 8-10s |

**The breakthrough:** MCTS provides the lookahead that the network cannot learn. The Heuristic player wins by planning enclosures 10-20 moves ahead. MCTS gives our network the same capability.

---

#### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Slow training (MCTS per move) | Batch evaluation, start with 50 sims |
| Bug in tree logic (memory leak) | Unit tests, tree visualization, N tracking |
| Overfitting to MCTS policy | Dirichlet noise, temperature schedule |
| Still doesn't beat Heuristic | Increase simulations, tune c parameter |
| Implementation complexity | Start simple (sequential), optimize later |

---

```
┌─────────────────────────────────────────────────────────────┐
│                    MCTS Integration                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Selection: Start from root, pick promising moves        │
│      • Use policy network to guide selection                 │
│      • Use UCT formula: Q + c × P × √N/(1+n)                │
│                                                             │
│   2. Expansion: Add new node when reaching leaf             │
│      • Use policy network to rank moves                     │
│                                                             │
│   3. Evaluation: At leaf, use value network                 │
│      • v = value_network(position)                         │
│                                                             │
│   4. Backpropagation: Update statistics up tree             │
│      • Q(node) += v                                         │
│      • N(node) += 1                                        │
│                                                             │
│   5. Return: After N simulations, return improved policy   │
│      • π(a) = N(child) / N(root)                            │
│                                                             │
│   Key insight: MCTS + NN > NN alone                        │
│   The tree does the planning, NN does the evaluation.       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 5.1 | Implement `MCTSNode` class with parent/children pointers, Q, N, P fields | 1 hour |
| 5.2 | Implement `select()` with UCT formula and c parameter | 45 min |
| 5.3 | Implement `expand()` with policy network priors | 45 min |
| 5.4 | Implement `evaluate()` with value network | 30 min |
| 5.5 | Implement `backpropagate()` | 20 min |
| 5.6 | Implement `search(root, simulations)` main loop | 30 min |
| 5.7 | Add Dirichlet noise at root node | 20 min |
| 5.8 | Add temperature schedule for move selection | 15 min |
| 5.9 | Test: 100 simulations from start position, verify tree builds | 30 min |
| 5.10 | Implement batched evaluation for M2 Ultra optimization | 2 hours |
| 5.11 | Add virtual loss for parallel search | 1 hour |
| 5.12 | Integrate MCTS into self-play training loop | 1 hour |
| 5.13 | Run experiment TF-B8 (MCTS 50 sims, 10 cycles) | 2 hours |
| 5.14 | Run experiment TF-B9 (MCTS 200 sims, 10 cycles) | 4 hours |
| 5.15 | Analyze: compare MCTS vs no-MCTS performance | 30 min |
| 5.16 | Document results in TRAINING_BOOK | 30 min |
| **Total** | | **~16 hours** |

**Success Criteria:**
- [ ] MCTS completes 50 simulations in < 3 seconds
- [ ] MCTS improves policy vs raw network (KL divergence > 0.1)
- [ ] vs Random 70%+ (from current 50%)
- [ ] vs Heuristic **5%+** (from current 0%) — first wins!
- [ ] No memory leaks during extended search

**M2 Ultra Optimization:**
- Batch evaluate leaves (collect all leaves, evaluate in one forward pass)
- Use Web Workers for parallel simulations
- Virtual loss for parallel MCTS (add −1 to Q during search)

**Code Location:** `src/server/MCTS.js`

---

#### MCTS Implementation Details

**File Structure:**
```
src/server/MCTS.js
├── class MCTSNode {
│   ├── constructor(grid, move, parent, prior)
│   ├── select(c = 1.0)           // UCT selection
│   ├── expand(policy)           // Add children
│   ├── update(value)            // Backprop
│   └── isLeaf()                 // Check for expansion
├── class MCTS {
│   ├── constructor(network, config)
│   ├── search(grid, simulations) // Main entry
│   ├── selectPath(node)          // Selection phase
│   ├── evaluateBatch(nodes)      // Batch NN evaluation
│   └── getPolicy(node, temp)     // Visit count policy
└── exports: { MCTS, MCTSNode }
```

**MCTSNode API:**
```javascript
class MCTSNode {
  constructor(grid, move, parent, prior) {
    this.grid = grid;           // GridManager state
    this.move = move;           // {x, y, rotation} that led here
    this.parent = parent;
    this.children = new Map();  // move -> MCTSNode
    this.P = prior;             // Policy prior from NN
    this.Q = 0;                 // Average value
    this.N = 0;                 // Visit count
    this.player = grid.currentPlayer;
  }

  // UCT selection with exploration constant c
  select(c = 1.5) {
    let bestScore = -Infinity;
    let bestChild = null;
    
    for (const [move, child] of this.children) {
      const uct = child.Q + c * child.P * Math.sqrt(this.N) / (1 + child.N);
      if (uct > bestScore) {
        bestScore = uct;
        bestChild = child;
      }
    }
    return bestChild;
  }

  // Expand with policy vector from NN
  expand(policy) {
    const validMoves = this.grid.getValidMoves();
    for (const move of validMoves) {
      const idx = (move.y * 10 + move.x) * 4 + move.rotation;
      const prior = policy[idx];
      const newGrid = this.grid.clone();
      newGrid.placeTriangle(move.x, move.y, move.rotation, this.player);
      this.children.set(JSON.stringify(move), new MCTSNode(newGrid, move, this, prior));
    }
  }

  // Backpropagation
  update(value) {
    this.N += 1;
    this.Q += (value - this.Q) / this.N;  // Running average
  }
}
```

**MCTS Class API:**
```javascript
class MCTS {
  constructor(network, config = {}) {
    this.network = network;  // TFConvNetwork instance
    this.c = config.c || 1.5;  // Exploration constant
    this.alpha = config.alpha || 0.03;  // Dirichlet noise
    this.epsilon = config.epsilon || 0.25;  // Noise weight
  }

  async search(grid, simulations = 50) {
    const root = new MCTSNode(grid, null, null, 1.0);
    
    // Initial evaluation
    const { policy, value } = await this.network.run(
      encodeCanonical(grid, grid.currentPlayer)
    );
    
    // Add Dirichlet noise at root
    const noisyPolicy = this.addNoise(policy);
    root.expand(noisyPolicy);
    
    // Run simulations
    for (let i = 0; i < simulations; i++) {
      const path = this.selectPath(root);
      const leaf = path[path.length - 1];
      const { value } = await this.network.run(
        encodeCanonical(leaf.grid, leaf.player)
      );
      this.backpropagate(path, value);
    }
    
    return this.getPolicy(root);
  }

  getPolicy(node, temperature = 1.0) {
    const visits = [];
    for (const [move, child] of node.children) {
      visits.push({ move, count: child.N });
    }
    
    if (temperature === 0) {
      // Argmax
      return visits.sort((a, b) => b.count - a.count)[0].move;
    }
    
    // Softmax with temperature
    const probs = visits.map(v => Math.pow(v.count, 1/temperature));
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map(p => p / sum);
    
    // Sample
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < visits.length; i++) {
      acc += normalized[i];
      if (r < acc) return visits[i].move;
    }
    return visits[visits.length - 1].move;
  }
}
```

**Configuration Parameters:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| c (exploration) | 1.5 | 0.5-3.0 | Higher = more exploration |
| alpha (Dirichlet) | 0.03 | 0.01-0.1 | Higher = more noise at root |
| epsilon (noise weight) | 0.25 | 0.1-0.5 | Higher = more exploration |
| temperature | 1.0 | 0.0-2.0 | Higher = more random move selection |
| batchSize | 8 | 1-32 | Parallel leaf evaluation |

**Temperature Schedule:**
- First 10 moves: temperature = 1.0 (exploratory)
- Later moves: temperature = 0.0 (greedy)

This prevents the AI from playing the same opening every game.

---

### Phase 6: Self-Play Tournament + Iterative Retraining

**Priority:** Lower (polish after core improvements)

**Problem:** Network may overfit to its own play style.

**Solution:** Maintain pool of historical versions. Train against diverse opponents.

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Tournament Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Self-play: Current net vs current net                 │
│      → Generate games, add to replay buffer                  │
│                                                             │
│   2. Evaluation: Current net vs past versions                │
│      → If win rate < 55%, demote current                     │
│      → If win rate > 70%, promote candidate                  │
│                                                             │
│   3. Diversity injection: Add random/heuristic games        │
│      → Prevent style collapse                               │
│                                                             │
│   4. Retrain: Sample from buffer, train batch               │
│      → Repeat N times                                       │
│                                                             │
│   Benefit: Robust AI that handles varied opponents          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 6.1 | Implement `ModelPool` class to manage network versions | 1 hour |
| 6.2 | Implement tournament evaluation loop | 45 min |
| 6.3 | Add candidate promotion logic | 30 min |
| 6.4 | Add diversity injection (5-10% random games) | 20 min |
| 6.5 | Run experiment TF-B10 (tournament, 20 cycles) | 2 hours |
| 6.6 | Analyze: win rate vs past versions | 30 min |
| 6.7 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Current network beats all past versions (55%+)
- [ ] vs Random 80%+ (from Phase 5's 75%)
- [ ] vs Heuristic 15%+ (from Phase 5's 10%)
- [ ] No catastrophic forgetting (past skills retained)

**Code Location:** `src/server/ModelPool.js`

---

### Progress Tracking

Use this table to track implementation progress. Update after each experiment.

| Phase | Experiment | Status | vs Random | vs Heuristic | Notes |
|-------|------------|--------|-----------|--------------|-------|
| Baseline | TF-B1_pure | ✅ Done | 49% | 0% | Pure self-play |
| Baseline | TF-B2_mixed | ✅ Done | 44% | 0% | Mixed opponents |
| Baseline | TF-B3_cluster | ✅ Done | 52% | 0% | TD(λ) rewards |
| 1: Replay | TF-B4 | ❌ No gain | 45% | 0% | Replay alone doesn't help; architecture limit |
| 2: Conv+Slot+Val | TF-B5 | ✅ Done | 50% | 0% | Canonical 18ch + Conv + Value head |
| 3: Actor-Critic | TF-B6 | ✅ Done | 50% | 0% | Optimized hyperparams (LR=0.02, value weight=1.0) |
| 4: Deep | TF-B7 | ⏳ Pending | — | — | — |
| 5: MCTS-50 | TF-B8 | ⏳ Pending | — | — | — |
| 5: MCTS-200 | TF-B9 | ⏳ Pending | — | — | — |
| 6: Tournament | TF-B10 | ⏳ Pending | — | — | — |

**Status Legend:**
- ✅ Done — Experiment complete, documented
- 🔄 Running — Currently executing
- ⏳ Pending — Not started
- ❌ Failed — Did not meet success criteria (requires revision)

---

### Summary: Expected Impact

| Phase | Change | Expected Δvs Random | Expected Δvs Heuristic | Complexity |
|-------|--------|---------------------|------------------------|------------|
| 1 | Replay buffer | +0% (45→45%) | +0% (0→0%) | Low |
| 2 | Conv layers | +5% (45→50%) | +0% (0→0%) | Medium |
| 3 | Value head | +0% (50→50%) | +0% (0→0%) | Medium |
| 4 | BatchNorm + Deep | +5% (50→55%) | +5% (0→5%) | Medium |
| 5 | MCTS | +20% (55→75%) | +10% (5→15%) | High |
| 6 | Tournament | +5% (75→80%) | +5% (15→20%) | Medium |

**Total Expected:** 80% vs Random, 20% vs Heuristic

**Key Insight:** Architecture changes (conv + value head) matter more than training tricks (replay buffer), but cannot break the Heuristic Wall. MCTS adds strategic lookahead — essential for multi-step planning.

---

## Appendix A: Command Reference

```bash
# Basic training (pure self-play)
node src/server/train_tfjs.cjs --cycles 20 --games 100

# Mixed opponents
node src/server/train_tfjs.cjs --cycles 20 --games 100 \
  --opponent-mix network:0.5,random:0.3,heuristic:0.2

# With cluster reward
node src/server/train_tfjs.cjs --cycles 20 --games 100 \
  --enable-cluster-reward --cluster-reward-factor 0.1

# Optimized for M2 Ultra
node src/server/train_tfjs.cjs --cycles 10 --games 100 \
  --learning-rate 0.01 --iterations 100 --batch-size 32

# Continue from checkpoint
node src/server/train_tfjs.cjs --cycles 10 \
  --start-from docs/data/exp20_network.json
```

### Evaluation Commands

```bash
# Evaluate against random
node src/server/evaluate.cjs --network docs/data/TF-B1_pure_network.json \
  --opponent random --games 100

# Evaluate against heuristic
node src/server/evaluate.cjs --network docs/data/TF-B1_pure_network.json \
  --opponent heuristic --games 100

# Full tournament
node src/server/evaluate.cjs --network docs/data/exp20_network.json \
  --tournament --games 50
```

---

## Appendix B: Project Structure

```
triantica/
├── src/
│   ├── core/
│   │   ├── GridManager.js      # Game logic
│   │   └── GameState.js       # State management
│   ├── players/
│   │   ├── AIPlayer.js        # Brain.js interface
│   │   ├── HeuristicPlayer.js # Rule-based AI
│   │   └── RandomPlayer.js    # Baseline
│   └── server/
│       ├── train_tfjs.cjs     # TF.js training
│       ├── train_brain.cjs    # Brain.js training (legacy)
│       ├── evaluate.cjs      # Evaluation harness
│       └── GameAuditor.js     # Bug detection
├── docs/
│   ├── data/                  # Training outputs
│   ├── TRAINING_BOOK.md       # This guide
│   ├── dashboard/             # Visualization
│   └── experiments.json       # Metadata
└── browser/
    └── index.html             # Game interface
```

---

## Appendix C: Dashboard Parameters

The training dashboard (`docs/index.html`) shows these parameters for each experiment:

| Dashboard Field | Description |
|-----------------|-------------|
| cycles | Number of training cycles |
| games | Games per cycle |
| evalGames | Evaluation games per cycle |
| iterations | Training iterations per cycle |
| learningRate | Learning rate (α) |
| exploration | Exploration rate (ε) |
| selfPlayMode | Self-play opponent type |
| opponentMix | Distribution of opponent types |
| trainingP2 | Detailed opponent breakdown |
| batchSize | Batch size for gradient updates |
| gradientIsolation | Whether gradient isolation is enabled |
| enableClusterReward | Whether TD(λ) rewards are enabled |
| clusterRewardFactor | Cluster bonus multiplier |
| clusterLambda | TD(λ) decay rate |
| clusterGamma | Future reward discount |
| framework | ML framework (tensorflow.js / brain.js) |
| backend | Compute backend (Metal GPU / CPU) |
| audit | Whether game auditing is enabled |
| auditRate | Fraction of games audited |

---

*Last updated: 2026-10-24*
*Version: 3.0 (Local Experiments)*
*Framework: TensorFlow.js with Metal GPU*
*Hardware: Mac Studio M2 Ultra*
