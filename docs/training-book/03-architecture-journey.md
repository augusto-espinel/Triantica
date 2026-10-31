> **[📚 Training Book Index](./README.md)** | [← Chapter 2: RL Fundamentals](./02-rl-fundamentals.md) | [Chapter 4: Hardware & Matrix Sizing →](./04-hardware-m2-ultra.md)

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

To break past the limitations of a CPU-only feedforward network, we migrated the training pipeline to **TensorFlow.js (`@tensorflow/tfjs-node`)**. This unlocked custom masked cross-entropy loss functions, 2D convolutional layers (`Conv2D`), multi-head Actor-Critic outputs, manual gradient control (`tf.variableGrads`), and hardware-accelerated matrix math on the Mac Studio M2 Ultra:

| Capability | `brain.js` (Phase 1) | `TensorFlow.js` (`tfjs-node`, Phases 2–5) |
| :--- | :--- | :--- |
| **Loss Functions** | Unmasked 400-output MSE only | Custom Legal-Move Masked Cross-Entropy + Value MSE |
| **Network Topology** | 1-layer Feedforward MLP (`401 → 200 → 400`) | Shared $3\times 3$ Conv2D Trunk + Dual Policy/Value Heads + `ToroidalPad2D` |
| **Hardware Acceleration** | Single-threaded CPU JavaScript | Native C++ / Apple Accelerate / Metal unified memory execution |
| **Gradient Control** | Black-box `.train()` loop | Explicit `tf.tidy()` / `tf.variableGrads()` with custom mini-batching |

![The 5-Stage Architecture Climb — Breaking the Heuristic Wall](./assets/diagrams/architecture-climb.svg)

<details>
<summary>📐 Diagram: Why TensorFlow.js? (click to expand ASCII)</summary>

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

</details>

### Migration Challenges: Why a Rewrite Initially Performed Worse

When we first ported the training loop (`src/server/train_tfjs.cjs`) and neural network wrapper (`src/ai/TFNeuralNetwork.js`) to TensorFlow.js, the new implementation scored only **`49%` vs Random**—a dramatic regression from `91%` in `brain.js`. Side-by-side log comparison revealed three classic framework migration bugs:

1. **Bug 1 — $400\times$ Reward Signal Dilution:** In `brain.js`, only the target entry for the chosen move was modified while the remaining 399 outputs copied the network's current prediction (zero error). In our first TensorFlow.js port, the loss averaged MSE across all 400 outputs against static baseline targets (`0.5` for legal moves, `0.01` for illegal moves), diluting the chosen move's reward gradient by a factor of $400$.
2. **Bug 2 — Input Normalization Mismatch:** `brain.js` dynamically scaled inputs based on observed range, whereas our initial TensorFlow.js port used hardcoded rotation scaling (`r / 3.0` instead of `r / 4.0`), shifting the input distribution seen by the first dense layer.
3. **Bug 3 — Epoch Under-Training (`10` vs. `200` Iterations):** A configuration typo ran only `10` gradient passes per cycle instead of `brain.js`'s `200` iterations, leaving the network under-fitted on each batch of self-play games.

<details>
<summary>📐 Diagram: Migration Bugs (click to expand ASCII)</summary>

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

</details>

**How we found them:** Compared TF.js training logs with brain.js logs side-by-side. The win signal magnitude was visibly different.

### Forensic Audit: Why `exp20` (`brain.js`, `91%`) Beat Early `TF-B1` (`TF.js`, `49%`)

Even after fixing the initial migration bugs above, a deeper mystery remained: why did `exp20` in `brain.js` reach **`91%` vs Random**, whereas our initial clean `TF-B1` feedforward run in TensorFlow.js plateaued around **`49–53%` vs Random**?

We built a dedicated numerical audit suite (`reward_forensic_audit.cjs` and `reward_signal_audit.cjs`) to trace every gradient update and sample count across both engines:

| Metric / Mechanism | `exp20` (`brain.js` Champion) | `TF-B1` (Early TensorFlow.js Baseline) |
| :--- | :--- | :--- |
| **Games per Cycle** | `100` games (`~8,500` board positions) | `20` games (`~850` board positions) |
| **Experience Replay Buffer** | **Yes** (`+50%` historical mix = `12,750` states) | **No** (trained on current cycle only) |
| **Weight Update Mode** | Online per-sample SGD (`batchSize = 1`) | Mini-batch MSE (`batchSize = 32`) |
| **Weight Updates per Cycle** | **`382,500` updates** | **`5,300` batched updates** |
| **Gradient Isolation** | `false` (full 400-output MSE) | `true` (masked 1-of-400 chosen move) |
| **Win Rate vs. Random / Heuristic** | **`91.0%`** / `0.0%` | **`49.0%`** / `0.0%` |

- **The Gradient Dilution & Isolation Paradox:** Without gradient isolation (full 400-output MSE), `~350` illegal moves are pushed toward `0.01` on every step, generating `98.2%` of the total gradient magnitude while the single chosen move receives only `1.8%`. In `brain.js`, `382,500` online SGD steps combined with a 50% replay buffer brute-forced the network to suppress all 350 illegal moves first—which alone was enough to beat `RandomPlayer` `91%` of the time. Conversely, when `TF-B1` enabled 1-of-400 gradient isolation on a tiny 20-game buffer *without* replay memory, the network overfit the 850 chosen moves to zero error ($e_{\text{end}} \to 0.000000$) while forgetting illegal move boundaries from earlier cycles.

<details>
<summary>📐 Diagram: FORENSIC AUDIT: exp20 (brain.js, 91%) vs TF-B1 (TF.js, 49%) (click to expand ASCII)</summary>

```
┌──────────────────────────────────────────────────────────────────────────────┐
│        FORENSIC AUDIT: exp20 (brain.js, 91%) vs TF-B1 (TF.js, 49%)           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Metric / Mechanism        │ exp20 (brain.js)      │ TF-B1 (Early TF.js)     │
│  ──────────────────────────┼───────────────────────┼─────────────────────────│
│  Games per Cycle           │ 100 games (~8,500 pos)│ 20 games (~850 pos)     │
│  Experience Replay Buffer  │ YES (+50% = 12,750)   │ NO (current cycle only) │
│  Weight Update Mode        │ Online SGD (batch = 1)│ Mini-batch MSE          │
│  Weight Updates / Cycle    │ 382,500 updates       │ 5,300 batched updates   │
│  Gradient Isolation        │ false (full 400 MSE)  │ true (masked 1-of-400)  │
│  vs Random Win Rate        │ 91.0%                 │ 49.0%                   │
│  vs Heuristic Win Rate     │  0.0%                 │  0.0%                   │
│                                                                              │
│  THE GRADIENT DILUTION & ISOLATION PARADOX:                                  │
│  ──────────────────────────────────────────                                  │
│  • Without gradient isolation (full 400-output MSE), ~350 illegal moves are  │
│    pushed toward 0.01 (gradient ≈ 0.12 × 350 = 42.0), while the ONE chosen   │
│    move receives gradient ≈ 0.04 (only 1.8% of the total gradient!).         │
│  • In brain.js, 382,500 online SGD steps + a 50% Replay Buffer brute-forced  │
│    the network to suppress all 350 illegal moves first, which alone was      │
│    enough to beat a Random player 91% of the time!                           │
│  • Conversely, when TF-B1 enabled Gradient Isolation on a tiny 20-game       │
│    buffer with NO replay memory, the network overfit the 850 chosen moves    │
│    to zero error (e_end → 0.000000) while forgetting illegal move boundaries │
│    from prior cycles (catastrophic forgetting).                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

</details>

**Three Enduring Lessons from the Migration Audit:**
1. **Experience Replay is Non-Negotiable:** Training only on the current cycle's games causes the network to oscillate and forget earlier board configurations. Adding the **500,000-state sliding-window Replay Buffer** (`--replay-buffer-size 500000 --replay-ratio 2.0`) stabilized every subsequent TF.js run.
2. **Suppressing Illegal Moves Beat `Random`, Not `Heuristic`:** `exp20`'s `91%` win rate against `RandomPlayer` was largely achieved by learning legal corner-touch boundaries and basic 1-ply captures—not deep territorial strategy. Against `HeuristicPlayer`, both `exp20` and `TF-B1` scored **`0.0%`**.
3. **Core Engineering Principle (Masked Cross-Entropy over Active Decisions):** A policy loss function must neither average MSE across 399 unchosen outputs (which dilutes the chosen move gradient by $400\times$, see Bug #10 in Chapter 7) nor leave unchosen legal moves unconstrained. This realization directly led to our **Legal-Move Masked Softmax Cross-Entropy** ($-\sum_{a \in \text{legal}} \pi_{\text{MCTS}}(a) \log p(a)$) in the convolutional Actor-Critic network (`src/server/ConvNetwork.cjs`).

---

> [← Chapter 2: RL Fundamentals](./02-rl-fundamentals.md) | **[📚 Training Book Index](./README.md)** | [Chapter 4: Hardware & Matrix Sizing →](./04-hardware-m2-ultra.md)
