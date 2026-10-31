> **[📚 Training Book Index](./README.md)** | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | [11B: Milestone Runs (TF-B5 to TF-B13) →](./11b-milestone-runs-tf-b5-to-b13.md)

---

## 11. Deep Dive: Convolutions, Dual Heads, MCTS & Pipeline Consolidation {#11-optimization-roadmap}

This chapter walks through the step-by-step architectural roadmap that took Triantica from the feedforward baseline (`37–52%` vs Random, `0%` vs Heuristic) to **100.0% Full Torus ($T^2$) mastery (`TF-B15`)**. Each phase builds on the previous one, with exact tensor dimensions, success verification checklists, and implementation notes specific to TensorFlow.js on Mac Studio M2 Ultra.

### Initial Pre-ConvNet Baseline vs. Final Achieved State

| Metric | Initial Dense Baseline (`TF-B1`–`B3`) | Original Roadmap Target | **Final Achieved (`TF-B15` Champion)** |
|--------|--------------------------------------|-------------------------|----------------------------------------|
| vs Random | 37–52% | 80%+ | **100.0% (`1,000W / 0L / 0D` across all 10 cycles)** |
| vs Heuristic | 0% | 10%+ | **99.0% Peak (`95.9%` Avg) Raw 1-Ply / 100.0% (`1,000W / 0L`) +Safety** |
| Training time (10 cycles) | ~40 min (Dense) | Fast M2 Ultra batching | **~140 ms / 50-sim MCTS move (Dual-State GPU Batching)** |
| Replay Buffer Strategy | None (1-cycle memory) | Circular Replay Buffer | **45,000-sample FIFO Sliding Window (`~3.5` cycles of Dual-Side MCTS)** |

**Why these milestones mattered:** Breaking 10% vs Heuristic (`TF-B12` hit `63%`) proved the architecture could learn multi-turn tactical concepts; reaching `99.0%` raw 1-ply / `100.0%` with the tactical safety filter (`TF-B15`) proved that combining canonical spatial representations, quiescence tree search, and exact $T^2$ toroidal convolution equivariance solves the Heuristic Wall completely.

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

**Success Criteria (Verified in `TF-B4` & Upgraded to 45K Sliding-Window FIFO in `TF-B14`/`TF-B15`):**
- [x] Buffer fills to capacity (100K–500K samples, refined to a `45,000`-sample AlphaZero sliding window in `TF-B14`/`B15`)
- [x] Training loss decreases smoothly (eliminated the `0.0000` single-cycle overfitting collapse)
- [x] vs Random improves to 60%+ once paired with spatial ConvNet + MCTS (`87%` in `TF-B10`, `100%` in `TF-B12`–`B15`; standalone feedforward replay in `TF-B4` scored `45%`, proving the bottleneck was architectural)
- [x] No memory leaks (verified via `tf.tidy()` / explicit tensor disposal on M2 Ultra)

**M2 Ultra Optimization:**
- Set buffer size to 500K-1M samples (64GB unified memory)
- Use `tf.data.Dataset` for efficient streaming
- Store tensors in GPU memory when possible

**System Component:** Experience Replay Buffer Module (`src/server/ReplayBuffer.cjs`)

---

### Phase 2: Canonical Slot Encoding + Convolutional Layers

**Priority:** High (architectural fix for input representation AND spatial reasoning)

**Two problems, one solution:**
1. **Encoding is ambiguous** — rotation is a continuous value (0, 0.25, 0.5, 0.75), implying ordinal relationships that don't exist (90° isn't "between" 0° and 180°)
2. **Architecture flattens the board** — losing spatial relationships that are essential for detecting enclosures

**Legacy Feedforward Architecture (`brain.js` / `TF-B1`–`B4`):** `401 inputs → [200 hidden] → 400 outputs` (dense, continuous rotation)

**Implemented Phase 2–6 Architecture (`ConvNetwork.cjs`):** `10×10×18 one-hot input → [ToroidalPad2D + 3×3 Conv2D + BatchNorm + ReLU] → Policy head + Value head`

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

#### Step 2B: How Convolutional Filters Scan the Board & Connect to Policy and Value Heads

To understand why a Convolutional Neural Network (CNN) dramatically outperforms a flattened Dense network on spatial games, let's visualize **how a $3 \times 3$ filter slides across the $10 \times 10 \times 18$ board tensor**, how stacked convolutions expand the **receptive field**, and how the shared feature map branches into the **Policy Head** and **Value Head**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     1. HOW A 3×3×18 CONVOLUTION FILTER DETECTS AN ENCLOSURE AT CELL (x,y)    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     10×10 Board (Spatial Grid)             3×3 Receptive Window Around (x,y) │
│   ┌───┬───┬───┬───┬───┬───┬───┐                 ┌───────┬───────┬───────┐    │
│   │ . │ . │ . │ . │ . │ . │ . │                 │(x-1,  │ (x,   │(x+1,  │    │
│   ├───┼───╔═══╦═══╦═══╗───┼───┤                 │ y-1)  │ y-1)◢ │ y-1)  │    │
│   │ . │ . ║   ║ ◢ ║   ║ . │ . │                 ├───────┼───────┼───────┤    │
│   ├───┼───╠═══╬═══╬═══╣───┼───┤   Dot-Product   │(x-1,  │ (x,y) │(x+1,  │    │
│   │ . │ . ║ ◤ ║ ? ║ ◥ ║ . │ . │ ──────────────► │  y) ◤ │ EMPTY │  y) ◥ │    │
│   ├───┼───╠═══╬═══╬═══╣───┼───┤   over 3×3×18   │       │Ch0=1.0│       │    │
│   │ . │ . ║   ║ ◣ ║   ║ . │ . │   (162 weights) ├───────┼───────┼───────┤    │
│   ├───┼───╚═══╩═══╩═══╝───┼───┤                 │(x-1,  │ (x,   │(x+1,  │    │
│   │ . │ . │ . │ . │ . │ . │ . │                 │ y+1)  │ y+1)◣ │ y+1)  │    │
│   └───┴───┴───┴───┴───┴───┴───┘                 └───────┴───────┴───────┘    │
│                                                                              │
│   Filter k computes:  z_k(x,y) = ReLU( BatchNorm( ∑_{dx,dy,c} W_k · X + b )) │
│                                                                              │
│   • Because the SAME 162 weights slide over all 100 cells, a filter that     │
│     learns to recognize a "1-move enclosure trap" at (2,3) INSTANTLY         │
│     recognizes that exact geometric trap at all 100 cells (Translation       │
│     Invariance)! A Dense network must re-learn it 100 separate times.        │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Hierarchical Feature Extraction Across the 3 Conv Layers (Receptive Field Growth)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     2. HIERARCHICAL SPATIAL ABSTRACTION (32 → 64 → 64 FILTERS)               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Tensor       Conv1 (32 filters)   Conv2 (64 filters)   Conv3 (64 flt) │
│  [10, 10, 18]  ───► [10, 10, 32]    ───► [10, 10, 64]    ───► [10, 10, 64]   │
│                                                                              │
│  Receptive Field:   3×3 cells (9 cells)  5×5 cells (25 cells) 7×7 (49 cells) │
│  What it sees:      • Shared diagonals   • Multi-cell wedges  • Enclosed     │
│                     • Corner touches     • 2-move traps         pockets      │
│                     • Empty vs occupied  • Local cluster      • Territory    │
│                       slots                bridges              boundaries   │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### How the Shared Conv Trunk Connects to the Policy & Value Heads

Once `Conv3` produces the `[B, 10, 10, 64]` spatial feature representation (where each cell `(x, y)` holds a 64-dimensional vector summarizing its $7 \times 7$ neighborhood), the network splits into two specialized heads that work together during **MCTS Search** and **Backpropagation**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     3. DUAL-HEAD SYNERGY: POLICY (ACTOR) + VALUE (CRITIC) IN MCTS            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                  Shared Conv3 Feature Map: [B, 10, 10, 64]                   │
│                                      │                                       │
│                    ┌─────────────────┴─────────────────┐                     │
│                    ▼                                   ▼                     │
│     ┌─────────────────────────────┐     ┌─────────────────────────────┐      │
│     │   POLICY HEAD ("The Actor") │     │   VALUE HEAD ("The Critic") │      │
│     ├─────────────────────────────┤     ├─────────────────────────────┤      │
│     │ 1×1 Conv (2 filters)        │     │ 1×1 Conv (1 filter)         │      │
│     │   [B, 10, 10, 64]           │     │   [B, 10, 10, 64]           │      │
│     │   ──► [B, 10, 10, 2]        │     │   ──► [B, 10, 10, 1]        │      │
│     │ Flatten: [B, 200]           │     │ Flatten: [B, 100]           │      │
│     │ Dense(400) + Sigmoid        │     │ Dense(64, ReLU) ──► Dense(1)│      │
│     │ Output P(s,a): [B, 400]     │     │ Output V(s) = tanh: [B, 1]  │      │
│     └──────────────┬──────────────┘     └──────────────┬──────────────┘      │
│                    │                                   │                     │
│                    ▼                                   ▼                     │
│     ROLE IN MCTS TREE SEARCH:           ROLE IN MCTS TREE SEARCH:            │
│     • Provides Prior P(s,a) in UCT:     • Evaluates leaf board state s_L     │
│       UCT = Q/N + c·P(s,a)·√N/(1+n)       without playing random rollouts    │
│     • Prunes 400 actions down to the      to the end of the game!            │
│       top ~5–10 tactical candidates     • Backpropagates V(s_L) ∈ [-1,+1]    │
│       so 50 sims aren't wasted on         up the tree to update Q(s,a).      │
│       terrible moves.                                                        │
│                                                                              │
│     ROLE IN TRAINING PERFORMANCE:       ROLE IN TRAINING PERFORMANCE:        │
│     • Trained against MCTS visit        • Trained against actual game        │
│       distribution π_MCTS(a):             outcome z ∈ {-1, 0, +1}:           │
│       L_policy = -∑_a π_a log(p̂_a)        L_value = (V(s) - z)^2             │
│     • Distills 50-step lookahead        • Forces the shared 32→64→64 Conv    │
│       into instant 1-step intuition!      filters to learn features that     │
│                                           predict LONG-TERM WINNING, not     │
│                                           just greedy 1-move captures!       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why Dual-Head Co-Training Significantly Improves Performance:**
1. **Shared Representation Regularization:** Because both heads backpropagate gradients into the same `Conv1 → Conv2 → Conv3` filters ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$), the convolutional filters must extract spatial features that simultaneously explain *where MCTS wants to play next* ($\pi$) and *who will ultimately win the game* ($z$). This prevents the Policy Head from memorizing superficial local patterns that lose in the endgame.
2. **Mutual Amplification Loop (Policy Iteration):**
   - A sharper **Policy Head** focuses MCTS simulations onto the best 5–10 candidate moves, allowing a 50-simulation budget to search 4–6 plies deep instead of 1–2 plies wide.
   - A more accurate **Value Head** gives MCTS reliable leaf evaluations at depth 4–6, producing higher-quality visit counts $N(s,a)$, which in turn become a stronger training target $\pi_{\text{MCTS}}$ for the Policy Head in the next cycle!

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

**Success Criteria (Verified across `TF-B5` → `TF-B15`):**
- [x] Roundtrip test passes: canonical 18-channel encoding (`encodeBoardState` in `ConvNetwork.cjs`) captures 100% of board state
- [x] Network produces valid probability distribution over legal moves (via masked softmax cross-entropy)
- [x] Training time per cycle < 3 minutes for 1-ply Actor-Critic (`~2.1 min/cycle` in `TF-B5`/`TF-B6`)
- [x] vs Random matches dense baseline (50%) in `TF-B5`/`B6` (`50%`)
- [x] vs Random exceeds dense baseline (60%+) once paired with MCTS (`79%` in `TF-B9`, `87%` in `TF-B10`, **`100%`** in `TF-B12`–`B15`)
- [x] vs Heuristic achieves breakthrough wins once paired with Quiescence MCTS (`63%` in `TF-B12` → `90%` peak in `TF-B14` → **`99%` raw / `100%` +Safety** in `TF-B15`)

**M2 Ultra Optimization:**
- Use `tf.conv2d` with `tf.memory()` monitoring
- Batch size 64-128 (larger batches for conv)
- Enable `tf.webgl.forceHalfFloat()` for faster computation

**System Component:** 2-Head Convolutional Actor-Critic Network (`src/server/ConvNetwork.cjs`)

---

---

> [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | **[📚 Training Book Index](./README.md)** | [11B: Milestone Runs (TF-B5 to TF-B13) →](./11b-milestone-runs-tf-b5-to-b13.md)
