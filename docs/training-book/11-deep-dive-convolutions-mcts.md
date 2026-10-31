> **[📚 Training Book Index](./README.md)** | [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | [Chapter 12: Toroidal 3D Visualization & Human Telemetry →](./12-toroidal-3d-and-telemetry.md)

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

#### TF-B8: MCTS-50 Training (AlphaGo Zero Style)

**Goal:** Test if MCTS with 50 simulations can provide strategic lookahead to break the Heuristic Wall.

**Architecture:** Conv (Phase 2 - TF-B5 proven)

**Parameters:**
- Cycles: 10
- Games/cycle: 100
- MCTS sims: 50 (batched, parallel evaluation)
- MCTS batch: 8
- Learning rate: 0.02
- Batch size: 256
- Exploration: 0.1 (MCTS provides exploration via Dirichlet noise)
- Training target: MCTS visit policy π(a)

**Command:**
```bash
node src/server/train_tfjs.cjs --cycles 10 --games 100 --mcts-sims 50 --architecture conv --output dashboard/data/TF-B8.json
```

**Expected:** vs Random 65-70%, vs Heuristic 3-8%

**Actual:** vs Random **30%**, vs Heuristic **0%**

**Status:** ❌ Failed — Network learned to lose (performance degraded)

**Runtime:** 438.1 minutes (~7.3 hours)

**Key Finding:** The network consistently got worse over training:
- Started: 40% vs Random
- Peak (cycle 2): 42% vs Random
- Final (cycle 10): 30% vs Random

This confirms a **sign flip bug** in the training signal. The network IS learning — just learning the wrong thing. The learning mechanism works; the reward signal is inverted.

**Cycle-by-Cycle Results:**

| Cycle | vs Random | vs Heuristic | Training Loss |
|-------|-----------|--------------|---------------|
| 1 | 40% | 0% | 0.99 → 1.00 |
| 2 | **42%** ⬆️ | 0% | 1.98 → 1.00 |
| 3 | 30% | 0% | 1.88 → 1.00 |
| 4 | 37% | 0% | 1.81 → 1.00 |
| 5 | 31% | 0% | 1.93 → 1.00 |
| 6 | **26%** ⚠️ | 0% | 2.34 → 1.00 |
| 7 | 37% | 0% | 1.79 → 1.00 |
| 8 | 37% | 0% | 2.09 → 1.00 |
| 9 | 32% | 0% | 2.04 → 1.00 |
| 10 | **30%** | 0% | 2.03 → 1.00 |

**Analysis:**
- Win rate fluctuates but trend is **downward**
- Final performance (30%) is **worse than random** (50%)
- Training loss converges well (always reaches ~1.00)
- This is NOT overfitting — loss decreases smoothly
- This is NOT capacity issue — network CAN learn
- **This is inverted signal** — network learned to play badly

**Bug Hypotheses (to debug):**
1. `rewardForPlayer` sign flipped somewhere in self-play
2. Canonical encoding: `myPieces=1, enemy=-1` inverted
3. Value target: game outcome from wrong player's perspective
4. Visit count normalization: high visits = bad move?

**Key Implementation Details:**

1. **Training Pipeline Change:**
   - Policy target: Visit distribution π(a) = N(child)/N(root)
   - Value target: Game outcome (unchanged)
   - Network learns: "What would MCTS do?"

2. **MCTS Parameters:**
   - Exploration constant c = 1.5
   - Dirichlet noise α = 0.03, ε = 0.25
   - Temperature: 1.0 early, 0.0 late game
   - Batch size: 8 (parallel leaf evaluation)

3. **Performance Notes:**
   - ~50x slower than regular self-play
   - 1 game ≈ 50 seconds with MCTS-50
   - Batched evaluation: 2x speedup over sequential

---

#### TF-B8 (Clean Consolidated Re-Run): MCTS-50 with All 5 Engine/Search Bugs Fixed

Before incrementing to `TF-B9`, we re-ran `TF-B8` (`dashboard/data/TF-B8.json`) across all 10 cycles (309.1 minutes) with the consolidated `GridManager` rules (`isValidMove`, `executeMove` with enclosures & extra turns, slot-based `getCanonicalState`, ReplayBuffer value targets, and single-perspective MCTS filtering) plus **P1 vs P2 role win-share tracking**:

| Cycle | vs Random (Overall) | AI as P1 (1st Move) | AI as P2 (2nd Move) | vs Heuristic | Eval AI Mean Cluster | Training Loss (Start → End) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | **42.0%** | 42.0% | 42.0% | 0.0% | 37.0 (med 32.0) | `1.394 → 2.056` |
| **C2** | **41.0%** | 40.0% | 42.0% | 0.0% | 35.7 (med 30.0) | `2.289 → 2.222` |
| **C3** | **33.0%** | 26.0% | 40.0% | 0.0% | 33.6 (med 30.0) | `2.272 → 2.252` |
| **C4** | **49.0%** | 38.0% | **60.0%** | 0.0% | 38.8 (med 33.5) | `2.116 → 2.177` |
| **C5** | **41.0%** | **54.0%** | 28.0% | 0.0% | 35.3 (med 31.5) | `1.764 → 1.976` |
| **C6** | **47.0%** | **52.0%** | 42.0% | 0.0% | 37.7 (med 33.0) | `2.206 → 2.149` |
| **C7** | **42.0%** | 36.0% | 48.0% | 0.0% | 34.6 (med 31.0) | `1.738 → 1.955` |
| **C8** | **43.0%** | 48.0% | 38.0% | 0.0% | **40.4 (med 35.0)** | `2.194 → 2.116` |
| **C9** | **45.0%** | **52.0%** | 38.0% | 0.0% | 36.5 (med 33.0) | `1.923 → 2.017` |
| **C10** | **47.0%** | **48.0%** | **46.0%** | 0.0% | 37.1 (med 34.0) | `2.104 → 2.075` |

**Findings from the Clean `TF-B8` Re-Run:**
- **Cluster sizes increased by +60% to +85%** (`33.6–40.4` vs `18.4–26.5` in `TF-B8-V2`) because MCTS now sees enclosures and extra turns via `grid.executeMove`.
- **Stability restored:** Average win rate rose from `30.0%` (`TF-B8 v1`) and `35.4%` (`TF-B8-V2`) to **`43.0%`**, finishing at **`47.0%`** (`48% P1 / 46% P2`) with zero late-cycle collapses.
- **Discovery of Bug 11 (Loss Dilution & Value Saturation):** Inspecting `TF-B8_network.json` revealed why win rate plateaued near 47%: `.mean()` over `[256, 400]` diluted policy gradients by $400\times$ (leaving policy outputs stuck in `0.450–0.545`), while `momentum(0.05, 0.9)` saturated the `tanh` Value Head at `-1.0`.

---

#### TF-B9: MCTS-50 with Policy Cross-Entropy Sum (`400x` Gradient Fix) & Adam (`0.001`)

**Goal:** Isolate and verify the impact of fixing Bug 11 (`.neg().sum(-1).mean()` policy loss reduction + persistent `tf.train.adam(0.001)` optimizer) on top of the consolidated `TF-B8` engine/MCTS baseline.

**Architecture:** Conv (`10×10×18` → `32→64→64` Conv2D + BatchNorm + Policy & Value Heads)

**Parameters:**
- Cycles: 10
- Games/cycle: 100 (Eval: 100 games vs Random + 100 vs Heuristic, with P1/P2 role tracking)
- MCTS sims: 50 (batched leaf evaluation, `mctsBatchSize = 8`)
- Optimizer & LR: `tf.train.adam(0.001)`
- Batch size: 256 (`replayRatio = 2x`, `500,000` buffer)

**Command:**
```bash
node src/server/train_tfjs.cjs --cycles 10 --games 100 --eval-games 100 --mcts-sims 50 --architecture conv --learning-rate 0.001 --output dashboard/data/TF-B9.json
```

**Live Pre-Run Verification (20 Games, Cycle 1):**
- Policy Head immediately separates top MCTS moves (`0.985`) from bad/illegal moves (`0.000`).
- Value Head stays unsaturated (`+0.2538` on mid-game boards).
- Achieved **56.0% vs Random** (`52.0% as P1`, `60.0% as P2`) in the 20-game live verification test.

**Status:** ✅ **Completed** (`292.6` minutes, 10/10 cycles, **84.0% Peak / 79.0% Final vs Random**, **69.9% 10-Cycle Mean**, published to `dashboard/data/TF-B9.json` and `trainedAI_network.json`).

**Complete 10-Cycle Trajectory (`TF-B9`):**

| Cycle | Initial Error (`Epoch 1`) | Final Error (`Epoch 10`) | vs Random (Overall) | AI as P1 (1st Move) | AI as P2 (2nd Move) | Eval AI Cluster (`Mean / Med / Max`) | vs Heuristic (Cluster Mean) | Self-Play (`P1/P2/D`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | `3.672` | `2.577` | **52.0%** | 58.0% | 46.0% | `41.5 / 35 / 96` | `0.0%` (`4.0`) | `55 / 44 / 1` |
| **C2** | `3.324` ⬇️ | `2.387` | **55.0% ⬆️** | 58.0% | 52.0% ⬆️ | `45.0 / 36 / 111` | `0.0%` (`6.5`) | `49 / 48 / 3` |
| **C3** | `3.205` ⬇️ | `2.375` | **63.0% ⬆️** | 58.0% | 68.0% ⬆️ | `52.7 / 45 / 132` | `0.0%` (`3.5`) | `55 / 45 / 0` |
| **C4** | `3.154` ⬇️ | `2.371` | **64.0% ⬆️** | 60.0% ⬆️ | 68.0% | `55.4 / 47 / 148` | `0.0%` (`2.5`) | `50 / 49 / 1` |
| **C5** | **`3.260` ⬆️** | `2.392` | **74.0% 🚀** | **80.0% 🚀** | 68.0% | `58.4 / 56 / 124` | `0.0%` (**`10.5`**) | `53 / 44 / 3` |
| **C6** | `3.193` ⬇️ | `2.373` | **72.0%** | **80.0%** | 64.0% | `65.7 / 63 / 135` | `0.0%` (`8.5`) | `59 / 41 / 0` |
| **C7** | **`3.249` ⬆️** | `2.375` | **84.0% 🚀** | **88.0% 🚀** | **80.0% 🚀** | `66.2 / 63 / 132` | `0.0%` (`6.5`) | `59 / 41 / 0` |
| **C8** | `3.151` ⬇️ | **`2.357`** | **79.0%** | 84.0% | 74.0% | `66.2 / 64 / 135` | `0.0%` (`8.0`) | `60 / 39 / 1` |
| **C9** | **`2.997` ⬇️** | `2.366` | **77.0%** | 82.0% | 72.0% | **`66.5 / 61 / 138`** | `0.0%` (`9.0`) | **`50 / 50 / 0`** |
| **C10** | `3.095` | `2.367` | **79.0% ✅** | **84.0%** | **74.0%** | `66.2 / 63 / 136` | `0.0%` (`4.5`) | `61 / 37 / 2` |
| **Mean / Peak** | — | **`2.357` (best)** | **`69.9%` (`84.0%` Peak)** | **`73.2%` (`88.0%` Peak)** | **`66.6%` (`80.0%` Peak)** | **`66.5` Peak Mean** | `0.0%` (`10.5` Peak) | `70,846` samples |

**Key Findings from `TF-B9`:**
- **+35.0 Percentage Point Peak Breakthrough over `TF-B8` (`84.0%` vs `49.0%`):** Restoring the $400\times$ policy cross-entropy gradient (`.neg().sum(-1).mean()`) and preventing `tanh` Value Head saturation (`tf.train.adam(0.001)`) transformed the exact same ConvNet + MCTS-50 architecture from a `43.0%` average (`TF-B8`) to a **`69.9%` 10-cycle average**, peaking at **`84.0%`** in Cycle 7 and finishing cleanly at **`79.0%`** in Cycle 10 (`84.0% as P1`, `74.0% as P2`) with **zero late-cycle collapses**.
- **Massive Territory Expansion (`Mean 66.2–66.5` vs `36–40` in `TF-B8` and `18–26` in `TF-B8-V2`):** In Cycles 6–10, the AI consistently builds contiguous territories averaging **`66+` triangles** (median `63–64`, max `135–148`), more than **$3\times$ larger** than the pre-audit `TF-B8-V2` network.
- **Strong Dual-Role Mastery:** By Cycle 7–10, the network dominates Random both as **First Mover P1 (`82%–88%`)** and as **Second Mover P2 (`72%–80%`)**, and in Cycle 9 achieved an exact `50W P1 / 50W P2 / 0D` equilibrium in MCTS self-play.

---

#### Theoretical Deep Dive: Why Initial Error "Sawtooths" Across Self-Play Cycles (Strategic Valleys & Phase Transitions)

A fascinating empirical phenomenon appears in the `TF-B9` training logs above: while the **within-cycle loss** drops smoothly every single cycle (e.g., `3.249 → 2.375` across the 10 epochs of Cycle 7), the **initial error at the start of each new cycle** oscillates in a distinct sawtooth pattern (`3.672 → 3.324 → 3.205 → 3.154 → 3.260 ⬆️ → 3.193 → 3.249 ⬆️ → 3.151`).

Does this oscillation in initial error mean the network is reaching **new valleys in the loss landscape where qualitatively different strategies are learned**? **Yes — and the mathematics of Generalized Policy Iteration explains exactly why.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     THE "SAWTOOTH" LOSS LANDSCAPE OF ALPHAZERO SELF-PLAY (TF-B9)             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Loss                                                                        │
│  3.7 ┼ ╲ C1                                                                  │
│      │  ╲   ╲ C2                                                             │
│  3.2 ┼   ╲   ╲   ╲ C3  ╲ C4    ▲ C5 Spike!       ▲ C7 Spike!                 │
│      │    ╲   ╲   ╲     ╲      │╲  (New P1       │╲  (New Counter-    ╲ C8   │
│  2.8 ┼     ╲   ╲   ╲     ╲     │ ╲  Enclosures)  │ ╲  Tactics)         ╲     │
│      │      ╲   ╲   ╲     ╲    │  ╲    ╲ C6      │  ╲                   ╲    │
│  2.3 ┼───────●───●───●─────●───┘───●────●────────┘───●───────────────────●── │
│              │             │       │                 │                       │
│         [Valley 1: Basic   ]   [Valley 2: P1    ] [Valley 3: Dual-Role ]     │
│         [Contiguous Growth ]   [Enclosure Traps ] [MCTS Counter-Play   ]     │
│         Win Rate: 52% → 64%    Win Rate: 74%      Win Rate: 84% (88%/80%)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Supervised Learning vs. Non-Stationary Self-Play
In standard **supervised learning**, the dataset $\mathcal{D}$ is static. Because the data distribution never changes, the loss surface is fixed: the final error at the end of Epoch 10 (`2.371`) would be the exact initial error at the start of Epoch 11 (`2.371`).

In **MCTS Self-Play (Generalized Policy Iteration)**, the training loop alternates between two coupled operators:
1. **Policy Evaluation & Improvement (Within Cycle $k$):** For 10 epochs, the network parameters $\theta_k$ descend into the local valley of the *current* state distribution $\mathcal{D}_k$, reducing loss from $\sim 3.20 \to \sim 2.37$.
2. **Distribution Shift via Tree Search (Between Cycle $k$ and $k+1$):** In Cycle $k+1$, the newly trained network $\theta_k$ guides **MCTS-50** through 100 fresh self-play games. Because $\theta_k$ now plays stronger moves:
   - Both players steer the board into **previously unreachable state sub-manifolds** $\mathcal{D}_{k+1}$ (contested diagonal wedges, multi-cell bridges, and tighter endgames).
   - Inside those new board states, 50-simulation lookahead discovers **tactical refutations and multi-turn enclosure combos** that contradict the raw 1-ply intuition of $\theta_k$.

Therefore, `initialError` at the start of Cycle $k+1$ is **not** a continuation of Cycle $k$'s loss curve—it is the **Kullback-Leibler / Cross-Entropy "Surprise"** of network $\theta_k$ when confronted with MCTS's 50-step lookahead in newly unlocked board states:
$$\mathcal{L}_{\text{initial}}^{(k+1)} = \mathbb{E}_{s \sim \mathcal{D}_{k+1}} \left[ \underbrace{-\sum_{a=1}^{400} \pi_{\text{MCTS}}^{(k+1)}(a \mid s) \log \hat{p}_{\theta_k}(a \mid s)}_{\text{Tactical Surprise (MCTS found a new move)}} + \underbrace{\left(V_{\theta_k}(s) - z^{(k+1)}\right)^2}_{\text{Outcome Surprise (Who wins flipped)}} \right]$$

##### 2. Empirical Proof: Every `initialError` Spike Preceded a Major Win-Rate Breakout
Look closely at the `TF-B9` table above:
- **Cycles 1 → 4 (Descending into Valley 1):** `initialError` drops monotonically (`3.672 → 3.324 → 3.205 → 3.154`) while the AI learns fundamental contiguous placement. By Cycle 4, it reaches the bottom of Valley 1 and its win rate plateaus (`63.0% → 64.0%`).
- **Cycle 5 (Phase Transition 1 — `initialError` spikes `3.154 → 3.260 ⬆️`):** Armed with the C4 network, MCTS self-play breaks out of Valley 1 and discovers aggressive **P1 opening enclosures**. The old C4 weights are initially "surprised" by these new board states (`3.260`), but after 10 epochs of descending into **Valley 2** (`3.260 → 2.392`), the AI's win rate vs Random leaps by **+10 points to `74.0%`** (**`80.0%` as P1**), and its mean cluster size against the Heuristic opponent jumps $4\times$ (`2.5 → 10.5`)!
- **Cycle 7 (Phase Transition 2 — `initialError` spikes `3.193 → 3.249 ⬆️`):** Once P1 knows how to set enclosure traps (C5–C6), MCTS in Cycle 7 discovers **P2 defensive counter-tactics** that refute naive P1 expansion. Again, `initialError` spikes to `3.249` as the network enters **Valley 3**. Once digested (`3.249 → 2.375`), overall win rate surges by **+12 points to `84.0%`**, with **P1 reaching `88.0%`** and **P2 jumping from `64.0%` to `80.0%`**!

##### 3. Why the Replay Buffer Turns "Hopping Valleys" Into "Widening the Basin"
In naive self-play without an experience replay buffer, moving from Valley 1 $\to$ Valley 2 $\to$ Valley 3 often causes **catastrophic forgetting** (rock-paper-scissors cycling: the network learns Strategy B to beat Strategy A, forgets how to beat Random, and collapses).
- Because `TF-B9` trains every cycle on a **1:2 mixture** of **new self-play states (`~7,050` samples from Valley $k+1$)** and **historical Replay Buffer states (`~14,336` samples uniformly sampled across Valleys $1 \dots k$)**, the optimizer is forbidden from abandoning earlier valleys.
- When the combined loss converges down to a new global low (`2.357` in Cycle 8), it proves the ConvNet has found a **shared parameterization $\theta$ whose basin simultaneously satisfies the geometry of all previously discovered strategies**.

#### TF-B10: Breaking the Heuristic Wall via Mixed-Opponent MCTS Training (`50% Network / 50% Heuristic`)

**Diagnostic Root Cause — Why Pure Self-Play (`TF-B9`) Loses to `HeuristicPlayer`:**
A move-by-move trace between `TF-B9_network.json` (P1) and `HeuristicPlayer` (P2) revealed the exact tactical mechanism behind the Heuristic Wall:
- Because `executeMove` grants extra turns via `Math.floor((1 + claimed) / 2)`, **even a 1-triangle enclosure (`claimed = 1`) awards a full extra turn**.
- `HeuristicPlayer` explicitly evaluates 1-step enclosure claims (`+0.3 × claimed`), contiguous cluster growth (`+0.1 × largestCluster`), and relative lead (`+0.2 × (newLead - currentLead)`). Starting as early as **Move 3**, `HeuristicPlayer` chains 6–8 consecutive `claimed = 1` moves in a single turn sequence (`P2_clust` snowballing `1 → 3 → 5 → 19 → 55` by Move 38 while `P1_clust` remains at `1`), getting `96` moves to the AI's `14` moves across a 110-move game.
- In pure self-play (`TF-B9`), neither side played this hyper-aggressive early 1-triangle enclosure chain, so the network's policy and value heads never encountered (or learned to block/contest) early extra-turn snowballs.

**Configuration (`TF-B10`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001 \
  --opponent-mix mixed --network-ratio 0.5 --heuristic-ratio 0.5 --random-ratio 0.0 \
  --start-from dashboard/data/TF-B9_network.json \
  --output dashboard/data/TF-B10.json > dashboard/data/TF-B10.log 2>&1 < /dev/null &
```
- **Warm-Start:** Initialized from `dashboard/data/TF-B9_network.json` (`79.0%` final / `84.0%` peak vs Random).
- **Opponent Curriculum:** `50%` MCTS-50 Self-Play + `50%` `HeuristicPlayer` (`explorationFactor: 0.1`), exposing the AI to aggressive early enclosure chains while maintaining self-play stability via the 500K Replay Buffer.
- **Status:** ✅ **Completed** (`161.3` minutes, 10/10 cycles, **87.0% Peak & Final vs Random**, **81.8% 10-Cycle Mean**, published to `dashboard/data/TF-B10.json` and `trainedAI_network.json`).

**Complete 10-Cycle Trajectory (`TF-B10`):**

| Cycle | vs Random (Overall) | AI as P1 | AI as P2 | Mean Cluster vs Random | vs Heuristic | AI Cluster vs Heuristic (Mean / Max) | Loss (`Initial → Final`) | Self-Play Wins (P1/P2/D) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **78.0%** (78W/22L/0D) | 76.0% | 80.0% | 70.9 | 0.0% | 9.5 / 10 | `3.158 → 2.185` | 49 / 51 / 0 |
| **2** | **76.0%** (76W/21L/3D) | 74.0% | 78.0% | 66.5 | 0.0% | 6.0 / 7 | `3.015 → 2.196` | 45 / 55 / 0 |
| **3** | **79.0%** (79W/21L/0D) | 80.0% | 78.0% | 72.8 | 0.0% | **13.0 / 20** | `3.190 → 2.211` | 49 / 51 / 0 |
| **4** | **84.0%** (84W/16L/0D) | 80.0% | **88.0%** | 74.9 | 0.0% | 9.5 / 11 | `2.929 → 2.212` | 56 / 44 / 0 |
| **5** | **81.0%** (81W/18L/1D) | 80.0% | 82.0% | 69.5 | 0.0% | 5.5 / 6 | `3.130 → 2.240` | 47 / 52 / 1 |
| **6** | **87.0% 🚀** (87W/12L/1D) | **92.0% 🚀** | 82.0% | 74.3 | 0.0% | 6.5 / 9 | `3.067 → 2.243` | 48 / 51 / 1 |
| **7** | **86.0%** (86W/14L/0D) | 86.0% | 86.0% | 73.7 | 0.0% | 7.0 / 12 | `2.865 → 2.246` | 52 / 47 / 1 |
| **8** | **82.0%** (82W/18L/0D) | 86.0% | 78.0% | **79.2** | 0.0% | 5.0 / 8 | `3.222 → 2.252` | 46 / 54 / 0 |
| **9** | **78.0%** (78W/22L/0D) | 86.0% | 70.0% | 71.2 | 0.0% | 4.5 / 6 | `3.050 → 2.262` | 56 / 44 / 0 |
| **10** | **87.0% ✅** (87W/11L/2D) | **90.0%** | **84.0%** | 76.5 | 0.0% | 4.0 / 7 | `2.938 → 2.249` | 51 / 48 / 1 |

**Key Takeaways from `TF-B10`:**
1. **New All-Time TF.js Record vs Random (`87.0%` Peak & Final, `81.8%` 10-Cycle Mean):**
   - Overall 10-cycle average climbed from `69.9%` (`TF-B9`) to **`81.8%`** (`83.0%` as P1, `80.6%` as P2), peaking at **`92.0%` as P1** (Cycle 6) and **`88.0%` as P2** (Cycle 4).
   - Mean contiguous cluster size against Random reached **`79.2` triangles** in Cycle 8 (`76.5` in Cycle 10).
   - Standalone 1-ply browser network verification (via the standalone game verification script, `verify_game_network.cjs`, without MCTS) improved from `68%` (`TF-B9`) to **`76%` vs Random (`76W / 24L / 0D`)**.
2. **Why `0.0%` vs `HeuristicPlayer` Persists Even With Mixed Opponent Training:**
   - During the 50 Heuristic games per cycle, the deterministic heuristic opponent (`HeuristicPlayer`) chains 1-triangle enclosures (`Math.floor((1 + claimed) / 2) = 1` extra turn per single claim) from Move 3 onward, taking ~85% of the moves (`~14–16` AI turns per game vs `~70` in self-play).
   - An inflection-point audit of `TF-B10` revealed that the standard Monte Carlo Tree Search engine (`MCTS.cjs`) evaluates leaf states purely via the neural Value Head (which rated dangerous hole-creating moves at `+0.4794` vs safe moves at `+0.1060`). Without playing out forced enclosure cascades at the leaves, 50 MCTS simulations actively amplified single-move openings that `HeuristicPlayer` immediately exploited.

#### TF-B12: Quiescence MCTS + Mixed-Opponent Training (`50% Network / 50% Heuristic`)

**Architectural Principle — Let Quiescence Search Teach Both Enclosing Trains and Hole Avoidance:**
Rather than complicating the neural network with auxiliary risk heads or artificial TD($\lambda$) reward shaping, **TF-B12** keeps the clean 2-head ConvNet (`Policy + Value`) and upgrades the MCTS teacher to the **Quiescence MCTS Engine** (`src/server/QuiescenceMCTS.cjs`):
1. **Enclosing Moves Are Never Risky (`claimed > 0` $\implies$ `risk = 0`):** Because any move with `claimed > 0` grants `Math.floor((1 + claimed) / 2) >= 1` extra turns, the active player retains the turn and can immediately chain further enclosures ("enclosing trains") or patch any holes created. `QuiescenceMCTS` gives enclosing moves a positive bonus (`enclosureBonusWeight`) at both the root prior and leaf evaluation.
2. **Penalize Single-Move Openings (`claimed === 0` with Opponent Enclosures):** Only non-enclosing moves that pass the turn to the opponent can be exploited. If a candidate move leaves enclosure opportunities for the opponent, `QuiescenceMCTS` plays out the deterministic greedy cascade at the leaf (`_runQuiescenceRollout`) and penalizes both the leaf Q-value and the root visit policy target (`_getRiskAdjustedPolicy`).
3. **Batched Pre-Cascade Expansion + Post-Cascade Evaluation (`searchBatched` / `_runBatch`):** Leaf nodes are always expanded using the policy of `leaf.grid` (pre-cascade), while leaf values are evaluated on the quiescent post-cascade board in a single combined `batchPredict` GPU call.

**Configuration (`TF-B12`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001 \
  --opponent-mix mixed --network-ratio 0.5 --heuristic-ratio 0.5 --random-ratio 0.0 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.2 \
  --start-from dashboard/data/TF-B10_network.json \
  --output dashboard/data/TF-B12.json > dashboard/data/TF-B12.log 2>&1 < /dev/null &
```

#### TF-B13: Dual-Side Quiescence MCTS + 75/25 Curriculum (`89% Peak / 73% Final vs Heuristic`)

**Architectural & Curriculum Refinements:**
Building directly on `TF-B12_network.json`, **TF-B13** introduced three targeted improvements to consolidate strategic mastery against `HeuristicPlayer`:
1. **Dual-Side MCTS Self-Play Sample Collection (`!step.mctsPolicy` Filter):**
   In `TF-B9` through `TF-B12`, the MCTS sample loop filtered by `if (step.player !== aiPlayerNumber || !step.mctsPolicy) continue;`, discarding half of all MCTS-50 searches in network-vs-network games. Because every state in `hist` is already canonically encoded from `step.player`'s perspective (`1` = mover, `-1` = opponent) and `rewardForPlayer` is computed relative to `step.player`, both P1 and P2 MCTS moves are 100% perspective-consistent. Changing the filter to `if (!step.mctsPolicy) continue;` captured **both sides of network-vs-network MCTS games** while still excluding non-MCTS `HeuristicPlayer` moves—nearly doubling sample generation from **`~6,096` samples/cycle (`60,961` total in `TF-B12`)** to **`~11,589` samples/cycle (`115,892` total in `TF-B13`)**.
2. **75% Network / 25% Heuristic Curriculum + Fine-Tuning Schedule (`LR = 0.0005`, `riskPenalty = 0.25`):**
   Shifted the opponent mix to `75%` dual-side `QuiescenceMCTS-50` self-play and `25%` `HeuristicPlayer`, halved the Adam learning rate (`0.001` $\to$ `0.0005`), and raised `riskPenalty` (`0.20` $\to$ `0.25`) to sharpen punishment of single-move holes while stabilizing high-level self-play.
3. **100-Opening Random 2-Ply Protocol Across All 10 Cycles & Browser Parity:**
   Evaluated all 10 cycles using the 100-game random-first-2-plies protocol (50 as P1, 50 as P2) and updated the browser game controller (`game.js`) so browser AI/Heuristic games also play random first 2 plies (`this.totalMovesPlayed < 2`).

**Configuration (`TF-B13`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.0005 \
  --opponent-mix mixed --network-ratio 0.75 --heuristic-ratio 0.25 --random-ratio 0.0 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.25 \
  --start-from dashboard/data/TF-B12_network.json \
  --output dashboard/data/TF-B13.json > dashboard/data/TF-B13.log 2>&1 < /dev/null &
```

---

### Phase 3: Value Head (Actor-Critic) — MERGED INTO PHASE 2

**Status:** ✅ Merged — Value head added as step 2.5 in the combined Phase 2 plan. Canonical slot encoding + conv layers + value head are being implemented together since they're deeply interdependent. A value head on top of the wrong encoding would give misleading signal; the canonical encoding makes the value head's job tractable.

**Original Problem:** Network only learns "what moves are good" (policy). It doesn't know "how good is my position" (value).

**Legacy Phase 1 Baseline:** Pure policy gradient — only learns from game outcomes.

**Implemented Actor-Critic Architecture (`ConvNetwork.cjs`):** Learn both policy ($\pi_\theta$) and value ($V_\theta \in [-1, +1]$) simultaneously on top of the shared convolutional trunk.

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

**Success Criteria (Verified across `TF-B9` → `TF-B15`):**
- [x] Batched MCTS (`mctsBatchSize = 8`) completes 50 simulations in `~140 ms` (`0.14s`, **20× faster** than the `< 3 seconds` target!)
- [x] MCTS improves policy vs raw network (verified via `test_quiescence.cjs` and cross-entropy distillation)
- [x] vs Random exceeds 70%+ (`79%` in `TF-B9`, `87%` in `TF-B10`, **`100.0%`** across all 4,000 eval games in `TF-B12`–`TF-B15`)
- [x] vs Heuristic breaks 5%+ once extended with Quiescence rollouts (`63%` in `TF-B12` → `89%` peak in `TF-B13` → `90%` peak in `TF-B14` → **`99%` raw / `100%` +Safety** in `TF-B15`)
- [x] No memory leaks during extended 10-hour MCTS self-play runs (`tf.tidy()` + explicit tensor disposal)

**M2 Ultra Optimization:**
- Batch evaluate leaves (collect all leaves, evaluate in one forward pass)
- Use Web Workers for parallel simulations
- Virtual loss for parallel MCTS (add −1 to Q during search)

**System Component:** Monte Carlo Tree Search Engine (`src/server/MCTS.cjs`, extended by the Quiescence MCTS Engine, `src/server/QuiescenceMCTS.cjs`)

---

#### MCTS Implementation Details

**Module Structure (`src/server/MCTS.cjs`):**
```
MCTS Engine (src/server/MCTS.cjs)
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

### Phase 5.5: Quiescence MCTS — Why It Improves Over Simple MCTS

**Status:** ✅ Completed (Quiescence MCTS Engine, `src/server/QuiescenceMCTS.cjs`, and Mathematical Specification, `docs/QUIESCENCE_MCTS.md`) — **Broke the Heuristic Wall in `TF-B12` (`63%` vs Heuristic), Reached `89%` Peak in `TF-B13`, Achieved `90.0%` Peak (`80.1%` Avg) in `TF-B14`, and Culminated in `99.0%` Peak (`95.9%` Avg) Raw / `100.0%` (`1,000W/0L`) +Safety in `TF-B15` (`100%` vs Random across all 40 cycles)**

#### 1. The Horizon Effect & Why Simple MCTS Fails Against Deterministic Cascades
In `TF-B10` (Simple MCTS-50 + 50% Heuristic opponents), the AI reached a record **87% vs Random**, yet remained at **0% vs `HeuristicPlayer`** (averaging only ~14 turns to Heuristic's ~96 turns per game). An inflection-point audit of `TF-B10` exposed two structural blind spots in the standard Monte Carlo Tree Search engine (`MCTS.cjs`):

1. **The Horizon Effect at Unquiet Leaves:**
   Simple MCTS expands a leaf node $s_L$ after a single candidate move and immediately queries the neural Value Head $V_\theta(s_L)$. If $s_L$ is a **"noisy" (tactically unstable) position**—specifically, a single move (`claimed = 0`) that passes the turn to the opponent while leaving a 3-sided enclosure hole—`HeuristicPlayer` will deterministically claim that enclosure with probability $P(\text{capture}) = 1.0$, earn an extra turn via $\lfloor (1 + \text{claimed})/2 \rfloor$, and chain a 5–10 move capture cascade.
2. **Self-Play Value Head Inversion on Single-Move Openings:**
   Because the Value Head was trained primarily on states where local cluster density correlated with winning against non-cascading play, $V_\theta(s_L)$ rated a hole-creating blunder at **`+0.4794`** while rating a safe move at **`+0.1060`**. With 50 simulations spread across ~35 legal moves, simple MCTS never searched deep enough down the forced opponent cascade to see the collapse—instead, UCT **amplified** the `+0.4794` blunder (`13` visits, `Q = +0.344` vs `1–2` visits for safe moves).

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Simple MCTS vs. Quiescence MCTS at a Leaf Node              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SIMPLE MCTS (Blind to Forced Cascades):                                │
│  Root ──(Move A: claimed=0, leaves hole)──► Leaf s_L                    │
│    • Evaluates raw V_θ(s_L) = +0.4794 ("cluster grew, looks great!")    │
│    • Backpropagates +0.4794 → UCT visits Move A 13 times (Q = +0.344)   │
│    • Reality: Heuristic chains 6 enclosures + extra turns → AI loses    │
│                                                                         │
│  QUIESCENCE MCTS (Resolves Tactical Cascades Before Evaluation):        │
│  Root ──(Move A: claimed=0, leaves hole)──► Leaf s_L                    │
│    1. Expands s_L using pre-cascade policy P_θ(s_L)                     │
│    2. Plays out greedy enclosure cascade: s_L → s_1 → s_2 → s_quiet     │
│    3. Evaluates V_θ(s_quiet) - Σ(oppClaimed × riskPenalty) = -0.65      │
│    4. Backpropagates -0.65 → UCT abandons Move A and selects safe move  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2. Game-Mechanic Invariant: Enclosing Trains vs. Single-Move Openings
A foundational rule of Triantica dictates when a move is safe versus exploitable:

$$\text{extraTurns} = \left\lfloor \frac{1 + \text{claimed}}{2} \right\rfloor$$

- **Enclosing Moves (`claimed > 0`) Are NEVER Risky (`risk = 0`):**
  Claiming even a single enclosed triangle (`claimed = 1`) awards $\lfloor 2/2 \rfloor = 1$ full extra turn, meaning **the active player retains the turn (`nextPlayer === mover`)**. Even if placing that triangle creates a new 3-sided pocket elsewhere on the board, the AI immediately moves again—allowing it to either **chain an enclosing train in its own favor** or patch the hole before the opponent ever gets a turn. Therefore, the enclosure risk calculators (`calculateEnclosureRisk` and `calculateChildNodeRisk` in `src/server/QuiescenceMCTS.cjs`) strictly enforce `risk = 0` whenever `claimed > 0 || nextPlayer === mover`.
- **Single-Move Openings (`claimed === 0`) Carry All Opponent Risk:**
  Only when `claimed === 0` (and `extraTurns === 0`) does the turn pass to the opponent. If that single move leaves enclosure opportunities $\mathcal{O}_{\text{opp}}$ for the opponent, each opportunity $o \in \mathcal{O}_{\text{opp}}$ contributes both a direct triangle-loss weight (`0.3` per enclosed triangle) and a turn-snowball weight (`0.5` per bonus turn earned):
  $$\text{risk}(s, a) = \begin{cases} 0 & \text{if } \text{claimed}(s, a) > 0 \;\lor\; \text{nextPlayer} = \text{mover} \\[6pt] \displaystyle\sum_{o \in \mathcal{O}_{\text{opp}}(s')} \Big(0.3 \cdot o.\text{claimed} + 0.5 \cdot \Big\lfloor \frac{1 + o.\text{claimed}}{2} \Big\rfloor\Big) & \text{otherwise} \end{cases}$$

#### 3. Four Mechanisms That Make `QuiescenceMCTS` a Superior Teacher
Recall that during evaluation (`getAIMove`) and browser/mobile play (via the AI player controller, `AIPlayer.js`), the AI uses a **single forward pass of the Policy Head** (no MCTS at test time). MCTS is the **training teacher** that generates target distributions $\pi_{\text{MCTS}}$ and value targets for the 2-head ConvNet. The Quiescence MCTS engine (`src/server/QuiescenceMCTS.cjs`) directly upgrades the teacher in four ways without needing a 3rd network head or artificial TD($\lambda$) reward shaping:

1. **Root Prior Shaping (`_expandRootWithRisk`):**
   At the root node, every legal child move records `child.claimed` and `child.enclosureRisk`.
   - Enclosing moves (`child.claimed > 0`) receive both a multiplicative and additive prior boost, ensuring that even if an early-cycle Policy Head assigns $P_0(a) \approx 0$ to an enclosing move, the additive floor (`+0.05 × claimed`) guarantees the 50 MCTS simulations immediately explore offensive **enclosing trains**:
     $$P'(s_0, a) \propto \begin{cases} P_0(s_0, a)\cdot\big(1 + 1.5\cdot\text{claimed}\big) + 0.05\cdot\text{claimed} & \text{if } \text{claimed} > 0 \\[4pt] P_0(s_0, a)\cdot\max\!\big(0.05,\; 1 - 2\cdot w_{\text{risk}}\cdot\text{risk}\big) & \text{if } \text{claimed} = 0 \;\land\; \text{risk} > 0 \\[4pt] P_0(s_0, a) & \text{otherwise} \end{cases}$$
   - Hole-creating single moves (`child.claimed === 0` and `risk > 0`) have their prior scaled down by $\max(0.05, 1 - 2 \cdot \text{risk} \cdot w_{\text{risk}})$, preventing wasted simulations on obvious blunders.
2. **Concatenated Dual-State GPU Batching for Pre-Cascade Expansion + Post-Cascade Quiescent Value (`_runBatch`):**
   During batched MCTS (`searchBatched`), every non-terminal leaf `leaf` requires two distinct evaluations when a tactical cascade occurs (`qEval.depth > 0`):
   - **Pre-Cascade Expansion (`expandStates`):** `leaf` must be expanded using the neural policy $P_\theta(\text{leaf.grid})$ of the true **pre-cascade** board so child moves and priors match `leaf.grid`.
   - **Post-Cascade Quiescent Value (`quietStates`):** `leaf` must be evaluated by running a deterministic greedy cascade (`_runQuiescenceRollout`, up to `quiescenceDepth = 5`, configurable across `quiescenceMode ∈ {'none', 'greedy', 'heuristic', 'hybrid'}`) to reach a quiet board `qEval.grid`, then combining the perspective-signed neural value $\pm V_\theta(\text{qEval.grid})$ with tactical cascade adjustments (`+claimed × enclosureBonusWeight` for AI claims, `-claimed × riskPenaltyWeight` for opponent claims):
     $$V_{\text{leaf}} = \text{clip}_{[-1, 1]}\!\left(\text{sign}(p_{\text{quiet}}, p_{\text{root}})\cdot V_\theta(s_{\text{quiet}}) + w_{\text{bonus}}\sum \text{claimed}_{\text{AI}} - w_{\text{risk}}\sum \text{claimed}_{\text{opp}}\right)$$
     *(Note: If the quiescence cascade fills the board or leaves zero legal moves, `_getGridTerminalValue(qEval.grid, rootPlayer)` computes the exact terminal cluster winner $\in \{-1, 0, +1\}$ directly without a neural call.)*
   - **Zero-Extra-Roundtrip GPU Packing:** Rather than issuing two separate GPU calls per simulation batch, `_runBatch` concatenates `allStates = expandStates.concat(quietStates)` (up to `8 + 8 = 16` canonical `10×10×18` tensors) into a **single combined `network.batchPredict(allStates)` call**, slicing `policies[0 .. expandStates.length - 1]` for leaf expansion and `values[quietOffset + k]` for quiescent leaf evaluation!
3. **Direct Policy Target Distillation (`_getRiskAdjustedPolicy`):**
   After all 50 simulations finish, the raw root visit distribution $N(s_0, a)/N(s_0)$ is further refined before being returned as the training target $\pi_{\text{MCTS}}$:
   $$\pi_{\text{MCTS}}(a) \propto \begin{cases} \dfrac{N(s_0, a)}{N(s_0)}\big(1 + w_{\text{bonus}}\cdot\text{claimed}\big) + 0.02\cdot\text{claimed} & \text{if } \text{claimed} > 0 \\[8pt] \dfrac{N(s_0, a)}{N(s_0)}\max\!\big(0.02,\; 1 - 2\cdot w_{\text{risk}}\cdot\text{risk}\big) & \text{if } \text{claimed} = 0 \;\land\; \text{risk} > 0 \\[8pt] \dfrac{N(s_0, a)}{N(s_0)} & \text{otherwise} \end{cases}$$
   Because the ConvNet Policy Head trains directly via cross-entropy against $\pi_{\text{MCTS}}$, **the 1-ply Policy Head internalizes both offensive enclosing trains and defensive hole avoidance**—which is verified in the standalone quiescence regression suite (`test_quiescence.cjs`), where `searchBatched()` selects enclosing chains even when initialized with a completely flat (`0.5` uniform) mock network prior.
4. **$O(1)$ Local Neighborhood Enclosure Detection (`countWouldBeEnclosed`):**
   Because `isTriangleEnclosed(cx, cy, rot)` only depends on cell `(cx, cy)` and its 4 orthogonal neighbors, placing a triangle at `(mx, my)` can only enclose `(mx, my)` itself or one of its 4 orthogonal neighbors:
   $$\mathcal{N}_5(x, y) = \Big\{(x, y),\; \big((x+1)\bmod 10, y\big),\; \big((x+9)\bmod 10, y\big),\; \big(x, (y+1)\bmod 10\big),\; \big(x, (y+9)\bmod 10\big)\Big\}$$
   Pushing the candidate triangle onto `grid[my][mx].triangles`, checking just those 5 cells in-place via `mapping = [2, 3, 0, 1]` (the complementary rotation in a 1-triangle cell), and popping the triangle back off avoids cloning the $10 \times 10$ grid (`80×` fewer checks than scanning all 400 slots) and runs **3.2× faster** (~140ms per 50-sim move on M2 Ultra).

#### 4. Experiment `TF-B12` Results: Breaking the Heuristic Wall (`100%` vs Random, `63%` vs Heuristic)
Warm-started from `TF-B10_network.json`, `TF-B12` ran 10 cycles (100 games/cycle, 50% `QuiescenceMCTS-50` self-play / 50% `HeuristicPlayer`, `quiescenceMode = greedy`, `quiescenceDepth = 5`, `riskPenalty = 0.2`, `enclosureBonusWeight = 0.15`, `LR = 0.001` Adam, runtime `270.5 min`).

| Cycle | Loss (`Start → End`) | Training MCTS-50 vs Heuristic (`W/L/D`, `P1`, `P2`, `Med AI–Opp`) | Eval vs Random (`100 games`) | Mean Score vs Random (`AI – Rand`) | Eval vs Heuristic (Dashboard) | Eval vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`) | Eval Mode vs Heuristic |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (`TF-B10`)** | — | `8.0%` (`4W/46L`, `med 4–177`) | `87.0%` | `79.2 – 26.8` | `0.0%` | `0% / 0%` | `7.8 – 175.9` | Both Det & Random 2-Ply (`0%`) |
| **1** | `5.204 → 3.146` | **`32.1%`** (`18W/38L`, P1:`10/27`, P2:`8/29`, `med 69.5–118`) | **100.0%** | `116.7 – 16.0` | `0.0%` | `0% / 0%` | `12.0 – 169.0` | Deterministic (2 trajectories) |
| **2** | `4.827 → 3.116` | **`31.3%`** (`15W/32L/1D`, P1:`8/26`, P2:`7/22`, `med 74.5–109.5`) | **100.0%** | `130.0 – 11.3` | `0.0%` | `0% / 0%` | `45.0 – 141.5` | Deterministic (2 trajectories) |
| **3** | `4.418 → 3.029` | **`36.0%`** (`18W/32L`, P1:`8/22`, P2:`10/28`, `med 76–105.5`) | **100.0%** | `144.2 – 8.8` | `0.0%` | `0% / 0%` | `55.0 – 141.0` | Deterministic (2 trajectories) |
| **4** | `4.378 → 3.013` | **`42.6%`** (`20W/27L`, P1:`10/29`, P2:`10/18`, `med 89–99`) | **100.0%** | `150.3 – 8.1` | `0.0%` | `0% / 0%` | `36.5 – 147.0` | Deterministic (2 trajectories) |
| **5** | `4.149 → 3.002` | **`29.5%`** (`13W/31L`, P1:`5/17`, P2:`8/27`, `med 79–109`) | **100.0%** | `152.2 – 7.7` | `0.0%` | `0% / 0%` | `79.0 – 111.0` | Deterministic (2 trajectories) |
| **6** | `4.033 → 2.961` | **`40.0%`** (`22W/33L`, P1:`10/26`, P2:`12/29`, `med 79–105`) | **100.0%** | `157.6 – 5.8` | **🏆 `50.0%`** | `0% / 100%` | **`91.0 – 77.5`** | Deterministic (`P2 won 147–24!`) |
| **7** | `4.007 → 2.961` | **`35.7%`** (`15W/27L`, P1:`9/22`, P2:`6/20`, `med 79.5–102.5`) | **100.0%** | `162.1 – 5.4` | `0.0%` | `0% / 0%` | `58.0 – 115.5` | Deterministic (2 trajectories) |
| **8** | `3.984 → 2.947` | **`35.8%`** (`19W/34L`, P1:`9/29`, P2:`10/24`, `med 75–108`) | **100.0%** | `163.7 – 5.1` | **🏆 `50.0%`** | **`50% / 50%`** | **`90.3 – 88.4`** | **Random First 2 Plies (100 openings)** *(Det: `50%`)* |
| **9** | `3.892 → 2.951` | **`49.0%`** (`24W/25L`, P1:`12/27`, P2:`12/22`, `med 88–93`) | **100.0%** | `163.1 – 5.1` | **🚀 `62.0%`** | **`68% / 56%`** | **`100.8 – 78.8`** | **Random First 2 Plies (100 openings)** *(Det: `0%`)* |
| **10** | `3.720 → 2.899` | **`42.3%`** (`22W/29L/1D`, P1:`12/26`, P2:`10/26`, `med 79–108`) | **100.0%** | **`166.2 – 4.4`** | **🚀 `63.0%`** | **`66% / 60%`** | **`101.7 – 79.5`** | **Random First 2 Plies (100 openings)** *(Det: `50%`)* |

#### 5. Methodological Discovery: Why Random First 2 Plies Are Essential for `Eval vs Heuristic`
During `TF-B12`, we discovered a critical flaw in how `Eval vs Heuristic` had been measured since `TF-B1`:
- Because both `getAIMove(..., exploration=0)` ( greedy argmax ) and `new HeuristicPlayer(..., { explorationFactor: 0 })` (with deterministic cell-order tie-breaking in `HeuristicPlayer.getMove`) are **100% deterministic**, running 100 evaluation games from an empty board was actually replaying **only 2 unique game trajectories** (50 identical copies of 1 game with AI as P1, and 50 identical copies of 1 game with AI as P2).
- Consequently, `Eval vs Heuristic` could only ever output `0%`, `50%`, or `100%`! For example, in Cycle 9, the network lost its 2 canonical trajectories (`0%` deterministic), even though its raw 1-ply policy was actually winning **62.0% of games (`68%` as P1, `56%` as P2)** across diverse openings!
- **Protocol Upgrade (Cycle 8 Onwards & Permanent Default in the Self-Play Training Pipeline, `train_tfjs.cjs`):**
  Starting at Cycle 8 of `TF-B12` (and now permanently baked into the self-play training pipeline, `src/server/train_tfjs.cjs`, and the 2-ply evaluation harness, `scripts/eval_random_2ply.cjs`), `Eval vs Heuristic` plays **random moves for the first 2 plies (`m < 2`, i.e., 1 random opening triangle per player)** before switching to `exploration = 0` greedy policy vs deterministic `HeuristicPlayer`. This tests the trained network across **100 genuinely distinct opening board configurations** while keeping all mid-game and endgame play strictly deterministic.
- As validation, we also ran the exact same 100-game random-2-ply audit on `TF-B10_network.json` (Cycle 0 baseline): `TF-B10` still scored **`0.0%` (`0W / 100L`, mean cluster `7.8` vs `175.9`)**, proving that `TF-B12`'s jump from **`0.0%` → `50.0%` (C8) → `62.0%` (C9) → `63.0%` (C10)** is a genuine, massive architectural breakthrough driven by `QuiescenceMCTS`.

#### 6. Experiment `TF-B13` Results: Scaling to `89.0%` Peak & `73.0%` Final vs Heuristic (`100%` vs Random)
Warm-started from `TF-B12_network.json`, `TF-B13` ran 10 cycles (100 games/cycle, 75% dual-side `QuiescenceMCTS-50` self-play / 25% `HeuristicPlayer`, `quiescenceMode = greedy`, `quiescenceDepth = 5`, `riskPenalty = 0.25`, `enclosureBonusWeight = 0.15`, `LR = 0.0005` Adam, runtime `340.8 min`, `115,892` total replay buffer samples). All 10 cycles were evaluated across 100 distinct random-first-2-plies openings against both `RandomPlayer` and `HeuristicPlayer`.

| Cycle | Loss (`Start → End`) | Buffer Samples (`Util`) | Training MCTS-50 vs Heuristic (`W/L`, `P1`, `P2`, `Med AI–Opp`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic (`100 games`) | Eval vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B12`)** | `3.720 → 2.899` | `60,961` (`12.2%`) | `42.3%` (`22W/29L/1D`, `med 79–108`) | **100.0%** (`166.2 – 4.4`) | `63.0%` | `66% / 60%` | `101.7 – 79.5` | `3.78 / 1.84` |
| **1** | `4.046 → 2.934` | `12,036` (`2.4%`) | **`75.0%`** (`12W/4L`, P1:`6/7`, P2:`6/9`, `med 127.5–57`) | **100.0%** (`165.2 – 4.4`) | **`52.0%`** (`52W/47L/1D`) | `44% / 60%` | `91.8 – 87.7` | `4.13 / 1.84` |
| **2** | `4.002 → 2.912` | `23,507` (`4.7%`) | **`52.2%`** (`12W/11L`, P1:`4/15`, P2:`8/8`, `med 95–94`) | **100.0%** (`167.8 – 3.7`) | **`61.0%`** (`61W/39L/0D`) | `54% / 68%` | `98.0 – 78.2` | `3.96 / 1.82` |
| **3** | `3.829 → 2.874` | `34,816` (`7.0%`) | **`79.3%`** (`23W/6L`, P1:`10/11`, P2:`13/18`, `med 124–62`) | **100.0%** (`167.7 – 3.9`) | **`66.0%`** (`66W/34L/0D`) | `70% / 62%` | `102.8 – 75.0` | `4.21 / 1.81` |
| **4** | `3.723 → 2.881` | `46,558` (`9.3%`) | **`77.3%`** (`17W/5L`, P1:`9/11`, P2:`8/11`, `med 132.5–59.5`) | **100.0%** (`171.8 – 3.2`) | **`72.0%`** (`72W/28L/0D`) | `74% / 70%` | `107.8 – 72.4` | `4.33 / 1.81` |
| **5** | `3.838 → 2.880` | `57,556` (`11.5%`) | **`71.0%`** (`22W/9L`, P1:`10/14`, P2:`12/17`, `med 113–84`) | **100.0%** (`171.6 – 3.3`) | **`71.0%`** (`71W/29L/0D`) | `78% / 64%` | `108.2 – 69.8` | `4.19 / 1.79` |
| **6** | `3.724 → 2.886` | `69,363` (`13.9%`) | **🔥 `90.9%`** (`20W/2L`, P1:`9/9`, P2:`11/13`, `med 114.5–71.5`) | **100.0%** (`172.0 – 3.2`) | **`75.0%`** (`75W/25L/0D`) | `74% / 76%` | `117.8 – 62.3` | `4.07 / 1.82` |
| **7** | `3.723 → 2.883` | `81,090` (`16.2%`) | **`73.7%`** (`14W/5L`, P1:`7/10`, P2:`7/9`, `med 109–71`) | **100.0%** (`172.6 – 3.4`) | **`79.0%`** (`79W/19L/2D`) | `78% / 80%` | `117.9 – 61.6` | `3.94 / 1.82` |
| **8** | `3.777 → 2.908` | `93,218` (`18.6%`) | **`81.3%`** (`13W/3L`, P1:`10/10`, P2:`3/6`, `med 122.5–69`) | **100.0%** (`173.6 – 3.0`) | **🏆 `89.0%`** (`89W/11L/0D`) | **`90% / 88%`** | **`126.7 – 52.6`** | `4.51 / 1.84` |
| **9** | `3.718 → 2.894` | `104,119` (`20.8%`) | **`62.5%`** (`20W/12L`, P1:`9/18`, P2:`11/14`, `med 107–84`) | **100.0%** (**`173.9 – 3.0`**) | **`78.0%`** (`78W/22L/0D`) | `86% / 70%` | `121.9 – 57.3` | `4.27 / 1.81` |
| **10** | `3.645 → 2.905` | `115,892` (`23.2%`) | **`58.8%`** (`10W/7L`, P1:`7/11`, P2:`3/6`, `med 99–80`) | **100.0%** (`172.9 – 3.5`) | **🚀 `73.0%`** (`73W/27L/0D`) | **`80% / 66%`** | **`114.1 – 65.8`** | `4.63 / 1.83` |

**Key Takeaways from `TF-B13`:**
- **Sustained Dominance Over `HeuristicPlayer`:** The standalone 1-ply ConvNet (`exploration = 0`, no MCTS at eval time) climbed steadily from `52.0%` (Cycle 1) to an all-time peak of **`89.0%` (`89W / 11L`, `90%` as P1, `88%` as P2, mean cluster `126.7` vs `52.6`)** in Cycle 8, finishing at **`73.0%` (`80%` as P1, `66%` as P2)** in Cycle 10 and averaging **`71.6%`** across all 10 cycles.
- **MCTS-50 Self-Play vs Heuristic:** Across all 227 training games played against `HeuristicPlayer`, `QuiescenceMCTS-50` won **`71.8%` (`163W / 64L`)**, peaking at **`90.9%` (`20W / 2L`)** in Cycle 6.
- **Flawless vs Random:** Achieved **`100.0%` (`1,000W / 0L / 0D`)** across all 10 cycles (`2,000` consecutive wins across `TF-B12` + `TF-B13`), with mean AI cluster size reaching **`173.9` vs `3.0`** in Cycle 9.
- **Dual-Side MCTS Efficiency:** Collecting both players' MCTS policy targets in network-vs-network games nearly doubled the replay buffer growth (`115,892` samples vs `60,961` in `TF-B12`) while maintaining balanced 50/50 self-play parity (`503` P1 wins vs `494` P2 wins, `3` draws across 1,000 games).

#### 7. Post-`TF-B13` Forensic Audit: Architectural Ceiling, Remaining Bottlenecks & Toroidal Topology

While `TF-B13` reached **`89.0%` vs Heuristic** in Cycle 8 (using a raw 1-ply ConvNet forward pass with `exploration = 0`), Cycles 9 and 10 dipped slightly to `78.0%` and `73.0%`. A forensic audit of the training dynamics, network padding, and game rules identified **five key bottlenecks** that explain both the `89%` peak and the path to `95%–99%+` mastery—as well as how to generalize Triantica to **true 3D toroidal and arbitrary manifold surfaces**:

1. **Bottleneck 1 — Replay Buffer Staleness ("Ghost of Cycle 1"):**
   - `TF-B13` used an unbounded `500,000`-capacity replay buffer (`115,892` samples accumulated by Cycle 10 at `~11.6K` samples/cycle).
   - Because early samples were never evicted, by Cycles 9–10 roughly **`70%` of training mini-batches were drawn from Cycles 1–6** (when the policy was weaker: `52%–75%` vs Heuristic). Training 10 epochs at a constant `LR = 0.0005` over stale trajectories pulled Cycle 8's converged weights (`loss = 2.883`) back toward earlier MCTS visit distributions (`loss = 2.908` in C8, `2.905` in C10).
   - **Next Step:** Implement a **sliding-window replay buffer** (`35,000–50,000` samples = last 3–4 cycles, AlphaZero style) + **cosine/step LR decay** (`0.0005 → 0.0001`) + automatic **`best_network.json` checkpointing** whenever `eval.vsHeuristic` sets a new high-water mark.

2. **Bottleneck 2 — Edge Asymmetry in Corner Placement (`hasCorner`) vs. Toroidal Enclosures & Clusters:**
   - In the core board & rules engine (`src/core/GridManager.js` and `game.js`), **edge adjacency** (`getAdjacentTriangles`) and **enclosure detection** (`isTriangleEnclosed`) wrap around the `10×10` grid toroidally via `(x ± 1 + cols) % cols` and `(y ± 1 + rows) % rows`.
   - However, **corner placement legality** (`hasCorner(x, y)` + `getTriangleCorners(x, y, rotation)`) uses **unwrapped planar vertex coordinates** `cx ∈ [0..cols]`, `cy ∈ [0..rows]` and checks `if (cellX >= 0 && cellY >= 0 && cellX < this.cols && cellY < this.rows)`.
   - **Consequence:** A triangle placed on border `x = 0` places corners on vertex line `cx = 0`, which does **not** wrap to `cx = cols` (`10`). Thus, players **cannot** branch legal placements across the outer boundary based on corner contact alone, even though enclosures and cluster merges **do** wrap across the boundary!
   - **Topological Fix (True Torus $T^2$):** In a true $10 \times 10$ torus, vertex `(cols, y)` is identical to `(0, y)` and `(x, rows)` is identical to `(x, 0)` (there are exactly `rows × cols = 100` unique vertices, not `11 × 11 = 121`). Wrapping vertex coordinates modulo `(cols, rows)` inside `hasCorner`:
     ```js
     hasCorner(x, y) {
       const vx = ((x % this.cols) + this.cols) % this.cols;
       const vy = ((y % this.rows) + this.rows) % this.rows;
       for (let dy = -1; dy <= 0; dy++) {
         for (let dx = -1; dx <= 0; dx++) {
           const cellX = (vx + dx + this.cols) % this.cols;
           const cellY = (vy + dy + this.rows) % this.rows;
           const cell = this.grid[cellY][cellX];
           if (cell.triangles.some(t =>
             this.getTriangleCorners(cellX, cellY, t.rotation).some(([tx, ty]) =>
               (tx % this.cols) === vx && (ty % this.rows) === vy
             )
           )) {
             return true;
           }
         }
       }
       return false;
     }
     ```
     eliminates the artificial planar boundary completely, making every cell topologically identical—essential for a **3D rotatable torus UI** where there is no "edge" of the board.

3. **Bottleneck 3 — Zero-Padding (`padding: 'same'`) vs. Circular Toroidal Convolutions:**
   - `TFConvNetwork` currently applies `padding: 'same'` (zero-padding) in its three `3×3` `Conv2D` layers (`receptive field = 7×7`).
   - Because zero-padding fills out-of-bounds neighbors with `0` instead of wrapping to the opposite side of the board, the ConvNet sees artificial walls at `x = 0, 9` and `y = 0, 9`, splitting wrap-around enclosures across opposite sides of the tensor.
   - **Next Step:** Once `hasCorner` is wrapped toroidally, replace zero-padding with **Toroidal Circular Padding** (wrapping 1 row/col around each border before each `3×3` conv, or circularly padding the input tensor), giving the `[32, 64, 64]` filters exact translation invariance across the entire torus.

4. **Bottleneck 4 — 1-Ply Raw Policy Inference vs. Inference-Time Hole-Risk Filtering:**
   - Evaluation and browser play currently select moves using the raw 1-ply policy head (`predictMove()`, `exploration = 0`) with zero search lookahead.
   - Applying `QuiescenceMCTS`'s $O(1)$ local neighborhood risk detector (`countWouldBeEnclosed`) as a lightweight **inference-time safety filter** on the top-$K$ policy candidates (or running light `QuiescenceMCTS-25` at play time) eliminates residual 1-ply hole blunders immediately without changing network weights.

5. **Bottleneck 5 — 8-Fold Dihedral Symmetry ($D_4$) & Future 3D Manifold Surfaces (Graph / Mesh Generalization):**
   - On the $10 \times 10$ square torus, exploiting the 8 symmetries of $D_4$ (4 rotations × 2 reflections, permuting triangle rotations `[0, 1, 2, 3]`) yields an **8× sample efficiency multiplier** (and full $10 \times 10 \times 8 = 800$-fold symmetry once toroidal translation is enabled).
   - **Future Vision — Arbitrary 3D Surfaces:** When Triantica expands beyond a regular $10 \times 10$ torus to **arbitrary closed 3D triangulated meshes** (e.g., spheres, higher-genus surfaces, icosahedral geodesic grids, or irregular 3D manifolds rotated interactively by the player), a planar 2D grid `(x, y)` no longer applies. Instead, the board is naturally represented as a **Dual Face-Adjacency Graph $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$**:
     - Each triangle slot is a node $v \in V_T$ (with state channels: empty, my piece, opponent piece).
     - **Enclosure & Cluster Contiguity** operate strictly over the **3 edge-neighbors** $\mathcal{N}_{\text{edge}}(v)$ of each triangle (`isTriangleEnclosed` = all 3 edge-neighbors occupied; clusters = connected components over $\mathcal{N}_{\text{edge}}$).
     - **Placement Legality** operates strictly over the **3 vertices** $\mathcal{V}(v)$ of each triangle (`isValidPlacement` = at least one vertex in $\mathcal{V}(v)$ is already touched by an existing triangle).
     - **Neural Architecture:** On any 3D triangle mesh, the `3×3` grid `Conv2D` generalizes directly to a **Mesh / Graph Convolutional Network (GCN / Gauge-Equivariant MeshCNN)** aggregating features over $\mathcal{N}_{\text{edge}}(v)$ and $\mathcal{N}_{\text{vertex}}(v)$, while `QuiescenceMCTS` remains **100% unchanged** because it only queries legal moves and enclosure cascades!

#### 8. Experiment `TF-B14` Results: Bottleneck 1 Fixed + 100% Pure Network Self-Play (`90.0%` Peak / `80.1%` 10-Cycle Avg vs Heuristic)

Warm-started from `TF-B13_network.json`, **`TF-B14`** tested the resolution of **Bottleneck 1** (`45,000`-sample sliding-window `ReplayBuffer` + Cosine LR decay `0.0004 → 0.0001` with in-place Adam moment retention + automatic `*_best_network.json` peak checkpointing) under **100% pure network self-play** (`0%` Heuristic teacher, `Dual-Side QuiescenceMCTS-50`, runtime `397.2 min`, `129,962` total samples generated across 10 cycles):

| Cycle | Cosine LR | Buffer (`45K` Window, `Total Added`) | Loss (`Start → End`) | Self-Play (`P1 / P2 / D`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic (`100 games`) | Role Split vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`, `Margin`) | Win/Loss Anatomy (`Blowout ≥+50 / CloseW / D / CloseL / CascadeL <-30`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B13` C10)** | `0.000500` | `115,892 / 500K` | `3.645 → 2.905` | — | **100.0%** (`172.9 – 3.5`) | `73.0%` (`73W/27L/0D`) | `80% / 66%` | `114.1 – 65.8` (`+48.3`) | — | `4.63 / 1.83` |
| **1** | `0.000400` | `12,973 / 45K` (`12,973`) | `3.933 → 2.829` | `54 / 46 / 0` | **100.0%** (`171.3 – 3.5`) | **`83.0%`** (`83W/17L/0D`) | `82% / 84%` | `120.4 – 58.3` (`+62.0`) | `58 / 25 / 0 / 8 / 9` | `4.06 / 1.86` |
| **2** | `0.000391` | `25,939 / 45K` (`25,939`) | `3.710 → 2.823` | `49 / 51 / 0` | **100.0%** (`171.7 – 3.5`) | **`76.0%`** (`76W/24L/0D`) | `86% / 66%` | `113.5 – 65.8` (`+47.7`) | `53 / 23 / 0 / 9 / 15` | `4.59 / 1.87` |
| **3** | `0.000365` | `38,972 / 45K` (`38,972`) | `3.743 → 2.863` | `53 / 47 / 0` | **100.0%** (`173.5 – 2.9`) | **`80.0%`** (`80W/20L/0D`) | `82% / 78%` | `117.4 – 63.3` (`+54.1`) | `61 / 19 / 0 / 7 / 13` | `4.67 / 1.88` |
| **4** | `0.000325` | **`45,000 / 45K`** (`52,062`) | `3.820 → 2.901` | `47 / 52 / 1` | **100.0%** (`172.7 – 3.2`) | **`79.0%`** (`79W/21L/0D`) | `80% / 78%` | `113.8 – 65.2` (`+48.6`) | `50 / 29 / 0 / 5 / 16` | `4.40 / 1.88` |
| **5** | `0.000276` | **`45,000 / 45K`** (`65,123`) | `3.817 → 2.963` | `43 / 56 / 1` | **100.0%** (`173.5 – 3.2`) | **`82.0%`** (`82W/18L/0D`) | `84% / 80%` | `122.2 – 59.0` (`+63.2`) | `63 / 19 / 0 / 7 / 11` | `4.38 / 1.87` |
| **6** | `0.000224` | **`45,000 / 45K`** (`78,036`) | `3.800 → 2.970` | `46 / 53 / 1` | **100.0%** (`173.2 – 2.9`) | **`76.0%`** (`76W/24L/0D`) | `78% / 74%` | `116.0 – 63.8` (`+52.3`) | `53 / 23 / 0 / 12 / 12` | `4.33 / 1.87` |
| **7** | `0.000175` | **`45,000 / 45K`** (`90,998`) | `3.886 → 2.977` | `67 / 33 / 0` | **100.0%** (`175.0 – 2.8`) | **`78.0%`** (`78W/22L/0D`) | `80% / 76%` | `118.7 – 60.6` (`+58.1`) | `62 / 16 / 0 / 8 / 14` | `3.99 / 1.86` |
| **8** | `0.000135` | **`45,000 / 45K`** (`104,004`) | `3.804 → 3.017` | `44 / 56 / 0` | **100.0%** (`174.2 – 3.0`) | **`79.0%`** (`79W/20L/1D`) | `82% / 76%` | `114.8 – 63.2` (`+51.7`) | `57 / 22 / 1 / 7 / 13` | `4.02 / 1.87` |
| **9** | `0.000109` | **`45,000 / 45K`** (`117,017`) | `3.916 → 3.081` | `41 / 59 / 0` | **100.0%** (`174.6 – 2.9`) | **🏆 `90.0%`** (`90W/9L/1D`) | **`88% / 92%`** | **`123.8 – 54.2` (`+69.7`)** | **`62 / 28 / 1 / 2 / 7`** | `4.43 / 1.87` |
| **10** | `0.000100` | **`45,000 / 45K`** (`129,962`) | `3.860 → 3.089` | `48 / 51 / 1` | **100.0%** (**`175.3 – 2.6`**) | **`78.0%`** (`78W/21L/1D`) | `78% / 78%` | `118.4 – 62.0` (`+56.5`) | `51 / 27 / 1 / 9 / 12` | **`5.03 / 1.85`** |

**Key Takeaways from `TF-B14` (Official 2D Baseline Champion — `v1.0.0-2d`):**
1. **New All-Time Record Peak (`90.0%` vs Heuristic in Cycle 9):**
   - In Cycle 9 (`LR = 0.000109`), the 1-ply ConvNet achieved **`90.0%` (`90W / 9L / 1D`, `88.0%` as P1 and `92.0%` as P2)** with an all-time record mean cluster score of **`123.8` vs `54.2` (`+69.7` margin)**, reducing close positional losses to just `2` games and cascade blunders to `7` games out of `100`.
   - Automatic peak checkpointing captured this exact model in **`dashboard/data/TF-B14_best_network.json`** (`MD5: f2eb302a23dc4a38397a7ed34bd27b2a`), which has been promoted to **`trainedAI_network.json`** as the official 2D Triantica AI.
2. **Massive Gain in 10-Cycle Stability (`80.1%` Avg vs `71.6%` in `TF-B13`):**
   - Across all 1,000 evaluation games vs `HeuristicPlayer`, `TF-B14` won **`801W / 196L / 3D` (`80.1%`)**—an **+8.5 percentage point increase** over `TF-B13` (`71.6%`), never dropping below `76.0%` in any cycle (`76%–90%` range vs `52%–89%` in `TF-B13`).
   - Against `RandomPlayer`, `TF-B14` went **`1,000W / 0L / 0D` (`100.0%`)**, reaching a record **`175.3` vs `2.6`** mean cluster score in Cycle 10 (`3,000` consecutive wins across `TF-B12`–`TF-B14`).
3. **Pure Network Self-Play Requires Zero Heuristic Teacher:**
   - Even with `0%` `HeuristicPlayer` games in training, `Dual-Side QuiescenceMCTS-50` (`riskPenalty = 0.25`, `enclosureBonus = 0.15`) retained and improved anti-Heuristic mastery purely from self-play.
4. **Confirming the 1-Ply `3×3` ConvNet Architectural Ceiling:**
   - The win/loss anatomy confirms that when the 1-ply ConvNet avoids blunders, it dominates `HeuristicPlayer` (`57.0%` of all games are `+50` point blowouts, plus `23.1%` close wins = `80.1%`). The remaining `~19.6%` losses consist of `~7.4%` close positional races and `~12.2%` sporadic 1-ply cascade blunders (where the `7×7` receptive field and zero-padded borders miss a 1-move hole over a 40-move game, $0.995^{40} \approx 82\%$). Breaking into `95%–99%` requires the Phase 3 upgrades (Inference-Time Safety Filter, Toroidal Circular Padding, and 3D Mesh/Graph Global Context).

---

#### 9. Experiment `TF-B15` Results: Phase 3 Full Torus ($T^2$) + Circular Convolutions + Tactical Safety Filter (`99.0%` Peak Raw / `95.9%` Avg Raw / **`100.0%` Across All 10 Cycles with Safety Filter**)

Following the official release of the 2D baseline (`v1.0.0-2d`), **Phase 3 (`TF-B15`)** unified the board's topology and resolved **Bottlenecks 2, 3, and 4** simultaneously. Let's visualize how **Toroidal Circular Padding (`ToroidalPad2D`)** and the **Inference-Time Tactical Safety Filter** work under the hood before examining the 10-cycle telemetry:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     WHY ZERO-PADDING BLINDED TF-B14 AT THE BORDER vs. TOROIDALPAD2D (TF-B15) │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Suppose an enclosure cavity straddles the top/bottom seam (y = 0 ↔ y = 9): │
│                                                                              │
│  1. Standard `padding: 'same'` (Zero-Padding in TF-B5 through TF-B14):       │
│     ┌───────────────────────────┐                                            │
│     │  0   0   0   0   0   0    │  ◄── Out-of-bounds row y = -1 filled       │
│     ├───────────────────────────┤      with ZEROS! The 3×3 filter at y = 0   │
│     │ .   .  [▲] [?] [▼]  .    │      cannot see the enclosing pieces at    │
│     │ ...     (Interior)    ... │      row y = 9! (~19% border blunder rate) │
│     │ .   .  [◀] [█] [▶]  .    │                                            │
│     ├───────────────────────────┤                                            │
│     │  0   0   0   0   0   0    │  ◄── Out-of-bounds row y = 10 filled       │
│     └───────────────────────────┘      with ZEROS!                           │
│                                                                              │
│  2. `ToroidalPad2D` (`[B, 10, 10, C] ──► [B, 12, 12, C]` in TF-B15):         │
│     ┌───────────────────────────┐                                            │
│     │ .   .  [◀] [█] [▶]  .    │  ◄── Top pad copies Row y = 9!             │
│     ├───────────────────────────┤                                            │
│     │ .   .  [▲] [?] [▼]  .    │  ◄── Row y = 0 now sees its true           │
│     │ ...     (Interior)    ... │      toroidal neighbors at y = 9 and       │
│     │ .   .  [◀] [█] [▶]  .    │      x = 9! Followed by `padding: 'valid'` │
│     ├───────────────────────────┤      3×3 Conv2D ──► exact `[B, 10, 10, C]` │
│     │ .   .  [▲] [?] [▼]  .    │  ◄── Bottom pad copies Row y = 0!          │
│     └───────────────────────────┘                                            │
│                                                                              │
│     Mathematical Invariant Verified by Unit Test:                            │
│     Conv(Shift_{(dx,dy)}(X)) == Shift_{(dx,dy)}(Conv(X)) (diff = 5.96e-8)    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│     HOW THE O(1) INFERENCE-TIME TACTICAL SAFETY FILTER CLOSES THE LAST 1%    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  During fast 1-ply play (~0.8 ms/move), the raw Policy Head P_θ(a|s) already │
│  wins 99.0% of games (Cycle 7 & 10). Why does it ever lose that 1% game?     │
│  Because over a 40-move game, a 99.95%-accurate softmax distribution can     │
│  occasionally rank a subtle 1-move cavity blunder at #1 (p = 0.34) just      │
│  ahead of a completely safe territorial move at #2 (p = 0.31).               │
│                                                                              │
│  The Top-K Safety Filter (`topK = 12`, `safetyFilter = true`):               │
│  ─────────────────────────────────────────────────────────────               │
│  1. Extract the top K = 12 legal moves ranked by neural probability P_θ(a|s).│
│  2. For each candidate move a_k (k = 1..12):                                 │
│     • Check immediate AI enclosure claims (`claimed_AI`) in the 5-cell       │
│       orthogonal neighborhood (`countWouldBeEnclosed` — O(1) in-place!).     │
│     • If `claimed_AI > 0`, the AI retains the turn (`extraTurns >= 1`),      │
│       so opponent reply risk is ZERO (`oppRisk = 0`).                        │
│     • If `claimed_AI === 0` (turn passes to opponent), scan the 5×5 local    │
│       toroidal neighborhood for any immediate opponent enclosure replies     │
│       (`oppRisk = max opponent claims on next turn`).                        │
│  3. Score each top-K candidate:                                              │
│       S(a_k) = log P_θ(a_k|s) + β_enc · claimed_AI - β_risk · oppRisk        │
│  4. Result: If Move #1 leaves an unguarded cavity (`oppRisk > 0`), the       │
│     filter seamlessly promotes Move #2 (`oppRisk = 0`), turning a 99.0%      │
│     raw network into a 1,000W / 0L / 0D (100.0%) undefeated champion!        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

In summary, **Phase 3 (`TF-B15`)** combined three synergistic mechanisms:
1. **Full Toroidal Edge Symmetry ($T^2$):** Wrapped vertex coordinates modulo `(cols, rows)` in `hasCorner(x, y)` (`toroidalPlacement = true`), giving the $10 \times 10$ grid exactly $100$ unique topological vertices (each shared by $12$ triangle slots) so both corner placement and enclosure detection are 100% translation-invariant.
2. **Toroidal Circular Convolutions (`ToroidalPad2D` / `ToroidalPad3D`):** Replaced standard zero-padding (`padding: 'same'`) across all three `3×3` `Conv2D` layers with custom circular padding that wraps 1 row/column around all four borders (`10×10 → 12×12`) prior to `valid` convolution, achieving exact 2D periodic translation equivariance (`max abs diff = 5.96e-8`).
3. **Inference-Time Tactical Safety Filter (`topK = 12`):** Integrated `ConvNetwork.evaluateOpponentReplyRisk` (`5×5` toroidal neighborhood scan for 1-ply hole blunders + immediate AI enclosure bonuses) directly into `predictMove()`.

Warm-started from `TF-B14_best_network.json`, **`TF-B15`** ran 10 cycles of 100% pure network self-play (`Dual-Side QuiescenceMCTS-50`, Cosine LR `0.0003 → 0.00008`, `45,000`-sample sliding replay window, runtime `585.2 min`, `129,628` total samples generated):

| Cycle | Cosine LR | Buffer (`45K` Window, `New Samples`) | Loss (`Start → End`) | Self-Play (`P1 / P2 / D`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic **Raw Policy** (`Role P1 / P2`) | Raw Mean Score (`AI – Heur`, `Margin`) | Eval vs Heuristic **+Safety Filter** (`Mean AI–Heur`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B14` Best)** | `0.000109` | `45,000 / 45K` | `3.916 → 3.081` | — | **100.0%** (`174.6 – 2.9`) | `87.0%` on $T^2$ | `131.2 – 48.4` (`+82.8`) | **`99.0%`** (`173.8 – 15.1`) | `4.43 / 1.87` |
| **1** | `0.000300` | `12,976 / 45K` (`12,976`) | `4.007 → 2.992` | `55 / 44 / 1` | **100.0%** (`175.8 – 2.8`) | **`87.0%`** (`84% / 90%`) | `133.9 – 46.5` (`+87.4`) | **🛡️ `100.0%`** (`174.8 – 14.3`) | `4.41 / 1.89` |
| **2** | `0.000293` | `25,975 / 45K` (`12,999`) | `3.856 → 2.912` | `50 / 49 / 1` | **100.0%** (`177.7 – 2.5`) | **`93.0%`** (`92% / 94%`) | `144.8 – 39.1` (`+105.7`) | **🛡️ `100.0%`** (`176.2 – 13.8`) | `4.12 / 1.88` |
| **3** | `0.000274` | `38,983 / 45K` (`13,008`) | `3.759 → 2.923` | `50 / 50 / 0` | **100.0%** (`175.5 – 2.7`) | **`95.0%`** (`92% / 98%`) | `146.8 – 35.3` (`+111.5`) | **🛡️ `100.0%`** (`175.4 – 13.6`) | `4.61 / 1.90` |
| **4** | `0.000245` | **`45,000 / 45K`** (`12,868`) | `3.716 → 2.916` | `55 / 44 / 1` | **100.0%** (`177.6 – 2.5`) | **`96.0%`** (`96% / 96%`) | `149.7 – 32.2` (`+117.5`) | **🛡️ `100.0%`** (`175.1 – 14.2`) | `4.40 / 1.89` |
| **5** | `0.000209` | **`45,000 / 45K`** (`13,028`) | `3.906 → 2.962` | `48 / 52 / 0` | **100.0%** (`177.4 – 2.5`) | **`98.0%`** (`98% / 98%`) | `156.7 – 29.2` (`+127.5`) | **🛡️ `100.0%`** (`173.4 – 14.4`) | `4.69 / 1.90` |
| **6** | `0.000171` | **`45,000 / 45K`** (`13,028`) | `3.720 → 2.981` | `47 / 53 / 0` | **100.0%** (`178.1 – 2.4`) | **`98.0%`** (`100% / 96%`) | `158.7 – 25.9` (`+132.8`) | **🛡️ `100.0%`** (`173.0 – 14.8`) | `4.47 / 1.90` |
| **7** | `0.000135` | `13,002 / 45K` (`13,002`) | `3.822 → 3.016` | `57 / 43 / 0` | **100.0%** (`177.7 – 2.5`) | **🏆 `99.0%`** (`98% / 100%`) | `157.3 – 26.0` (`+131.3`) | **🛡️ `100.0%`** (`174.8 – 14.8`) | `4.22 / 1.90` |
| **8** | `0.000106` | `25,974 / 45K` (`12,972`) | `3.699 → 3.010` | `53 / 47 / 0` | **100.0%** (`177.5 – 2.6`) | **`97.0%`** (`98% / 96%`) | `155.5 – 28.6` (`+126.9`) | **🛡️ `100.0%`** (`173.1 – 16.0`) | `4.41 / 1.89` |
| **9** | `0.000087` | `38,862 / 45K` (`12,888`) | `3.679 → 3.073` | `45 / 53 / 2` | **100.0%** (`178.4 – 2.4`) | **`97.0%`** (`98% / 96%`) | **`159.8 – 24.3` (`+135.5`)** | **🛡️ `100.0%`** (`173.5 – 14.9`) | `4.66 / 1.89` |
| **10** | `0.000080` | **`45,000 / 45K`** (`12,859`) | `3.754 → 3.106` | `59 / 41 / 0` | **100.0%** (`178.0 – 2.4`) | **🏆 `99.0%`** (`100% / 98%`) | `154.6 – 28.2` (`+126.4`) | **🛡️ `100.0%`** (`173.0 – 15.7`) | `4.18 / 1.88` |

**Key Takeaways from `TF-B15` (Official Phase 3 Toroidal Champion):**
1. **Toroidal Circular Padding (`ToroidalPad2D`) Eliminated Border Blind Spots (`95.9%` Avg Raw Policy vs `80.1%` in `TF-B14`):**
   - Even **without** the inference safety filter (`safetyFilter = false`), replacing zero-padding with circular toroidal padding lifted the raw 1-ply ConvNet's 10-cycle average against `HeuristicPlayer` by **+15.8 percentage points** (`80.1% → 95.9%`, `959W / 40L / 1D`), peaking at **`99.0%` (`99W / 1L / 0D`)** in Cycles 7 and 10 and averaging **`97.7%` across Cycles 4–10**.
   - Mean raw cluster score margin more than doubled from `+56.4` in `TF-B14` to **`+135.5` (`159.8` vs `24.3`)** in Cycle 9, proving that a large fraction of `TF-B14`'s residual blunders occurred near the `x, y ∈ {0, 9}` borders where zero-padding previously severed wrap-around enclosures.
2. **Flawless Mastery with the Inference-Time Safety Filter (`1,000W / 0L / 0D` = `100.0%` Across All 10 Cycles):**
   - With `safetyFilter = true` (`topK = 12`), `TF-B15` went **`100.0%` (`100W / 0L / 0D`, `100% as P1`, `100% as P2`) in every single cycle from Cycle 1 to Cycle 10** (`1,000` consecutive wins vs `HeuristicPlayer`, mean score **`174.2` vs `14.7`**) and **`1,000W / 0L / 0D` (`100.0%`, `177.4` vs `2.5`)** vs `RandomPlayer`.
3. **Production Deployment & Multi-Tier Human Play (`trainedAI_network.json`):**
   - `TF-B15_best_network.json` (`Cycle 7` checkpoint: `99.0%` Raw / `100.0%` +Safety vs Heuristic) has been promoted to `trainedAI_network.json`, powering three selectable difficulty tiers in browser/mobile play: **Easy** (high-temperature sampling), **Medium** (Raw 1-ply ConvNet `99%`), and **Hard** (Master ConvNet + Tactical Safety Filter `100%`), alongside **`HumanGameLogger`** telemetry to record canonical `(state, move, claimed, outcome)` trajectories from real human matches.

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

**System Component:** Automated Peak Checkpoint Promotion & Tournament Harness (`*_best_network.json` auto-promotion in `src/server/train_tfjs.cjs` and cross-checkpoint tournament mode in `src/server/evaluate.cjs`)

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
| 4: BatchNorm+Conv | TF-B7 | ✅ Merged | — | — | BatchNorm integrated directly into Phase 2 ConvNet (`ConvNetwork.cjs`) |
| 5: MCTS-50 | TF-B8 | ❌ Failed | 30% | 0% | Inverted value perspective bug: network learned to lose (see Bug 5) |
| 5: MCTS-50 | TF-B9 | ✅ Done | 79% (84% peak) | 0% | Single-perspective MCTS + 400× policy loss fix; +41% jump vs Random |
| 5: MCTS-50 Mixed | TF-B10 | ✅ Done | 87% | 0% | 50% Network / 50% Heuristic; record 87% vs Random |
| 5.5: Quiescence MCTS | TF-B12 | ✅ Done | **100%** | **63%** | **Heuristic Wall Broken!** 100% vs Random (all 10 cycles), 63% vs Heuristic (66% P1, 60% P2; random 2-ply eval C8–C10) |
| 5.6: Dual-Side Quiescence | TF-B13 | ✅ Done | **100%** | **73% (89% peak)** | **New Record vs Heuristic!** 89% peak (C8: 90% P1, 88% P2), 73% final (C10: 80% P1, 66% P2), 71.6% 10-cycle avg; 115.9K dual-side MCTS samples |
| 5.7: Sliding-Window Pure SP | TF-B14 | ✅ Done | **100%** | **78% (90% peak, 80.1% avg)** | **Official 2D Release Champion (`v1.0.0-2d`)!** 90% peak (C9: 88% P1, 92% P2, score `123.8–54.2`), 80.1% 10-cycle avg (`76%–90%` band) under 100% pure network self-play + 45K sliding buffer + Cosine LR. See [`docs/EXECUTION_ROADMAP.md`](./EXECUTION_ROADMAP.md). |
| 6.0: Phase 3 Full Torus ($T^2$) | TF-B15 | ✅ Done | **100%** | **99% Raw (95.9% avg) / 100% +Safety** | **All-Time Record!** Full $T^2$ toroidal placement + `ToroidalPad2D` circular convolutions + Tactical Safety Filter (`topK=12`). `1,000W/0L/0D` (`100.0%`) with safety filter; `99.0%` peak (`95.9%` avg) raw policy. Promoted to `trainedAI_network.json`. |

**Status Legend:**
- ✅ Done — Experiment complete, documented
- 🔄 Running — Currently executing
- ⏳ Pending — Not started
- ❌ Failed — Did not meet success criteria (requires revision)

---

### Summary: Expected vs. Achieved Impact Across All Phases

| Phase | Architectural Upgrade | Expected vs Random | Expected vs Heuristic | **Actual Achieved (`TF-B4` → `TF-B15`)** |
|-------|-----------------------|--------------------|-----------------------|------------------------------------------|
| 1 | Experience Replay Buffer (`TF-B4`) | 45% | 0% | **45% vs Random, 0% vs Heuristic** (Feedforward bottleneck) |
| 2–3 | 18-Channel Canonical Slot Encoding + Conv2D + Value Head (`TF-B5`/`B6`) | 50% | 0% | **50% vs Random, 0% vs Heuristic** (Spatial features learned, but no lookahead) |
| 5 | Single-Perspective MCTS-50 + Bug 6–11 Fixes (`TF-B9`/`B10`) | 75% | 5–15% | **87% vs Random, 0% vs Heuristic** (Vanilla MCTS blind to deterministic cascades) |
| 5.5 | Quiescence MCTS + 50/50 Mixed Opponents (`TF-B12`) | 85% | 15–20% | **100% vs Random, 63% vs Heuristic** (🏆 Heuristic Wall Broken!) |
| 5.6 | Dual-Side Quiescence MCTS + 75/25 Mix (`TF-B13`) | 100% | 70%+ | **100% vs Random, 89% Peak (71.6% Avg) vs Heuristic** |
| 5.7 | 100% Pure Self-Play + 45K Sliding Replay + Cosine LR (`TF-B14`) | 100% | 80%+ | **100% vs Random, 90% Peak (80.1% Avg) vs Heuristic** (`v1.0.0-2d` Champion) |
| 6.0 | Full $T^2$ Torus + Circular Conv (`ToroidalPad2D`) + Safety Filter (`TF-B15`) | 100% | 95%+ | **100% vs Random, 99.0% Peak (95.9% Avg) Raw / 100.0% (`1,000W/0L`) +Safety** |

---

---

> [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | **[📚 Training Book Index](./README.md)** | [Chapter 12: Toroidal 3D Visualization & Human Telemetry →](./12-toroidal-3d-and-telemetry.md)
