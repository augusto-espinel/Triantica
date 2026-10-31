> **[📚 Training Book Index](./README.md)** | [← Chapter 3: The Architecture Journey](./03-architecture-journey.md) | [Chapter 5: Parameters Explored →](./05-parameters-explored.md)

---

## 4. Hardware: Mac Studio M2 Ultra & Matrix Sizing Deep Dive {#4-hardware}

### The Unified Memory Advantage

```
┌─────────────────────────────────────────────────────────────┐
│              Traditional x86 + Discrete GPU                 │
│                                                             │
│   CPU ────── PCIe Bus (32–64 GB/s) ────── GPU               │
│   System RAM                              Discrete VRAM     │
│   (JS Replay Buffer) ──[Serialize/Copy]─► (Tensor Memory)   │
│                           │                                 │
│                    (I/O Bottleneck!)                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              Apple M2 Ultra Unified Memory                  │
│                                                             │
│   24-Core CPU ◄──── Unified Memory Fabric ────► 60/76-Core  │
│                     800 GB/s Bandwidth          Metal GPU   │
│                     (Up to 192GB Shared Pool)               │
│                                                             │
│   Zero-copy tensor allocation. 500,000-state Replay Buffer  │
│   and batched MCTS leaf tensors reside in the same pool.    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Unified Memory is a Game-Changer for AlphaZero-Style MCTS:**
In standard supervised deep learning, large batches sit on the GPU while the CPU stays idle. In **MCTS Self-Play RL**, execution alternates thousands of times per game between:
1. **CPU-bound Tree Traversal & Board Simulation** (`GridManager.clone()`, `executeMove()`, `claimEnclosedTriangles()`).
2. **GPU-bound Neural Network Inference** (`TFConvNetwork.batchPredict()` on $B=8$ MCTS leaves).
3. **GPU-bound Mini-Batch Backpropagation** (`trainBatchWithIsolation()` on $B=256$ replay samples).

On discrete PCIe GPUs, copying small MCTS leaf batches ($B=8$) back and forth across PCIe introduces latency that dwarfs the matrix multiplication itself. On the **Mac Studio M2 Ultra** (800 GB/s memory bandwidth), CPU arrays (`Float32Array`) and TensorFlow C++/Metal tensors share physical memory pages, eliminating bus stalls.

---

### How Matrix & Tensor Sizes Come About at Every Step

Every layer in `TFConvNetwork` transforms a 4D spatial tensor or 2D flattened matrix. Understanding **where each dimension comes from** and **how `im2col` (Image-to-Column) turns 2D convolutions into dense Matrix Multiplications (`GEMM`)** explains both the network's representational power and its hardware efficiency on the M2 Ultra.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           END-TO-END TENSOR & MATRIX PIPELINE (Batch Size B = 256)           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [STEP 0: Canonical Board Encoding]                                          │
│   10×10 Grid × 2 Slots/Cell × 9 One-Hot States/Slot = 18 Channels            │
│   Tensor Shape: [256, 10, 10, 18]  (460,800 floats = 1.84 MB per batch)      │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 32 filters) + BN + ReLU     │
│  [STEP 1: Conv Block 1 — Local Edge & Diagonal Detector]                     │
│   Kernel Tensor: [3, 3, 18, 32]  ──► Matrix Multiply: [25,600 × 162]·[162×32]│
│   Output Shape:  [256, 10, 10, 32] (5,184 weights + 32 biases + 128 BN)      │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 64 filters) + BN + ReLU     │
│  [STEP 2: Conv Block 2 — Multi-Cell Corner & Wedge Detector]                 │
│   Kernel Tensor: [3, 3, 32, 64]  ──► Matrix Multiply: [25,600 × 288]·[288×64]│
│   Output Shape:  [256, 10, 10, 64] (18,432 weights + 64 biases + 256 BN)     │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 64 filters) + BN + ReLU     │
│  [STEP 3: Conv Block 3 — 7×7 Receptive Field Enclosure Detector]             │
│   Kernel Tensor: [3, 3, 64, 64]  ──► Matrix Multiply: [25,600 × 576]·[576×64]│
│   Output Shape:  [256, 10, 10, 64] (36,864 weights + 64 biases + 256 BN)     │
│         │                                                                    │
│         ├───────────────────────────────────────┐                            │
│         ▼                                       ▼                            │
│  [STEP 4A: POLICY HEAD ("Where to Play")]  [STEP 4B: VALUE HEAD ("Who Wins")]│
│   1×1 Conv (2 filters): [256, 10, 10, 2]    1×1 Conv (1 filter): [256,10,10,1]│
│   Flatten:              [256, 200]          Flatten:             [256, 100]  │
│   Dense (200 → 400):    [256, 400]          Dense 1 (100 → 64):  [256, 64]   │
│   Activation:           Sigmoid ∈ (0, 1)    Dense 2 (64 → 1):    [256, 1]    │
│   (400 actions = 10×10 cells × 4 rots)      Activation:          tanh ∈ [-1,1]│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Derivation of Matrix Dimensions

1. **Input Tensor `[B, 10, 10, 18]`:**
   - **Why `10 × 10`?** Matches the physical $10 \times 10$ Triantica board so spatial adjacency (up, down, left, right, diagonal) is preserved in memory coordinates.
   - **Why `18` channels?** Each cell holds up to **2 triangles** (Slot 0 for rotations $\{0,1\}$, Slot 1 for rotations $\{2,3\}$). Each slot has **9 mutually exclusive states**:
     - `1` channel for *Empty* (`ch 0` for Slot 0, `ch 9` for Slot 1)
     - `4` channels for *My Piece* at rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ$
     - `4` channels for *Opponent Piece* at rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ$
   - **Why `padding='same'` vs. Circular Toroidal Padding (`ToroidalPad2D`)?** In standard planar ConvNets (`TF-B5`–`TF-B14`), `padding='same'` adds a 1-cell border of zeros around the $10 \times 10$ grid to create a $12 \times 12$ input window ($H_{out} = \lfloor \frac{10 + 2(1) - 3}{1} \rfloor + 1 = 10$). However, because Triantica is played on a continuous $10 \times 10$ torus ($T^2$), zero-padding creates an artificial "wall of zeros" at the borders! In `TF-B15` (Phase 3), we replaced zero-padding with **`ToroidalPad2D`**—wrapping Row `0` $\leftrightarrow$ Row `9` and Col `0` $\leftrightarrow$ Col `9` into a $[B, 12, 12, C]$ tensor before every `Conv2D(padding='valid')` layer so boundary enclosures are detected identically to center-board enclosures.

2. **How Convolutions Become Matrix Multiplications (`im2col` GEMM):**
   On hardware accelerators (Apple AMX coprocessor & Metal GPU), a 2D convolution is not executed as slow nested `for` loops. Instead, it is unrolled into a single massive **General Matrix Multiplication (`GEMM`)** $Y = X_{\text{col}} \cdot W$:
   - Across a batch of $B = 256$ boards, there are $256 \times 10 \times 10 = \mathbf{25,600}$ spatial patch locations.
   - **Conv1 (`18 → 32`):** Each $3 \times 3$ patch across $18$ input channels is flattened into a row vector of length $3 \times 3 \times 18 = \mathbf{162}$. The 32 filters form a weight matrix of shape $\mathbf{[162, 32]}$ ($5,184$ weights). The layer executes:
     $$\underbrace{[25,600 \times 162]}_{\text{Unrolled Patches } X_{\text{col}}} \times \underbrace{[162 \times 32]}_{\text{Filter Matrix } W_1} = \underbrace{[25,600 \times 32]}_{\text{Reshaped to } [256, 10, 10, 32]}$$
   - **Conv2 (`32 → 64`):** Each $3 \times 3$ patch across $32$ channels flattens to $3 \times 3 \times 32 = \mathbf{288}$ elements. Multiplied by $W_2$ of shape $\mathbf{[288, 64]}$ ($18,432$ weights):
     $$[25,600 \times 288] \times [288 \times 64] = [25,600 \times 64] \longrightarrow [256, 10, 10, 64]$$
   - **Conv3 (`64 → 64`):** Each $3 \times 3$ patch across $64$ channels flattens to $3 \times 3 \times 64 = \mathbf{576}$ elements. Multiplied by $W_3$ of shape $\mathbf{[576, 64]}$ ($36,864$ weights):
     $$[25,600 \times 576] \times [576 \times 64] = [25,600 \times 64] \longrightarrow [256, 10, 10, 64]$$

3. **Why $1 \times 1$ Bottleneck Convolutions Before the Dense Heads?**
   - After Conv3, each board position has a `[10, 10, 64]` feature map ($6,400$ floats). Connecting $6,400$ features directly to $400$ policy outputs would require $6,400 \times 400 = \mathbf{2,560,000}$ parameters—causing severe overfitting and slowing down MCTS inference by $15\times$!
   - Instead, following AlphaGo Zero, we apply a **$1 \times 1$ bottleneck convolution**:
     - **Policy Bottleneck (`pConv`):** Projects `[10, 10, 64]` $\to$ `[10, 10, 2]` via a $[64 \times 2]$ matrix ($128$ weights). Flattening `[10, 10, 2]` yields a compact **200-element vector** (2 learned summary features per board cell), which then connects to the **400-unit Policy Dense layer** via a $[200 \times 400]$ matrix ($80,000$ weights).
     - **Value Bottleneck (`vConv`):** Projects `[10, 10, 64]` $\to$ `[10, 10, 1]` via a $[64 \times 1]$ matrix ($64$ weights). Flattening yields a **100-element vector** (1 scalar territory score per cell), which feeds `[100 × 64]` ($6,400$ weights) $\to$ `[64 × 1]` ($64$ weights) $\to$ `tanh`.

#### Complete Parameter & Compute Breakdown (`TFConvNetwork`)

| Layer | Input Shape | Matrix Operation (`GEMM`) | Output Shape | Parameters | FLOPs / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `[B, 10, 10, 18]` | — | `[B, 10, 10, 18]` | `0` | — |
| **Conv1 + BN** | `[B, 10, 10, 18]` | `[100B, 162] × [162, 32]` | `[B, 10, 10, 32]` | `5,344` | `1.04M` |
| **Conv2 + BN** | `[B, 10, 10, 32]` | `[100B, 288] × [288, 64]` | `[B, 10, 10, 64]` | `18,752` | `3.69M` |
| **Conv3 + BN** | `[B, 10, 10, 64]` | `[100B, 576] × [576, 64]` | `[B, 10, 10, 64]` | `37,184` | `7.37M` |
| **Policy `1×1` Conv**| `[B, 10, 10, 64]` | `[100B, 64] × [64, 2]` | `[B, 10, 10, 2]` | `130` | `25.6K` |
| **Policy Dense** | `[B, 200]` | `[B, 200] × [200, 400]` | `[B, 400]` | `80,400` | `160.0K` |
| **Value `1×1` Conv** | `[B, 10, 10, 64]` | `[100B, 64] × [64, 1]` | `[B, 10, 10, 1]` | `65` | `12.8K` |
| **Value Dense 1** | `[B, 100]` | `[B, 100] × [100, 64]` | `[B, 64]` | `6,464` | `12.8K` |
| **Value Dense 2** | `[B, 64]` | `[B, 64] × [64, 1]` | `[B, 1]` | `65` | `128` |
| **TOTAL** | | | **400 Policy + 1 Value**| **148,404** (~594 KB)| **~12.3M FLOPs** |

### Why These Exact Matrix Sizes Fly on the M2 Ultra

1. **Entire Network Weights Fit in L2 Cache (`594 KB`):**
   - At `148,404` `float32` parameters (`~594 KB`), the *entire neural network weight set* fits inside the M2 Ultra's **on-chip cache hierarchy** (32MB–64MB L2 + System Level Cache). During MCTS search (where the network is queried 50 times per move across ~150 moves = 7,500 forward passes per game), weights never have to be fetched from main DRAM.
2. **SIMD Tile Alignment (`32`, `64`, `256`):**
   - Apple Silicon matrix coprocessors (AMX) and Metal SIMD groups operate on **32-wide and 64-wide register tiles**. Choosing filter depths of **`32 → 64 → 64`**, a value hidden layer of **`64`**, a MCTS leaf batch size of **`8`**, and a training mini-batch size of **`256`** ensures zero partial-tile padding waste during `GEMM` execution.
3. **Zero-Swap 500K Replay Buffer (`~6.8 GB` in 192GB Unified RAM):**
   - Storing `500,000` canonical experiences—each containing a `1,800`-float state (`7.2 KB`), a `400`-float MCTS visit distribution (`1.6 KB`), `chosenIndex`, and `valueTarget`—consumes less than **4% of the M2 Ultra's 192GB unified memory**, allowing instant random batch sampling (`batchSize = 256`, `replayRatio = 2x`) with zero disk I/O.

### Node.js Version & Native Backend Notes

```
⚠️ CRITICAL: TensorFlow.js native bindings require Node v20 LTS

Node v25 → C++ V8 ABI mismatch → @tensorflow/tfjs-node fails to load
Node v20 → Native C++ / Accelerate / Metal bindings load cleanly

Use: fnm use v20.20.2  (or nvm use 20.20.2)
```

### Batch Size Optimization on M2 Ultra

```
Batch Size │  Hardware Utilization │  Throughput & Gradient Behavior
───────────┼───────────────────────┼──────────────────────────────────────────
    1      │  ~5%  (overhead bound)│  High kernel-launch overhead, noisy grads
    8      │  ~45% (MCTS leaf opt) │  Sweet spot for parallel MCTS leaf evals
   32      │  ~65%                 │  Good for pure feedforward policy updates
  256      │  ~98% (AMX/GPU full)  │  Optimal for ConvNet + BatchNorm + Replay
```

**Recommendation:** Use `mctsBatchSize: 8` during tree search (balancing virtual-loss tree diversity against vectorization) and `batchSize: 256` with `tf.train.adam(0.001)` during replay-buffer gradient updates.

### Wall-Clock Processing Time Derivation (Why Self-Play is 96% of Cycle Time)

A common intuition from supervised deep learning is that **backpropagation** (gradient descent after the games finish) is the primary computational bottleneck, while playing the games is fast. In **AlphaZero / QuiescenceMCTS reinforcement learning**, the exact opposite is true: **~96% of wall-clock time is spent generating the 100 self-play games**, while **backpropagation takes only ~85 seconds (~3% of cycle time)** and **3-way evaluation takes ~25 seconds (~1%)**.

Here is the first-principles derivation measured on the Mac Studio M2 Ultra (`TF-B14` / `TF-B15`, `QuiescenceMCTS-50`, `100` games/cycle):

#### 1. Phase A — Dual-Side `QuiescenceMCTS-50` Self-Play (`~2,580s` = `~43.0 min`, `96.1%`)

In pure Dual-Side MCTS self-play, **both** Player 1 and Player 2 execute a 50-simulation Monte Carlo Tree Search before every move on the board:
- **Moves per cycle:**
  $$N_{\text{moves}} = 100\text{ games/cycle} \times \approx 130\text{ moves/game} = \mathbf{13,000\text{ moves/cycle}}$$
- **MCTS tree simulations per cycle:**
  $$N_{\text{sims}} = 13,000\text{ moves} \times 50\text{ simulations/move} = \mathbf{650,000\text{ MCTS simulations/cycle}}$$
- **Work performed at each of the `650,000` leaf expansions:**
  1. **Board Cloning & Tree Traversal (CPU / V8 JS):** Clone the $10 \times 10$ toroidal `GridManager` along the search path, enumerate all legal moves (`isValidMove` across up to 400 slots), and apply `placeTriangle` + `claimEnclosedTriangles`.
  2. **Quiescence Cascade Rollout (`quiescenceMode = 'greedy'`, depth $\le 5$):** Whenever a leaf move is non-enclosing (`claimed === 0`), `QuiescenceMCTS` scans all opponent replies using $O(1)$ `countWouldBeEnclosed` checks and greedily plays out multi-step opponent enclosure chains (up to 5 plies) so the value head evaluates the *post-cascade quiescent board* rather than a mid-tactical horizon trap.
  3. **Batched Neural Network Forward Passes (`mctsBatchSize = 8`):** Every 8 virtual-loss leaf nodes are grouped into a `[8, 10, 10, 18]` tensor and evaluated by `TFConvNetwork.runBatch()`:
     $$N_{\text{forward\_batches}} = \frac{650,000\text{ simulations}}{8\text{ leaves/batch}} = \mathbf{81,250\text{ ConvNet forward calls/cycle}}$$
     At `~12.3M FLOPs` per sample (`~98.4M FLOPs` per batch of 8), the neural network executes **`~8.0 TeraFLOPs`** of forward inference per cycle across `81,250` separate JS $\leftrightarrow$ C++ TF.js bridge calls, plus `650,000+` JavaScript board state clones and quiescence checks.
  - **Measured Throughput:** `~25.8 seconds/game` (`~198 ms/move` for 50 simulations $\approx \mathbf{3.96\text{ ms}}$ per QuiescenceMCTS simulation) $\implies$ **`~43.0 minutes` per 100-game cycle**.

#### 2. Phase B — Backpropagation & Experience Replay (`~85s` = `~1.4 min`, `3.1%`)

Once the 100 self-play games finish, the pipeline has collected $\approx 13,000$ canonical `(state, mctsPolicy, chosenMoveIdx, valueTarget)` training tuples:
- **Why Backpropagation is $30\times$ Faster than Self-Play:**
  - Instead of small `batchSize = 8` forward passes interleaved with sequential JavaScript game logic, backpropagation packs states into large **`batchSize = 256`** tensors (`[256, 10, 10, 18]`), saturating the M2 Ultra's AMX / Metal SIMD tiles at ~98% hardware utilization.
  - **Current-cycle gradient updates (`10` epochs):**
    $$\frac{13,000\text{ samples}}{256\text{ samples/batch}} \approx 51\text{ batches/epoch} \times 10\text{ epochs} = \mathbf{510\text{ gradient steps}}$$
  - **Sliding-Window Replay Buffer updates (`replayRatio = 2.0`):**
    $$2.0 \times 51\text{ batches} \approx \mathbf{102\text{ replay gradient steps}}$$
  - **Total Backprop Compute:** **`612` large-batch (`B = 256`) forward+backward steps** (`156,672` sample evaluations total, compared to `650,000` leaf evaluations during self-play, and executed in `612` bulk C++ kernel calls instead of `81,250` small calls!).
  - **Measured Runtime (`TF-B14` Cycles 1–10):** **`83.2s – 86.3s` (`~1.4 minutes`) per cycle**.

#### 3. Phase C — 3-Way Evaluation (`300` Games Total: `~25s` = `~0.4 min`, `0.8%`)

After weight updates, the model plays **300 full evaluation games** (`100` vs Random Raw, `100` vs Heuristic Raw Policy, `100` vs Heuristic +Safety Filter):
- Because evaluation tests the **1-ply policy network** (`predictMove()` with `exploration = 0`, 1 forward pass per AI move instead of 50 MCTS simulations), 300 games require only:
  $$300\text{ games} \times \approx 65\text{ AI moves/game} = \mathbf{19,500\text{ single-state forward passes}}$$
- Even including the $O(1)$ `countWouldBeEnclosed` + $5 \times 5$ `evaluateOpponentReplyRisk` checks on the top-$K=12$ moves in the `+Safety Filter` split (`< 0.4 ms/move`), all 300 evaluation games complete in **`~22–28 seconds`**.

#### Summary Table: Per-Cycle Time Budget (`100` Games/Cycle, `QuiescenceMCTS-50`)

| Cycle Phase | Operations per Cycle | Tensor Batch Size | Wall-Clock Time | Share of Cycle | Scaling Law |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. QuiescenceMCTS Self-Play** | `100` games, `13,000` moves, **`650,000` tree sims + cascades**, `81,250` net calls | `B = 8` | **`~43.0 min`** (`2,580s`) | **`96.1%`** | Linear in `games × mctsSims` |
| **2. Backpropagation + Replay** | `13,000` new samples (`10` iters) + `26,000` replay samples (`612` steps) | `B = 256` | **`~1.4 min`** (`85s`) | **`3.1%`** | Linear in `games × (iterations + replayRatio)` |
| **3. 3-Way Evaluation** | `300` games (`100` Rand, `100` Heur Raw, `100` Heur +Safety), `~19,500` net calls | `B = 1` | **`~0.4 min`** (`25s`) | **`0.8%`** | Linear in `evalGames` |
| **Total per Cycle** | **400 total games played** (`100` MCTS + `300` 1-ply eval) | — | **`~44.8 min`** (`2,690s`) | **`100.0%`** | **`~7.4 hours` for 10 cycles** |

> **Practical Takeaway for Experiment Design:** Because self-play dominates 96% of runtime, total training duration scales almost purely with `cycles × games × mctsSims`. Increasing training epochs (`--iterations 10 → 20`), increasing `--replay-ratio`, or expanding evaluation from 200 to 300 games (as in our 3-way evaluation split) adds less than **1–2% overhead** to total cycle time!

---

---

> [← Chapter 3: The Architecture Journey](./03-architecture-journey.md) | **[📚 Training Book Index](./README.md)** | [Chapter 5: Parameters Explored →](./05-parameters-explored.md)
