# AI Architecture Technical Reference

**Version:** 2.1  
**Last Updated:** 2026-10-25  
**Status:** MCTS value perspective bug fixed

---

## Neural Network Specification

### Architecture Overview

```
Input Layer:     401 neurons (flattened state + player flag)
Hidden Layers:   Configurable by difficulty
Output Layer:    400 neurons (move confidence scores)
Activation:      Sigmoid (all layers)
Framework:       TensorFlow.js + Metal GPU (Node v20 LTS)
```

### Difficulty Configurations

| Difficulty | Hidden Layers | Total Weights | Approx Size |
|------------|---------------|---------------|-------------|
| Easy       | [100]         | 401×100 + 100×400 = 80,100 | ~320 KB |
| Medium     | [200]         | 401×200 + 200×400 = 160,200 | ~640 KB |
| Hard       | [267, 160]    | 401×267 + 267×160 + 160×400 = 160,667 | ~640 KB |

### Framework: TensorFlow.js

The training pipeline uses **TensorFlow.js with native Metal GPU acceleration** on macOS.
This replaces the original brain.js implementation, which lacked:
- Multi-head outputs (Policy + Value)
- Custom loss functions (MSE was hard-coded)
- Residual connections
- Convolutional layers
- GPU acceleration

**Migration verified:** TF.js sigmoid network with identical architecture reproduces
brain.js behavior. Weight format conversion confirmed correct. Training speed is
~100x faster with Metal GPU on M2 Ultra.

**Environment requirements:**
- **Node v20.20.2 LTS** (Node v25 breaks tfjs-node native bindings)
- `@tensorflow/tfjs-node` with Metal GPU support
- Managed via `fnm` for version switching

---

## Input Encoding

### State Vector Construction

```javascript
// Pseudocode for state representation
const state = [];

for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
        const cell = grid[y][x];
        const t1 = cell.triangles[0];  // First triangle (if any)
        const t2 = cell.triangles[1];  // Second triangle (if any)
        
        // Encode player ownership
        state.push(normalizePlayer(t1?.player));    // 1=AI, -1=opponent, 0=empty
        state.push(normalizeRotation(t1?.rotation)); // 0.0-0.75 or -1
        
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation));
    }
}

// Add current player context
state.push(currentPlayer === aiPlayerNumber ? 1 : -1);
```

### Encoding Functions

```javascript
normalizePlayer(p) {
    if (!p) return 0;
    return p === this.playerNumber ? 1 : -1;
}

normalizeRotation(r) {
    if (r === undefined || r === -1) return -1;
    return r / 4.0;  // Maps {0,1,2,3} → {0.0, 0.25, 0.5, 0.75}
}
```

### Input Vector Layout

```
Index 0-3:     Cell (0,0) - [P1, R1, P2, R2]
Index 4-7:     Cell (1,0) - [P1, R1, P2, R2]
...
Index 396-399: Cell (9,9) - [P1, R1, P2, R2]
Index 400:     Current player flag
```

---

## Output Encoding

### Output Vector Interpretation

```javascript
// Output index → move mapping
const i = outputIndex;
const rotation = i % 4;           // 0-3
const cellIndex = Math.floor(i / 4);
const x = cellIndex % cols;        // 0-9
const y = Math.floor(cellIndex / cols); // 0-9
```

### Output Value Semantics

| Value | Meaning |
|-------|---------|
| ~0.01 | Invalid move (default for illegal positions) |
| ~0.50 | Neutral expected value |
| ~0.99 | Highly confident good move |

### Move Selection Process

```javascript
predictMove() {
    // Epsilon-greedy: 10% exploration
    if (Math.random() < explorationFactor) {
        return getRandomValidMove();
    }
    
    // Get network output
    const output = network.run(state);
    
    // Sort all moves by confidence
    const moves = output.map((score, i) => ({
        ...indexToMove(i),
        score
    })).sort((a, b) => b.score - a.score);
    
    // Return highest-scoring valid move
    for (const move of moves) {
        if (isValidMove(move)) return move;
    }
    
    // Fallback
    return getRandomValidMove();
}
```

---

## Reward Calculation

### Immediate Reward Components

The AI evaluates every legal move by simulating it on a temporary board:

```javascript
calculateImmediateRewardVector(state, player) {
    const output = new Array(400).fill(0.01);  // Default: invalid
    
    for (each cell position) {
        for (each rotation 0-3) {
            if (isLegalMove(x, y, rotation)) {
                // Create temporary board
                const tempGrid = clone(currentGrid);
                
                // Simulate placement
                tempGrid.placeTriangle(x, y, rotation, player);
                const claims = tempGrid.claimEnclosedTriangles();
                
                // Calculate reward
                const claimReward = claims * 0.3;
                const clusterReward = tempGrid
                    .getConnectedClusterSize(x, y, rotation) * 0.1;
                
                const totalReward = claimReward + clusterReward;
                
                // Scale to [0.01, 0.99]
                output[index] = scaleReward(totalReward);
            }
        }
    }
    
    return output;
}
```

### Reward Components Summary

| Component | Type | Factor | Purpose | Default |
|-----------|------|--------|---------|--------|
| **Claim bonus** | Immediate | 0.3/triangle | Encourage enclosure creation | ON |
| **Cluster growth** | Immediate | 0.1/triangle | Encourage cluster building | ON |
| **Largest lead** | Immediate | 0.2/lead delta | Direct win condition optimization | ON |
| **Win/loss** | Final outcome | ±1.0 (λ=1) | Credit ALL moves equally | ON |
| **Cluster contribution** | Final outcome | 0.1/triangle (TD λ=0.9) | Credit moves IN winning cluster | **OFF** |

> ⚠️ **Exp 11 Result:** TD(λ) cluster-contribution reward **degraded performance** (49.2% vs 51-53% baseline). It's now **disabled by default** but configurable.

#### Configurable TD(λ) Cluster Reward

```javascript
// In AIPlayer constructor:
this.enableClusterReward = config.enableClusterReward ?? false;  // OFF by default
this.clusterRewardFactor = config.clusterRewardFactor ?? 0.1;
this.clusterLambda = config.clusterLambda ?? 0.9;
this.clusterGamma = config.clusterGamma ?? 0.99;
```

```bash
# Enable for future experiments:
node src/server/train.js --enable-cluster-reward true \
                          --cluster-reward-factor 0.1 \
                          --cluster-lambda 0.9
```

### Reward Scaling

```javascript
// REALISTIC bounds (Exp 10+ — not theoretical maximums!)
const maxClaimBonus = 1.5;    // max ~5 claims × 0.3
const maxClusterBonus = 10.0;  // realistic ~100 triangles × 0.1
const maxLeadBonus = 4.0;      // realistic ±20 lead delta × 0.2 (Exp 10+)
const maxImmediateReward = 1.5 + 10.0 + 4.0;  // = 15.5
const minReward = -5.0;        // realistic worst case
const rewardRange = maxImmediateReward - minReward;  // = 20.5

scaledReward = (reward - minReward) / rewardRange;
scaledReward = Math.max(0.01, Math.min(0.99, scaledReward));
```

> ⚠️ **Critical:** Never use theoretical maximums for normalization. The lead bonus theoretical max is 40 (200×0.2) but realistic max is ~4. Using theoretical values compresses 95% of the output range into unused territory, causing fake convergence.

### Final Outcome Integration (TD(λ) Cluster-Contribution)

After a self-play game completes, training data incorporates **three reward signals**:

| Reward Signal | Propagation | Who Receives It |
|--------------|-------------|----------------|
| **Immediate** | Per-move only | The move itself |
| **Win/loss** | λ=1 — equal to ALL moves | Every move in the game |
| **Cluster contribution** | TD(λ=0.9) — decays backwards | Only moves that placed triangles IN the final largest cluster |

#### Why TD(λ) for Cluster Contribution?

The win condition is **largest contiguous cluster**. Previously, the network learned to grow *any* cluster, not specifically the one that wins.

With TD(λ), only moves whose triangles are IN the final winning cluster get the cluster bonus. Credit decays going backwards, so **endgame moves get more credit than opening moves**.

#### Implementation

```javascript
const gamma = 0.99;    // Discount factor
const lambda = 0.9;    // Eligibility trace decay
const clusterRewardFactor = 0.1;  // Per-triangle reward

// After game ends, identify triangles in each player's largest cluster
const p1LargestCluster = gridManager.getLargestContiguousCluster(1);
const p2LargestCluster = gridManager.getLargestContiguousCluster(2);

// Build lookup sets
const p1ClusterSet = new Set(p1LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`));
const p2ClusterSet = new Set(p2LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`));

const p1ClusterBonus = p1LargestCluster.length * clusterRewardFactor;
const p2ClusterBonus = p2LargestCluster.length * clusterRewardFactor;

// Process trajectory backwards
let stepsFromEnd = 0;

for (let t = totalMoves - 1; t >= 0; t--) {
    const step = gameHistory[t];
    stepsFromEnd = totalMoves - 1 - t;
    
    const immediateReward = step.rawImmediateReward;  // Raw (de-normalized)
    const winReward = winner === step.player ? 1.0 : -1.0;  // λ=1 for all moves
    
    // Cluster contribution — only if this move's triangle is in the final largest cluster
    let clusterReward = 0;
    const moveKey = `${step.move.x},${step.move.y},${step.move.rotation}`;
    
    if (step.player === 1 && p1ClusterSet.has(moveKey)) {
        const eligibility = Math.pow(gamma * lambda, stepsFromEnd);
        clusterReward = p1ClusterBonus * eligibility;
    } else if (step.player === 2 && p2ClusterSet.has(moveKey)) {
        const eligibility = Math.pow(gamma * lambda, stepsFromEnd);
        clusterReward = p2ClusterBonus * eligibility;
    }
    
    // Combined: immediate + win/loss (λ=1) + cluster (TD λ=0.9)
    const combinedReward = immediateReward + winReward + clusterReward;
    
    // Normalize to [0.01, 0.99]
    const scaled = normalize(combinedReward);
    outputVector[step.actualMoveIndex] = scaled;
}
```

#### Credit Assignment Example

Winner's cluster has 45 triangles, game has 150 moves:

| Move | Position | Steps From End | Cluster Credit |
|------|----------|----------------|----------------|
| Last (IN cluster) | 149 | 0 | 4.50 × 1.000 = **4.50** |
| Late (IN cluster) | 140 | 9 | 4.50 × 0.387 = **1.74** |
| Mid (IN cluster) | 100 | 49 | 4.50 × 0.003 = **0.01** |
| Early (IN cluster) | 10 | 139 | 4.50 × 0.000 ≈ **0.00** |
| Any (NOT in cluster) | — | — | **0.00** |

All moves receive **win reward (+1.0)** equally. Only cluster moves get the additional bonus, with later contributions weighted higher.

---

## Training Pipeline

### Running Training

```bash
# Switch to Node v20 (required for tfjs-node Metal GPU)
fnm use v20.20.2

# Pure network self-play (simplest)
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --output dashboard/data/experiment.json

# Mixed opponents (recommended — curriculum learning)
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 30 \
  --learning-rate 0.1 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix mixed \
  --network-ratio 0.7 \
  --random-ratio 0.3 \
  --audit true \
  --output dashboard/data/experiment.json

# With cluster reward TD(λ)
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --enable-cluster-reward true \
  --output dashboard/data/experiment.json

# Continue from existing checkpoint
node src/server/train_tfjs.cjs \
  --cycles 10 \
  --start-from dashboard/data/TF-B10_network.json \
  --output dashboard/data/continued.json
```

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cycles` | 10 | Number of train-eval cycles |
| `--games` | 100 | Self-play games per cycle |
| `--eval-games` | 100 | Evaluation games per cycle |
| `--iterations` | 30 | Training iterations per cycle |
| `--learning-rate` | 0.1 | SGD learning rate |
| `--exploration` | 0.3 | Epsilon-greedy exploration rate |
| `--self-play-mode` | network | `network` or `random` |
| `--opponent-mix` | network | `network`, `random`, or `mixed` |
| `--network-ratio` | 1.0 | Fraction of network opponents (mixed mode) |
| `--random-ratio` | 0.0 | Fraction of random opponents (mixed mode) |
| `--enable-cluster-reward` | false | Enable TD(λ) cluster reward |
| `--audit` | false | Enable game auditing |
| `--audit-rate` | 0.05 | Fraction of games to audit |
| `--start-from` | null | Path to checkpoint JSON to resume from |
| `--output` | dashboard/data/training.json | Output path for training results |

### Why TensorFlow.js on Mac Studio M2 Ultra

The training pipeline is specifically optimized for the **Apple M2 Ultra** chip:

1. **Metal GPU acceleration**: TF.js uses Apple's Metal framework via `@tensorflow/tfjs-node`
   native bindings, which provides direct GPU compute access. This bypasses CPU
   entirely for tensor operations, yielding ~100x speedup over brain.js (CPU-only).

2. **Unified Memory Architecture**: The M2 Ultra has up to 192GB of unified memory,
   meaning the GPU can access training data without CPU↔GPU memory copies. This
   eliminates the PCIe bus bottleneck that plagues discrete GPU setups.

3. **Performance characteristics** (M2 Ultra, 24-core GPU):
   - Brain.js (CPU): ~8 minutes per 20-cycle experiment
   - TF.js (Metal GPU): ~35 minutes per 20-cycle experiment (with 5x more games/eval)
   - Effective per-game speedup: ~100x

4. **Node v20 LTS requirement**: Node v25's V8 engine breaks tfjs-node native
   bindings. Use `fnm use v20.20.2` before training.

5. **Export path**: Trained TF.js weights can be exported to brain.js-compatible
   JSON format for use in the browser game (Phaser client).

### Training Data Output

Each training run produces a JSON file containing:

```json
{
  "options": { /* all CLI parameters */ },
  "cycles": [
    {
      "cycle": 1,
      "selfPlay": {
        "games": 100,
        "wins": { "1": 48, "2": 48, "draws": 4 },
        "winsByOpponent": { "network": {...}, "random": {...} },
        "opponentTypeCounts": { "network": 70, "random": 30 },
        "scoresByOpponent": { ... },
        "openingDiversity": 6.15,  // Shannon entropy of first moves
        "boardDiversity": 1.95    // Entropy of final board states
      },
      "training": {
        "initialError": 0.014,     // Loss before training on new data
        "finalError": 0.003,       // Loss after training
        "errorSnapshots": [...],   // Per-iteration loss curve
        "iterations": 30,
        "learningRate": 0.1,
        "samples": 17000,
        "timeMs": 5234
      },
      "evaluation": {
        "opponent": "random",
        "aiWinRate": 50,
        "games": 100,
        "timeMs": 842
      }
    }
  ]
}
```

### Game Auditing

When `--audit true` is enabled, a sample of games are fully recorded and
validated against expected invariants:
- Games must play to completion (~170 moves for a 10×10 grid)
- Both players must make moves (no zero-move games)
- Grid must be fully filled at game end
- Final cluster sizes must be consistent

This feature caught the critical `isValidMove` bug that was silently breaking
all training runs.

---

## Monte Carlo Tree Search (MCTS)

MCTS provides strategic lookahead by simulating game continuations and using the neural network to evaluate positions.

### Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 MCTS Search Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   SELECT: Traverse tree using UCT from root to leaf        │
│   EXPAND: Add children with NN policy priors               │
│   EVALUATE: Use value network at leaves                     │
│   BACKPROP: Update Q and N statistics up to root            │
│                                                             │
│   Result: Improved policy from visit counts                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### UCT Formula

```
UCT(a) = Q(a)/N(a) + c × P(a) × √(N(parent)) / (1 + N(a))
         ───────────────   ─────────────────────────────────
         Exploitation:     Exploration:
         "What's proven    "What's promising
                 good?"                but unexplored?"
```

**Parameters:**
- `c`: Exploration constant (default 1.5)
- `P(a)`: Policy prior from neural network
- `Q(a)`: Average value from simulations
- `N(a)`: Visit count

### Key Components

| Component | File | Purpose |
|-----------|------|--------|
| MCTSNode | `MCTS.cjs` | Tree node with Q, N, P statistics |
| MCTS | `MCTS.cjs` | Search controller |
| Dirichlet noise | Root node | Exploration diversity |
| Temperature schedule | Move selection | Opening variety |

### Training Integration

When MCTS is enabled (`--mcts-sims N`), the training target changes:

| Target | Without MCTS | With MCTS |
|--------|--------------|-----------|
| Policy | Raw NN output | Visit distribution π(a) |
| Value | Game outcome | Game outcome (unchanged) |

The network learns to predict: "What move would MCTS choose?"

### Performance Optimization

**Batched Evaluation:** Evaluate multiple leaves in parallel
```javascript
// Sequential (slow): 50 sims × 1 NN call = 50 forward passes
for (let i = 0; i < 50; i++) {
  const value = await network.run(leaf.state);
}

// Batched (fast): 50 sims / batch_size NN calls
for (let batch = 0; batch < batches; batch++) {
  const leaves = collectLeaves(batch_size);
  const values = await network.batchPredict(leaves);  // ONE call
}
```

**Speedup:** 2-4x faster with batch size 8-16

### Configuration Parameters

| Parameter | CLI Flag | Default | Range |
|-----------|----------|---------|-------|
| Simulations | `--mcts-sims` | 0 | 0-1000 |
| Batch size | `--mcts-batch-size` | 8 | 1-32 |
| Exploration c | hardcoded | 1.5 | 0.5-3.0 |
| Dirichlet α | hardcoded | 0.03 | 0.01-0.1 |
| Temperature | hardcoded | 1.0→0.0 | 0.0-2.0 |

### Example Usage

```bash
# Enable MCTS with 50 simulations per move
node src/server/train_tfjs.cjs --cycles 10 --games 100 --mcts-sims 50 --architecture conv

# Quick test with 25 simulations
node src/server/train_tfjs.cjs --cycles 3 --games 10 --mcts-sims 25 --architecture conv
```

### Known Issues (Resolved)

**Value Perspective Bug (Fixed in `28ca800`):**

Prior to this fix, MCTS backpropagation incorrectly handled neural network value outputs. The NN evaluates positions from the perspective of the player whose turn it is at the leaf, but backpropagation expected values from the root player's perspective.

```javascript
// Bug: NN value from leaf.player's perspective passed directly
this._backpropagate(path, nnValue, rootPlayer);

// Fix: Flip value if leaf is opponent's turn
const valueFromRootPerspective = leaf.player === rootPlayer ? nnValue : -nnValue;
this._backpropagate(path, valueFromRootPerspective, rootPlayer);
```

**Impact:** Network trained on inverted values, learning to play badly (30% win rate vs Random).

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| State encoding | O(rows × cols) | < 1 ms |
| Network forward pass | O(weights) | ~5-10 ms |
| Move validation | O(1) per move | < 1 ms |
| Immediate reward calc | O(moves × weight) | ~200-500 ms |
| Training iteration | O(samples × weights) | varies |

### Memory Usage

- Network weights: 160K parameters × 8 bytes ≈ 1.3 MB
- Training data buffer: variable (depends on games stored)
- Per-game state: 401 floats × 200 moves ≈ 320 KB

### Bottlenecks

1. **Immediate reward calculation**: Simulates every legal move
2. **Training iterations**: Can be slow with large datasets
3. **Grid cloning**: JSON.stringify/parse for deep copy

---

## Known Issues & Limitations

### Current Architecture

1. **Heuristic wall at 0%**: Network cannot beat deterministic enclosure heuristic
2. **No spatial reasoning**: Feedforward net can't model toroidal adjacency
3. **No strategic lookahead**: Only immediate reward + final outcome
4. **Overfitting risk**: 30 iterations per cycle is a sweet spot; 100 causes crash-to-zero error

### Brain.js (Deprecated)

Brain.js is still used in the browser game for inference (lightweight, no GPU needed).
Training has fully migrated to TF.js. Key brain.js limitations that motivated the switch:

1. **No GPU acceleration**: All CPU-based
2. **Limited architecture options**: Only feedforward/LSTM
3. **No batch normalization**: Manual scaling required
4. **No learning rate scheduling**: Fixed rate
5. **No multi-head outputs**: Single output vector only
6. **Hard-coded MSE loss**: Can't implement policy+value loss

---

## Planned Architectural Improvements

### Value Head (Actor-Critic)

Add a value prediction output to enable strategic position evaluation:

```
Input: 401
Shared: [256, 256]
  ├─ Policy Head: 400 (move probabilities)
  └─ Value Head: 1 (position value -1 to +1)
```

### Canonical Board Encoding

Rotation/reflection-invariant state representation:
- 8-fold symmetry (4 rotations × 2 reflections)
- Choose canonical form (lexicographically smallest)
- 8× data efficiency from augmented training samples

### Deeper Network

Evaluate [256, 128] vs current [200] for strategic depth.

### Residual Connections

Skip connections to improve gradient flow in deeper networks.

---

## Human vs. AI Telemetry & Supervised / Offline RL Pipeline (`HumanGameLogger`)

To incorporate human strategic play into the training pipeline, the browser/PWA client includes **`HumanGameLogger`** (`game.js` and `src/data/GameLogger.js`):

### Key Design Properties
1. **Strict Pre-Move Snapshot (`capturePreMoveState`):** State vectors are captured *before* `placeTriangle()` and `claimEnclosedTriangles()` mutate the board, preventing action-into-state data leakage.
2. **Dual Player-Relative Encodings:**
   - **`state` (`1,800` floats):** Canonical `10×10×18` one-hot tensor matching `getStateCanonical()` in `src/server/train_tfjs.cjs` (2 diagonal slots × 9 states: `[empty, my_rot0..3, opp_rot0..3]`).
   - **`legacyState` (`401` floats):** Legacy `10×10×4 + 1` vector matching `getStateFixed()` (`rotation / 4.0` normalization + active player flag `1`).
3. **Policy & Value Targets:**
   - **`actionIndex` (`0..399`):** `(move.y * 10 + move.x) * 4 + move.rotation`, mapping 1-to-1 to the 400-logit policy head.
   - **`valueTarget` ($\{-1.0, 0.0, +1.0\}$):** Computed at game end (including explicit `winnerOverride` when a player forfeits) from the perspective of `move.player`.
4. **FIFO LocalStorage Buffer & Flush:** Up to `maxGames = 100` completed games are retained under `localStorage['triantica_human_games']` with ID deduplication and automatic oldest-game eviction on quota limits. Clicking the in-game **📤** button downloads `triantica_human_games_YYYY-MM-DD.json` and flushes the buffer.

---

## References

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [Brain.js Documentation](https://github.com/BrainJS/brain.js) (browser inference only)
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book.html)
- [Phaser 3 Documentation](https://photonstorm.github.io/phaser3-docs/)
