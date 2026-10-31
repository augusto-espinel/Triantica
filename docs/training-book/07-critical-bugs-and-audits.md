> **[📚 Training Book Index](./README.md)** | [← Chapter 6: Training Strategies](./06-training-strategies.md) | [Chapter 8: The Heuristic Wall →](./08-the-heuristic-wall.md)

---

## 7. Critical Bugs, Forensic Audits, and Why They Matter {#7-critical-bugs}

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

**Generalizable Principle (Canonical Ego-Centric State):** In symmetric multi-agent RL, the neural network must always view the state from the active agent's perspective (`+1` = self, `-1` = opponent), never in global player coordinates (`Player 1` vs `Player 2`).

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

**Generalizable Principle (Action-Specific Credit Assignment):** Only the action actually executed (or the target MCTS distribution) should drive the policy gradient; averaging over unselected outputs destroys signal-to-noise ratio.

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

**Generalizable Principle (Role-Swap Invariance):** Whenever an evaluation harness alternates first-mover and second-mover seats for fairness, opponent identity must be dynamically bound to `(3 - aiPlayerNumber)`.

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

**Generalizable Principle (Decouple Seat Index from Agent Identity):** Metrics must always be keyed by agent identity (`ai` vs `opponent`), while separately tracking role-split performance (`as P1` vs `as P2`) to detect first-mover bias.

### Bug 5: MCTS Value Perspective Mismatch (TF-B8)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Training converges smoothly (loss: 0.99 → 1.00)          │
│   But win rate degrades: 40% → 30% → 26%                   │
│                                                             │
│   Network IS learning — just learning to LOSE!              │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Neural network returns values from the perspective of the player whose turn it is at the leaf node, but MCTS backpropagation expected values from the root player's perspective.

```javascript
// WRONG: Pass NN value directly to backpropagation
const { value: nnValue } = await this.network.run(leafState);
leaf.expand(leafPolicy);
leafValue = nnValue;  // From leaf.player's perspective!
this._backpropagate(path, leafValue, rootPlayer);  // Expects rootPlayer's perspective!

// CORRECT: Flip value if from opponent's perspective
const valueFromRootPerspective = leaf.player === rootPlayer ? leafValue : -leafValue;
this._backpropagate(path, valueFromRootPerspective, rootPlayer);
```

**Why This Caused 30% Win Rate:**
- When P2's position was bad (NN returns -0.7), search thought it was good (+0.7)
- "Bad" positions got high Q-values → high visit counts → trained as good moves
- Network learned: "Seek positions where NN outputs negative values"
- Result: systematic losing behavior

**How we found it:** TF-B8 showed systematic degradation pattern:
- Started: 40% vs Random
- Peak: 42% (cycle 2)
- Worst: 26% (cycle 6)
- Final: 30% (cycle 10)

The smooth loss curve + consistent degradation = inverted learning signal.

**Key Insight:** "Consistently getting worse is more encouraging than consistently remaining random" — it means the learning mechanism works, just the signal was flipped.

---

### Part II: The Forensic Cross-Codebase Audit (Bugs 6–11)

Following the completion of `TF-B8-V2` (which still oscillated between 15% and 54% vs Random and scored 0% vs Heuristic despite fixing Bug 5), we conducted a complete line-by-line audit across the **Browser Game (`game.js`, `AIPlayer.js`)**, **Core Engine (`src/core/GridManager.js`)**, **MCTS & Training (`src/server/MCTS.cjs`, `src/server/train_tfjs.cjs`, `src/server/ReplayBuffer.cjs`)**, and **Evaluation (`eval_tfjs.cjs`, `scripts/evaluate.cjs`)**.

We uncovered **six severe systemic bugs** spanning game geometry, tree search physics, tensor encoding, and gradient calculus:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 THE 6 SYSTEMIC BUGS UNCOVERED IN PHASE 5                     │
├──────────┬───────────────────────────────────┬───────────────────────────────┤
│ Bug ID   │ Component                         │ Systemic Impact               │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 6    │ Diagonal Overlap Check Inverted   │ Server & MCTS played a mutated│
│          │ (`src/core/GridManager.js`)       │ game where triangles crossed! │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 7    │ MCTS Blind to Enclosures & Bonus  │ MCTS tree never claimed cells │
│          │ Turns (`MCTSNode.expand()`)       │ or granted extra turns!       │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 8    │ Canonical Encoding Dropped 2nd    │ Full cells hid 50% of their   │
│          │ Triangle (`getStateCanonical`)    │ triangles from the ConvNet!   │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 9    │ ReplayBuffer Zeroed Value Targets │ 67% of batches trained the    │
│          │ (`ReplayBuffer.cjs`)              │ Value Head to predict 0 (draw)│
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 10   │ Dual-Perspective Gradient Clash   │ Opponent moves in self-play   │
│          │ (`train_tfjs.cjs`)                │ destabilized role updates     │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 11   │ 1/400 Policy Loss Dilution &      │ Policy gradients 400× too weak│
│          │ Value `tanh` Saturation           │ (`0.45–0.54`); Value locked!  │
└──────────┴───────────────────────────────────┴───────────────────────────────┘
```

#### Bug 6: Inverted Diagonal Overlap Rule (Engine Split-Brain)

**Geometry of a Triantica Cell:**
Each cell `[x, y]` is split by one of two diagonals:
- **Group 0 (`╱` slash diagonal):** Rotations `0` (Top-Left `◤`) and `2` (Bottom-Right `◢`)
- **Group 1 (`╲` backslash diagonal):** Rotations `1` (Bottom-Left `◣`) and `3` (Top-Right `◥`)

```
       GROUP 0 (╱ Diagonal)                  GROUP 1 (╲ Diagonal)
   ┌───────────┐   ┌───────────┐         ┌───────────┐   ┌───────────┐
   │ ◤       ╱ │   │         ╱ │         │ ╲         │   │ ╲       ◥ │
   │   0   ╱   │ + │       ╱   │ = FULL  │   ╲       │ + │   ╲   3   │ = FULL
   │     ╱     │   │     ╱   2 │   CELL  │ 1   ╲     │   │     ╲     │   CELL
   │   ╱       │   │   ╱       │         │ ◣     ╲   │   │       ╲   │
   └───────────┘   └───────────┘         └───────────┘   └───────────┘
   Valid 2nd triangle: SAME diagonal group (newGroup === existingGroup)
                       OPPOSITE rotation   (rot !== existingRot)
```

**What Was Wrong:**
When the core board state & rules engine (`src/core/GridManager.js`) was extracted for server training, the boolean negation `!` was accidentally dropped from the overlap check:
```javascript
// WRONG (in the core rules engine, src/core/GridManager.js & game.js:isValidMove):
if (newGroup === existingGroup || cell.triangles[0].rotation === rotation) {
  return false; // ❌ Forbids sharing the diagonal; forces 50% physical overlap!
}

// CORRECT (Consolidated in the core rules engine, GridManager.isValidMove & placeTriangle):
const existingRot = cell.triangles[0].rotation;
const existingGroup = this.getDiagonalGroup(existingRot);
const newGroup = this.getDiagonalGroup(rotation);
if (newGroup !== existingGroup || existingRot === rotation) {
  return false; // ✅ Must share the diagonal and have opposite rotation
}
```
**Why It Mattered:** Every server training run (`TF-B1` through `TF-B8-V2`) and MCTS search was playing a **mutated game** where two triangles in the same cell had to cross diagonals (overlapping by 50%), while the browser placement routine (`game.js:placeTriangle`) enforced the real rule! Furthermore, we consolidated the neighbor adjacency logic (`getAdjacentTriangles()`) so two triangles in the same cell only form a contiguous cluster if they share the diagonal edge (`newGroup === existingGroup && t.rotation !== rotation`).

#### Bug 7: MCTS Was Completely Blind to Enclosures & Extra Turns

**What Was Wrong:**
In the Monte Carlo Tree Search engine (`src/server/MCTS.cjs`), the tree node expansion step (`MCTSNode.expand()`) advanced the board state using raw triangle placement (`placeTriangle()`) and unconditionally flipped `currentPlayer`:
```javascript
// WRONG (MCTS tree expansion, MCTSNode.expand, before fix):
const newGrid = this.grid.clone();
newGrid.placeTriangle(move.x, move.y, move.rotation, this.player);
newGrid.currentPlayer = this.player === 1 ? 2 : 1;
// ❌ Never called claimEnclosedTriangles(this.player)!
// ❌ Never granted extraTurns += Math.floor((1 + claimed) / 2)!
```
**Why It Mattered:** Inside MCTS lookahead, **enclosing territory and earning bonus turns literally did not exist!** MCTS could not see that completing a boundary captures enclosed triangles and grants an immediate extra turn, nor could it see when it left an enclosure combo open for the `HeuristicPlayer`.
- **The Fix (Unified Move Execution, `grid.executeMove`):** We consolidated move placement, enclosure claiming (`claimEnclosedTriangles(pl)`), and bonus turn calculation (`extraTurns += Math.floor((1 + claimed) / 2)`) into a single atomic engine transition method (`GridManager.prototype.executeMove(x, y, rotation, player)`) and called it everywhere across tree expansion, training, game sessions, and evaluation (`MCTSNode.expand`, `train_tfjs.cjs`, `GameSession.js`, and evaluation scripts). Immediately after fixing this, AI cluster sizes jumped by **+60% to +85%** (from ~19 to ~35–40)!

#### Bug 8: Canonical 18-Channel Encoding Dropped the 2nd Triangle in Every Full Cell

**What Was Wrong:**
The canonical tensor encoder (`getStateCanonical`) originally iterated `for (let group = 0; group < 2; group++)` and called `cell.triangles.find(t => grid.getDiagonalGroup(t.rotation) === group)`. Once Bug 6 was fixed, both triangles in a full cell belong to the **same** diagonal group (e.g., rotations `0` and `2` are both Group 0). Thus `.find()` only ever returned the first triangle and left the second 9-channel slot empty—**blinding the ConvNet to 50% of all triangles in completed cells!**
- **The Fix (Canonical State Encoder, `grid.getCanonicalState`):** We indexed the two 9-channel slots by **rotation half** (`Slot 0: rotation < 2` i.e. $\{0, 1\}$, `Slot 1: rotation >= 2` i.e. $\{2, 3\}$). Since a valid cell can hold at most one triangle from $\{0, 1\}$ and at most one from $\{2, 3\}$, this guarantees **zero collisions, 100% visibility of both triangles, and strict move-order invariance**.

#### Bug 9: Replay Buffer Zeroed Out All Historical Value Targets

**What Was Wrong:**
In the self-play training pipeline (`train_tfjs.cjs`), the experience replay module (`ReplayBuffer.cjs`) stored `(input, output, chosenIndex)` but omitted `valueTarget`. During replay training:
```javascript
// WRONG (in the self-play training pipeline, train_tfjs.cjs, before fix):
const replayValueTargets = new Array(replayBatchSize).fill(0); // ❌ Forces Value Head to 0!
```
With `replayRatio = 2.0`, **67% of all gradient updates trained the Value Head to predict `0.0` (draw) on every board position**, destroying its ability to evaluate winning vs losing states!
- **The Fix:** Updated the experience replay module (`ReplayBuffer.cjs`) to store and sample `valueTarget` (`+1`, `0`, `-1`) alongside state and policy.

#### Bug 10: Dual-Perspective Gradient Conflict in MCTS Self-Play

**What Was Wrong:**
While the non-MCTS training loop filtered training samples to the designated AI player (`if (step.player !== aiPlayerNumber) continue;`), the MCTS training loop pushed **both** players' moves from the same self-play game into the training batch.
- **The Fix:** Enforced `if (step.player !== aiPlayerNumber) continue;` (with `aiPlayerNumber` alternating between P1 and P2 every game) and added explicit **P1 (First Mover) vs P2 (Second Mover) win-share tracking** across all training logs and dashboard charts.

#### Bug 11: 1/400 Policy Loss Dilution & Value Head `tanh` Saturation (The `TF-B8` → `TF-B9` Breakthrough)

**What We Saw After Re-Running `TF-B8` With Bugs 6–10 Fixed:**
Win rate vs Random stabilized at **43.0% avg (47.0% in Cycle 10)** with zero collapses and large clusters (~37.1 mean), yet the AI still didn't dominate Random (expected 70%+). Inspecting the trained `TF-B8_network.json` weights revealed why:
- Policy Head outputs across all 400 moves were stuck between **`0.450` and `0.545`** (barely moved from `0.500` random initialization!).
- Value Head output was saturated at **`-1.0000`** on every board state!

**Root Cause (Mathematical Proof):**
1. **1/400 Policy Gradient Dilution:**
   In `TFConvNetwork.trainBatchWithIsolation()`, policy loss was computed on the `[256, 400]` tensor using `.mean()`:
   $$\mathcal{L}_{\text{policy}}^{\text{bug}} = \frac{1}{256 \times \mathbf{400}} \sum_{b=1}^{256} \sum_{a=1}^{400} \text{CE}(\pi_{b,a}, \hat{p}_{b,a})$$
   Because the MCTS target distribution $\sum_{a=1}^{400} \pi_{b,a} = 1.0$ already sums to $1$ across the 400 actions, dividing by `400` again **diluted the policy gradient by $400\times$**!
2. **Value Head Saturation (`tanh` + `LR=0.05, momentum=0.9`):**
   Meanwhile, the Value Head `[256, 1]` MSE loss was only divided by `256`—receiving a **$400\times$ stronger gradient** than the Policy Head! Combined with an aggressive `momentum(0.05, 0.9)` optimizer, the first gradient step pushed the Value Head pre-activation $z$ to $|z| > 8$, where $\tanh(z) = \pm 1.0$ and the derivative $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) \approx 0$. Once saturated at `-1.0`, the Value Head's gradient vanished permanently!

```javascript
// WRONG (Diluted Policy Loss + Saturated Value Head):
const bce = policyTargetTensor.mul(pClip.log())
  .add(tf.scalar(1).sub(policyTargetTensor).mul(tf.scalar(1).sub(pClip).log()))
  .neg().mean(); // ❌ Divides by (BatchSize * 400) -> 400x weaker gradient!
const optimizer = tf.train.momentum(0.05, 0.9); // ❌ Overshoots tanh into saturation!

// CORRECT (Fixed in TF-B9):
const logits = pClip.div(tf.scalar(1).sub(pClip)).log();
const logProbs = tf.logSoftmax(logits, -1);
// ✅ Sum over the 400 actions (-1), then average over the batch (256):
const policyLoss = policyTargetTensor.mul(logProbs).neg().sum(-1).mean();
// ✅ Persistent Adam optimizer (LR=0.001) prevents tanh saturation:
if (!this._optimizer || this._optimizerLR !== adamLR) {
  this._optimizer = tf.train.adam(adamLR);
  this._optimizerLR = adamLR;
}
```

**Live Verification Impact:**
In a 20-game live verification test immediately after applying Bug 11's fix:
- Policy Head outputs cleanly separated legal high-visit MCTS moves (**`0.985`**) from poor/illegal moves (**`0.000`**).
- Value Head remained unsaturated and calibrated (`+0.2538` on mid-game boards).
- Win rate vs Random jumped from `35–42%` to **`56.0%`** (`52% as P1`, `60% as P2`) in Cycle 1 alone!

#### Bug 12: Browser Train/Test Distribution Mismatch (90% Heuristic Override in the Browser Game Loop (`game.js`))

**What We Saw (2026-10-22 Night Audit):**
When watching "AI vs AI" in the browser game controller (`game.js`), both AIs appeared to play balanced, highly tactical games with large enclosure cascades—yet when we tested the exact same trained `brain.js` network in the server evaluation harness (`evaluate.cjs`), it lost `0–100` against the deterministic heuristic opponent (`HeuristicPlayer`)!

```
┌──────────────────────────────────────────────────────────────────────────────┐
│          THE BROWSER "ILLUSION OF COMPETENCE" (TRAIN/TEST MISMATCH)          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Move Selection Source     │ Browser AI (game.js)  │ Server Training/Eval    │
│  ──────────────────────────┼───────────────────────┼─────────────────────────│
│  Neural Network Prediction │  10% of moves! ❌      │  80% (Train) / 100% Eval│
│  Hardcoded Greedy Heuristic│  90% of moves! ❌      │  10% (Train) /   0% Eval│
│  Random Exploration        │   0% of moves         │  10% (Train) /   0% Eval│
│                                                                              │
│  Impact: Watching "AI vs AI" in the browser was actually watching            │
│          Heuristic vs Heuristic 90% of the time—masking the network's        │
│          true behavior and hiding the P2 snowball asymmetry!                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **The Fix (`76cece8` & Phase 6 Consolidation):** Aligned browser `AIPlayer` move selection directly with the pure neural network policy (`100%` network prediction during evaluation/play, with optional 1-ply safety filter toggle). Once the 90% heuristic crutch was removed, the browser immediately revealed the true state of the network—and later allowed us to verify `TF-B15`'s genuine `99%` raw / `100%` +Safety mastery in real time.

#### Bug 13: The Standalone Evaluator Role-Swap Bug (`eval_tfjs.cjs` — Why Every Model Looked Locked at `50.0%`)

**What We Saw (2026-10-28 Audit of `TF-B11` / `TF-B12` Standalone Evals):**
During training inside the self-play training pipeline (`train_tfjs.cjs`), `TF-B12` achieved **`100.0%` vs Random** and **`63.0%` vs Heuristic**. Yet when we ran the standalone CLI evaluation harness (`eval_tfjs.cjs`) on the saved checkpoint, it reported **exactly `50.0%` (`50W / 50L`)** across every single model—even when evaluating against `RandomPlayer`!

**Root Cause:**
In the standalone evaluation harness (`eval_tfjs.cjs`), the evaluation loop alternated `aiPlayer = (i % 2 === 0) ? 1 : 2` for seat fairness, but instantiated the game session with fixed player objects `(aiPlayerObj, opponentObj)` without swapping their seats when `aiPlayer === 2`:
```javascript
// WRONG (in the standalone evaluation harness, eval_tfjs.cjs, before fix):
const aiPlayer = (i % 2 === 0) ? 1 : 2;
// ❌ Passed (ai, opponent) in fixed order regardless of aiPlayer!
// In odd games (aiPlayer === 2), the AI played as P1 while the harness
// credited P2's wins to the AI!
```
Because P1/P2 wins were inverted on every odd-numbered game (`50` games normal + `50` games inverted), any deterministic win rate $w$ was averaged with $(1 - w)$:
$$\text{Reported Win Rate} = \frac{1}{2} w + \frac{1}{2}(1 - w) = \mathbf{50.0\%}$$
- **The Fix:** Consolidated all evaluation logic across the training pipeline, standalone CLI evaluator, and tournament runner (`train_tfjs.cjs`, `eval_tfjs.cjs`, and `scripts/evaluate.cjs`) to bind player seats strictly by `aiPlayerNumber`, enforce **2-ply random opening seeding** (`1` random legal move per player so 100-game evals test 100 distinct board openings rather than 1 repeated deterministic trajectory), and report explicit **`as P1` vs `as P2` win splits**.

#### Bug 14: The Subclass Inheritance Bypass (`search()` vs. `searchBatched()` in `QuiescenceMCTS.cjs`)

**What We Saw (2026-10-27 Pre-`TF-B12` Audit):**
When we first implemented the Quiescence MCTS engine (`src/server/QuiescenceMCTS.cjs`) as a subclass of the standard Monte Carlo Tree Search engine (`src/server/MCTS.cjs`), unit tests calling `qMcts.search(grid, 50)` passed—yet during full self-play training runs, the AI still played into 1-move enclosure traps and failed to discover enclosing trains!

**Root Cause:**
In the base MCTS engine (`MCTS.cjs`), there are two distinct entry points:
1. `search(grid, simulations)` — single-simulation sequential loop (used for unit debugging).
2. `searchBatched(grid, simulations)` $\to$ `_runBatch(root, batchSize, rootPlayer)` — virtual-loss batched GPU evaluation loop (called by the self-play training pipeline, `src/server/train_tfjs.cjs`, for M2 Ultra speed).

The initial `QuiescenceMCTS` implementation overrode `search()`, **but forgot to override `searchBatched()` and `_runBatch()`!** Consequently, when the training pipeline (`train_tfjs.cjs`) invoked `mcts.searchBatched(grid, 50)`, JavaScript's prototype chain silently inherited `MCTS.prototype.searchBatched` and `MCTS.prototype._runBatch`, **bypassing quiescence rollouts and root risk shaping 100% of the time during batched training!**
- **The Fix & Unit Verification (`test_quiescence.cjs`):** We overrode both `searchBatched()` and `_runBatch()` in the Quiescence MCTS engine (`src/server/QuiescenceMCTS.cjs`) so that pre-cascade expansion states and post-cascade quiescent states are packed into a single combined `network.batchPredict(allStates)` GPU call, and created a dedicated regression test suite (`test_quiescence.cjs`) verifying that `searchBatched()` selects enclosing moves and avoids hole traps even under a completely flat (`0.5` uniform) network prior.

---

> [← Chapter 6: Training Strategies](./06-training-strategies.md) | **[📚 Training Book Index](./README.md)** | [Chapter 8: The Heuristic Wall →](./08-the-heuristic-wall.md)
