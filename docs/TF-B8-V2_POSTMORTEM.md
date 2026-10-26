# TF-B8-V2 Post-Mortem & Bugfix Plan

**Created:** 2026-10-26 04:40 CET  
**Status:** Awaiting TF-B8-V2 completion before executing fixes  
**Target:** Next experiment run (TF-B9)

---

## 1. Experiment Context

### TF-B8-V2 Configuration
```
Cycles:          10
Games/cycle:     100
MCTS sims/move:  50
Architecture:    Conv (3 Conv2D layers: 32→64→64 filters)
Policy head:     400 outputs, sigmoid activation
Value head:      1 output, tanh activation
Learning rate:   0.05 (momentum=0.9)
Batch size:      256
Replay buffer:   500K samples, 2x replay ratio
Encoding:        Canonical 10×10×18 (getStateCanonical)
Bug fixes:       Sign flip corrected, fixed perspective encoding
Command:         node src/server/train_tfjs.cjs --cycles 10 --games 100 --mcts-sims 50 --architecture conv --output docs/data/TF-B8-V2.json
PID:             83332
Log:             docs/data/TF-B8-V2.log
```

### Final Results (All 10/10 Cycles)
| Cycle | vs Random | vs Heuristic | Loss (end) | Self-play (P1/P2/D) |
|-------|-----------|--------------|------------|----------------------|
| 1     | 35%       | 0%           | 1.312      | 51/47/2              |
| 2     | 42%       | 0%           | 1.311      | 49/49/2              |
| 3     | 24%       | 0%           | 1.316      | 57/42/1              |
| 4     | 48%       | 0%           | 1.310      | 56/41/3              |
| 5     | 25%       | 0%           | 1.309      | 45/51/4              |
| 6     | **54%**   | 0%           | 1.302      | 47/48/5              |
| 7     | 32%       | 0%           | 1.298      | 47/47/6              |
| 8     | 41%       | 0%           | 1.294      | 59/33/8              |
| 9     | **15%** ⚠️| 0%           | 1.300      | 50/46/4              |
| 10    | 38%       | 0%           | 1.299      | 48/46/6              |

**Runtime: 431.6 minutes (~7.2 hours)**  
**vs Random average: ~35.4% (highly unstable, range 15%–54%)**  
**vs Heuristic: 0% across ALL 10 cycles**  
**Loss: decreases smoothly (1.31 → 1.30) — network IS learning, but not useful patterns**

---

## 2. Confirmed Bugs

### Bug #1: Dual-Perspective Training Conflict (CRITICAL)

**Location:** `src/server/train_tfjs.cjs`, MCTS self-play section, lines ~1330–1440  
**Severity:** This is likely the primary cause of the Heuristic Wall

#### What's Wrong

In the MCTS self-play loop, the code alternates which player the "AI" plays as:
```js
const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;  // line ~1336
```

Both players use MCTS with the same network. The state is encoded from the **mover's perspective**:
```js
const state = getStateCanonical(grid, curP);  // line ~1341
```

This means:
- When P1 moves: state has P1's pieces = +1, P2's = -1
- When P2 moves: state has P2's pieces = +1, P1's = -1

The network sees the **same board state as two completely different inputs** depending on whose turn it is. This is fine in AlphaGo Zero because the network has a "color plane" or implicit turn indicator. Here, the final channel of the canonical encoding (channel 17) is **not** used as a turn indicator — it's just padding from the 18-channel structure.

**The critical problem is in training data construction** (lines ~1430–1439):
```js
for (const step of hist) {
  const rewardForPlayer = step.player === 1 ? rewardForWinner : -rewardForWinner;
  allInputs.push(step.state);
  allOutputs.push(step.mctsPolicy);
  allChosenIndices.push(step.moveIdx);
  allValueTargets.push(rewardForPlayer);
}
```

**ALL moves from BOTH players are pushed to training.** Compare with the non-MCTS path (line ~1296):
```js
if (step.player !== aiPlayerNumber) continue;  // Skip opponent moves
```

The non-MCTS path correctly filters to only train on one player's moves. The MCTS path trains on both.

#### Why This Is Destructive

The network receives contradictory training signals:

1. **Game where AI=P1, P1 wins:** All P1 moves get value target +1.0, all P2 moves get -1.0. The state for P1 moves has P1=+1. The state for P2 moves has P2=+1 (flipped encoding). So the network learns: "when my pieces are +1, I should win" AND "when my pieces are +1, I should lose" — for the **same board configuration** but from opposite perspectives.

2. **Game where AI=P2, P2 wins:** Same contradiction, mirrored.

3. The **policy targets** suffer the same issue. MCTS produces a visit distribution π(a) that reflects good play for the mover. But the policy output is a 400-dimensional vector over moves. Move (3,5,rot0) means "place at position (3,5) with rotation 0". The same move index means the same physical action regardless of who plays it. But the state encoding flips who "my pieces" are. So the network must learn: "In state S (P1=+1), move X is good" and "In state S' (P2=+1, which is the same board flipped), move X is bad" — because in the flipped game, the same move helps the other player.

4. **Net effect:** The policy and value gradients from the two perspectives partially cancel each other. The network converges to a mediocre compromise — the "highly unstable 37% average" we observe. Loss decreases because the network IS fitting the training data, but the training data contains conflicting signals, so the fit is a poor compromise.

#### Fix

**Option A (Simple, Recommended for TF-B9):** Only train on one player's moves per game, matching the non-MCTS path:
```js
for (const step of hist) {
  // Only train on AI's moves to avoid dual-perspective conflict
  // (Bug #1 fix: see docs/TF-B8-V2_POSTMORTEM.md for analysis)
  if (step.player !== aiPlayerNumber) continue;
  
  const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
  allInputs.push(step.state);
  allOutputs.push(step.mctsPolicy);
  allChosenIndices.push(step.moveIdx);
  allValueTargets.push(rewardForPlayer);
}
```

This ensures the network always learns from the AI's perspective. The state encoding (from mover's view) and value target (from AI's view) will be consistent because the AI IS the mover for all training samples.

**Option B (Better long-term):** Add a turn indicator channel to the canonical encoding. Add channel 18 as a binary flag: 1.0 if it's "my" turn, 0.0 if opponent's turn. Then train on both players' moves. This doubles training data efficiency but requires modifying `getStateCanonical()` and the conv network's input channels (18→19).

**Option C (AlphaGo Zero style):** Always encode from a fixed player's perspective (e.g., always P1=+1) and add a binary channel indicating whose turn it is. This is how AGZ actually works — the "color plane" tells the network which side it's playing.

**Recommendation:** Start with Option A for TF-B9 (minimal code change, isolates the bug fix). If it works, consider Option B/C for future runs to improve data efficiency.

---

### Bug #2: Leftover Immediate Reward Shaping (MEDIUM)

**Location:** `src/server/train_tfjs.cjs`, function `calculateImmediateRewardVector()`, lines 901–965  
**Severity:** Currently dormant in MCTS-only mode, but dangerous for mixed training

#### What's Wrong

`calculateImmediateRewardVector()` computes shaped rewards:
- Enclosure claims: +0.3 per triangle claimed
- Connected cluster size: +0.1 per connected triangle

These were designed for the old feedforward policy network that needed dense training signal. In the MCTS + value head architecture, they are **counterproductive**:

1. MCTS already provides an improved policy signal (visit distribution π(a)) that implicitly captures territorial value through search.
2. The value head directly predicts game outcome, making the shaped reward redundant.
3. Shaped rewards can create **reward hacking** — the network learns to maximize the proxy reward (claim enclosures early) rather than the true objective (win the game).

#### Current State

In the MCTS self-play path, `calculateImmediateRewardVector()` is **NOT called** — the MCTS path uses `mctsPolicy` as the policy target and game outcome as the value target. This is correct.

However, the non-MCTS self-play loop (lines ~1150–1270) DOES compute and use these rewards. This loop runs when `opponentMix` is not pure network. Currently with `--mcts-sims 50`, the non-MCTS loop is skipped via `continue` at line ~1153.

#### Fix

1. **For TF-B9:** Ensure MCTS-only training (already the case with `--mcts-sims 50`). No code change needed for immediate fix.

2. **For robustness:** Remove the `calculateImmediateRewardVector()` call entirely from the training pipeline, or gate it behind an explicit flag `--use-shaped-rewards` that defaults to `false`. This prevents accidental contamination if someone changes the opponent mix config.

3. **For the non-MCTS path:** Replace shaped rewards with pure outcome-based training: policy target = 1.0 for chosen move (if game won) / 0.0 (if lost), value target = game outcome. This aligns both paths.

```js
// Replace calculateImmediateRewardVector with:
function calculateOutcomePolicyTarget(moveIdx, won) {
  const target = new Array(400).fill(0.01); // small baseline
  target[moveIdx] = won ? 1.0 : 0.0;
  return target;
}
```

---

### Bug #3: Replay Buffer Cross-Contamination (MEDIUM)

**Location:** `src/server/train_tfjs.cjs`, replay buffer section, lines ~1450–1530  
**Severity:** Could silently degrade training quality

#### What's Wrong

The replay buffer stores samples from ALL self-play games regardless of which training path generated them. Currently in MCTS-only mode this isn't an issue. But if the non-MCTS path were re-enabled (e.g., with mixed opponents), the replay buffer would contain:

- MCTS samples: policy target = visit distribution π(a), value target = game outcome
- Non-MCTS samples: policy target = immediate-reward-shaped vector, value target = game outcome

These two sample types have **incompatible policy targets**. The MCTS visit distribution is a probability distribution that sums to ~1. The shaped reward vector is NOT a distribution — it contains arbitrary [0,1] values at each move index representing local reward quality. Training on a mix of these would produce a policy network that can't decide what its output represents.

#### Fix

1. **Tag samples with their type** when adding to the replay buffer: `{ source: 'mcts' | 'shaped' }`.
2. **Filter replay samples** during training to only use samples matching the current training mode.
3. **Or (simpler):** Clear the replay buffer when switching between MCTS and non-MCTS modes. Add a warning log when this happens.

---

### Bug #4: Value Target Range Mismatch (LOW)

**Location:** `src/server/train_tfjs.cjs`, line 1317 and 1437; network architecture line ~390  
**Severity:** Training inefficiency, not a correctness bug

#### What's Wrong

The value head uses `tanh` activation, outputting values in [-1, +1]. The value targets are:
- MCTS path: `rewardForPlayer = step.player === 1 ? rewardForWinner : -rewardForWinner` where `rewardForWinner` is `{0, 1, -1}`. So targets are in {-1, 0, +1}.
- Non-MCTS path: Same range.

This is correct for win/draw/loss outcomes. However, the mean squared error loss between tanh output (which rarely hits exactly ±1) and targets of ±1 creates a constant gradient pressure that never fully converges. AlphaGo Zero uses cross-entropy for the value head (treating it as a binary classification: win probability) rather than MSE.

#### Fix (Future)

Replace value head loss from MSE to binary cross-entropy:
```js
// Instead of:
const valueLoss = valuePred.sub(valueTargetTensor).square().mean();
// Use:
const valueLoss = tf.metrics.binaryCrossentropy(
  valueTargetTensor.mul(0.5).add(0.5),  // Map [-1,+1] → [0,1]
  valuePred.mul(0.5).add(0.5)
).mean();
```

This is a minor optimization — **do NOT include in TF-B9** to keep the experiment isolated to Bug #1 fix only.

---

## 3. Proposed Bugfix Plan: TF-B9

### Goal
Isolate Bug #1 (dual-perspective conflict) as the single variable change from TF-B8-V2. All other code remains identical.

### Changes Required

**File:** `src/server/train_tfjs.cjs`

**Change 1 — MCTS training data filter (lines ~1430–1439):**

Before:
```js
for (const step of hist) {
  const rewardForPlayer = step.player === 1 ? rewardForWinner : -rewardForWinner;
  allInputs.push(step.state);
  allOutputs.push(step.mctsPolicy);
  allChosenIndices.push(step.moveIdx);
  allValueTargets.push(rewardForPlayer);
}
```

After:
```js
for (const step of hist) {
  // Only train on AI's moves to avoid dual-perspective conflict
  // (Bug #1 fix: see docs/TF-B8-V2_POSTMORTEM.md for analysis)
  if (step.player !== aiPlayerNumber) continue;
  
  const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
  allInputs.push(step.state);
  allOutputs.push(step.mctsPolicy);
  allChosenIndices.push(step.moveIdx);
  allValueTargets.push(rewardForPlayer);
}
```

**Key differences from current code:**
1. Added `if (step.player !== aiPlayerNumber) continue;` — mirrors the non-MCTS path
2. Changed value target from `step.player === 1 ? rewardForWinner : -rewardForWinner` to `winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0)` — directly from AI's perspective, no P1/P2 mapping needed since we've already filtered to AI's moves only
3. Both changes ensure state encoding (from mover=AI's perspective) and value target (from AI's perspective) are always aligned

**No other changes.** Specifically:
- Same conv architecture
- Same MCTS-50 configuration
- Same learning rate (0.05), batch size (256)
- Same replay buffer settings
- Same canonical encoding (18 channels)

### Execution Command
```bash
cd ~/.openclaw/workspace/triantica
node src/server/train_tfjs.cjs --cycles 10 --games 100 --mcts-sims 50 --architecture conv --output docs/data/TF-B9.json
```

### Expected Runtime
~8 hours on M2 Ultra (same as TF-B8-V2, since MCTS search dominates and training data is halved but batch count stays similar)

### Expected Outcome

If Bug #1 is the primary cause of the Heuristic Wall:
- **vs Random:** Should stabilize above 50% with upward trend (currently oscillates 24%–54%)
- **vs Heuristic:** Should show >0% win rate by cycle 3–5 (currently stuck at 0%)
- **Loss:** Should decrease more meaningfully as training signal is no longer contradictory

If Bug #1 is NOT the primary cause:
- Results will look similar to TF-B8-V2 (unstable ~38% vs Random, 0% vs Heuristic)
- This would point to a deeper issue in the MCTS implementation, reward structure, or the game itself being too complex for the current architecture
- Next investigation: audit MCTS.cjs for search bugs, test with much higher simulation counts (200+), or simplify the game

### Validation Checks

After TF-B9 completes, verify:
1. **Training sample count:** Should be roughly half of TF-B8-V2 per cycle (since we're skipping opponent moves). If it's the same, the filter isn't working.
2. **Value target distribution:** Should be exactly {1.0, 0.0, -1.0}. Check for intermediate values that would indicate the old reward computation is still leaking in.
3. **Self-play win balance:** P1/P2 wins should be roughly equal since AI alternates roles. A persistent P1 advantage would suggest the filter is biased.
4. **Policy target sanity:** MCTS visit distributions should have a clear peak at the best move. Flat distributions would indicate MCTS is not confident, which could mean the network is too weak to provide useful evaluations.
5. **Compare with TF-B8-V2 final results:** Once TF-B8-V2 finishes, update the dashboard with all 10 cycles and use as direct baseline.

---

## 4. Deferred Improvements (Post-TF-B9)

These are NOT included in TF-B9 to maintain experimental isolation:

| ID | Improvement | Rationale | Priority |
|----|-------------|-----------|----------|
| D1 | Add turn indicator channel (canonical 18→19) | Enables training on both players' moves, doubles data efficiency | High |
| D2 | Remove `calculateImmediateRewardVector()` | Eliminate shaped reward contamination risk | Medium |
| D3 | Tag replay buffer samples by source | Prevent cross-contamination of training signals | Medium |
| D4 | Value head: MSE → binary cross-entropy | Better convergence for [-1,+1] outcomes | Low |
| D5 | Increase MCTS simulations (50→200) | More search = better policy targets, but 4x slower | Low |
| D6 | Reduce learning rate (0.05→0.01) | May stabilize training, slower convergence | Low |
| D7 | Increase network capacity (64→128 filters) | More representational power for complex patterns | Low |

---

## 5. Workflow Checklist

Pre-TF-B9 execution steps:

- [ ] Wait for TF-B8-V2 PID 83332 to complete
- [ ] Read full TF-B8-V2.log and extract cycles 9–10 results
- [ ] Update `docs/data/TF-B8-V2.json` dashboard with final data
- [ ] Update MEMORY.md with complete TF-B8-V2 results (all 10 cycles)
- [ ] Git commit and push all changes to `https://github.com/augusto-espinel/Triantica`
- [ ] Apply Bug #1 fix to `src/server/train_tfjs.cjs`
- [ ] Verify fix with a short dry-run (1 cycle, 10 games)
- [ ] Launch TF-B9 experiment
- [ ] Set cron to check TF-B9 completion