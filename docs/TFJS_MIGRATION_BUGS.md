# TF.js Migration Bug Analysis

## Executive Summary

The original `trainedAI_network.json` achieves **91% win rate vs Random** as expected.
The TF.js training experiments (B1, B2, B3, TF-C3) only achieved ~43-49% because of a
**state representation mismatch** with the original brain.js training.

---

## Bug #1: State Representation Mismatch (CRITICAL)

### Original AIPlayer.js (brain.js)
```javascript
normalizePlayer = (p) => p === this.playerNumber ? 1 : -1;
```
- **FIXED perspective**: State always shows AI's pieces as 1, opponent as -1
- AI's `playerNumber` is constant (e.g., always P1)
- State encoding is consistent across all training examples

### train_tfjs.cjs (TF.js)
```javascript
pv = t => t ? (t.player === curPlayer ? 1 : -1) : 0;
```
- **RELATIVE perspective**: State shows current player's pieces as 1
- Perspective FLIPS every turn
- State encoding changes based on whose turn it is

### Impact
| Turn | Board State | Original (Fixed P1) | TF.js (Relative) |
|------|-------------|---------------------|-------------------|
| P1   | P1=3, P2=5  | P1=1, P2=-1         | P1=1, P2=-1       |
| P2   | P1=3, P2=5  | P1=1, P2=-1         | P2=1, P1=-1 ❌    |

The networks trained by TF.js learn a different state encoding than the original
brain.js network, making direct comparison impossible.

---

## Bug #2: Normalization Constants (FIXED)

### Original AIPlayer.js
```javascript
const minImmediateReward = 0.01;
const maxImmediateReward = 0.99;
const minCombinedReward = minImmediateReward - 1.0; // -0.99
const maxCombinedReward = maxImmediateReward + 1.0; // 1.99
```

### train_tfjs.cjs (CORRECTED)
```javascript
const minImmediateReward = 0.01;
const maxImmediateReward = 0.99;
const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight);
const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight);
```
✅ This is now correct.

---

## Bug #3: Reward Signal Calculation (FIXED)

### Original AIPlayer.js
```javascript
const immediateRewardActualMove = immediateRewardVector[actualMoveIndex];
const combinedRewardActualMove = immediateRewardActualMove + scaledFinalReward;
```

### train_tfjs.cjs (CORRECTED)
```javascript
const chosenMoveReward = immediateRewardVector[moveIdx];
const combinedReward = step.chosenMoveReward + rewardForPlayer * finalRewardWeight * discountFactor;
```
✅ This is now correct.

---

## Bug #4: Training Iterations (CHECK)

### Original train.js (brain.js)
```bash
--iterations 200  # Default: 200
```

### train_tfjs.cjs (TF.js)
```bash
--iterations 10   # Default: 10 (20x less!)
```
⚠️ This could contribute to undertraining.

---

## Recommended Fix

### Option A: Use Fixed Perspective (Match brain.js exactly)
Change `getStateRelative` in train_tfjs.cjs:
```javascript
function getStateFixed(grid, aiPlayerNumber) {
  const state = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      const cell = grid.grid[y][x];
      const t1 = cell.triangles[0], t2 = cell.triangles[1];
      state.push(t1?.player === aiPlayerNumber ? 1 : (t1?.player ? -1 : 0));
      state.push(t1?.rotation ?? -1);
      state.push(t2?.player === aiPlayerNumber ? 1 : (t2?.player ? -1 : 0));
      state.push(t2?.rotation ?? -1);
    }
  }
  state.push(1); // Always from AI's perspective
  return state;
}
```

### Option B: Keep Relative Perspective (Better for self-play)
Train a fresh network from scratch using the relative perspective.
This allows the network to learn symmetric play (can play both sides).
Don't expect it to match the brain.js network's weights.

---

## Verification Steps

1. **Confirm original network works**: ✅ 91% vs Random
2. **Confirm TF.js evaluation works**: ✅ Loaded brain.js network in TF.js
3. **Identify state mismatch**: ✅ Fixed vs Relative perspective
4. **Fix and retrain**: Apply fix and run baseline experiments

---

## Conclusion

The normalization and scoring scheme in train_tfjs.cjs is **correct** after the previous fixes.
The remaining issue is the **state representation mismatch** between brain.js (fixed perspective)
and TF.js (relative perspective). This is a design choice, not a bug, but it means:

1. Networks trained with TF.js cannot be directly compared to brain.js networks
2. To match brain.js behavior, use fixed perspective
3. To enable symmetric self-play, use relative perspective but train from scratch
