

---

## Experiment 12: Curriculum Learning

**Started:** 2026-10-20 17:02 (v2 with player-switching fix)
**Completed:** 2026-10-20 20:56
**Log:** `experiments/logs/exp12_v2.log`

**Hypothesis:** The network needs a stable, competent opponent to learn from — not random (too easy) or self (circular equilibrium).

**Approach:** Mixed opponent training with HeuristicPlayer:
- 40% games vs Heuristic (competent, deterministic)
- 40% games vs Network (self-play for improvement)
- 20% games vs Random (maintain diversity)

### Critical Bug Fix (2026-10-20 19:01)

**Bug:** `_runGameVsOpponent()` never switched `gridManager.currentPlayer` after each move. P1 placed all 200 triangles, P2 placed zero.

**Fix:** Added player alternation at line 267:
```javascript
gridManager.currentPlayer = currentPlayer === 1 ? 2 : 1;
```

**Verification:**
- Before fix: P1 median=198, P2 median=0
- After fix: P1 median=57-63, P2 median=88-94

### Final Results

| Cycle | Win Rate | AI Med | Rand Med | Gap | Note |
|-------|----------|--------|----------|-----|------|
| 1 | 49.0% | 33 | 32 | +1 | |
| 2 | 54.5% | 40 | 37 | +3 | |
| 3 | 51.0% | 36 | 35 | +1 | |
| 4 | 49.0% | 37 | 39 | -2 | |
| 5 | 49.5% | 36 | 36 | 0 | |
| 6 | 53.5% | 39 | 36 | +3 | |
| 7 | 52.5% | 38 | 34 | +4 | |
| 8 | 45.5% | 37 | 38 | -1 | |
| 9 | 47.0% | 36 | 38 | -2 | |
| 10 | 48.0% | 39 | 38 | +1 | |
| 11 | **57.5%** | 43 | 36 | +7 | ✅ Peak |
| 12 | 49.0% | 36 | 36 | 0 | |
| 13 | 50.5% | 38 | 39 | -1 | |
| 14 | 50.0% | 38 | 38 | 0 | |
| 15 | 49.0% | 38 | 39 | -1 | |
| 16 | 52.0% | 39 | 36 | +3 | |
| 17 | 50.0% | 37 | 38 | -1 | |
| 18 | **55.5%** | 40 | 37 | +3 | ✅ |
| 19 | 54.0% | 38 | 36 | +2 | |
| 20 | **56.0%** | 43 | 36 | +7 | ✅ End |

**Starting win rate:** 49.0%
**Final win rate:** 56.0%
**Improvement:** +7.0%

**Conclusion:** Curriculum learning showed promise with peak at 57.5% (cycle 11), but oscillation pattern suggests fundamental limitation persists.

---

## Experiment 12b: Curriculum Continuation

**Started:** 2026-10-20 20:02
**Completed:** 2026-10-20 22:14
**Log:** `experiments/logs/exp12b.log`

**Hypothesis:** The 56% from Exp 12 might continue improving with more training.

**Approach:** Pure continuation of Exp 12 checkpoint — same config, 20 more cycles.

### Final Results

| Cycle | Win Rate | Trend |
|-------|----------|-------|
| 1 | 51.0% | |
| 2 | 50.5% | |
| 3 | 49.5% | |
| 4 | 56.0% | Spike |
| 5 | 51.0% | ↓ |
| 6 | 48.0% | ↓ |
| 7 | 49.0% | |
| 8 | 55.0% | Spike |
| 9 | 47.5% | ↓ |
| 10 | 50.5% | |
| 11 | 53.0% | |
| 12 | 46.5% | ↓ |
| 13 | 50.5% | |
| 14 | 49.5% | |
| 15 | 49.0% | |
| 16 | 53.5% | |
| 17 | 54.0% | |
| 18 | 44.0% | ↓ |
| 19 | 47.0% | |
| 20 | 47.5% | |

**Starting win rate:** 51.0%
**Final win rate:** 47.5%
**Change:** -3.5%

**Conclusion:** The 56% was noise. No sustained improvement — network oscillates in 44-56% range with mean ~50.5%. Curriculum alone is not the solution. Backup saved to `experiments/backups/exp12b/`.

**Key insight:** Training converges perfectly (error → 0.0009) but win rate doesn't improve. This confirms Gemini's gradient dilution diagnosis: network learns heuristic prediction, not strategic evaluation.

---

## Experiment 13a: Gradient Isolation

**Status:** Implemented (2026-10-20)

**Hypothesis:** The network is learning to predict 1-step greedy heuristic for 399/400 outputs, not the win/loss signal. This dilutes the learning signal by 399:1.

**Diagnosis (from Gemini analysis):**
- Target vectors initialize unchosen moves with immediate reward heuristic
- Network computes loss across all 400 outputs
- 99.75% of gradient forces memorization of heuristic, not win/loss

**Implementation:** Self-anchored target masking

```javascript
// Get current network prediction for unchosen moves
const currentPrediction = this.network.run(step.state);

// Build target: anchor unchosen moves to current prediction (zero gradient)
for (let i = 0; i < 400; i++) {
  if (i === step.actualMoveIndex) {
    finalOutputVector[i] = scaledCombined; // chosen move gets reward
  } else {
    finalOutputVector[i] = currentPrediction[i] || 0.5; // anchor (no gradient)
  }
}
```

**Key insight:** Setting target = prediction for unchosen moves creates zero-loss gradient → those outputs don't learn anything. Only the chosen move receives the learning signal.

**Configuration:**
```bash
node src/server/train.js \
  --cycles 20 \
  --games 100 \
  --eval-games 200 \
  --opponent-mix mixed \
  --gradient-isolation true \
  --log experiments/logs/exp13a.log
```

**Execution:**
- Fresh start from zero weights (meaning changes with gradient isolation)
- Same curriculum (40% heuristic, 40% network, 20% random)
- Same encoding (401-vector)

**Success criteria:**
- Win rate >58% sustained
- Training still converges (sanity check)

---

## Experiment 13b: Canonical State Encoding (Planned)

**Hypothesis:** Current encoding is non-canonical — same board state produces different input vectors depending on move order. This breaks generalization.

**Diagnosis:**
```javascript
// Current encoding (AIPlayer.js ~line 280):
for (const triangle of cell.triangles) {
  state.push(triangle.player === currentPlayer ? 1 : -1);
  state.push(triangle.rotation);  // Float 0-3
}
```

Problems:
1. Triangles appended chronologically → order varies
2. Rotation stored as float → network must decode
3. Input size (401) ≠ output size (400)

**Proposed Fix:** Fixed 4-slot structure per cell

```
[rot0_owner, rot1_owner, rot2_owner, rot3_owner]
```

Each slot: 1 = current mover, -1 = opponent, 0 = empty

Benefits:
- 1-to-1 correspondence: input[i] ↔ output[i]
- Canonical: same board → same input
- No rotation decoding needed

**Execution:**
- Fresh weights required (encoding changed)
- Gradient isolation from Exp 13a
- Same curriculum mix

**Success criteria:**
- Win rate >58% sustained
- Faster learning (simpler encoding)

---

## Future Experiment Queue

| Exp | Focus | Status | Dependencies |
|-----|-------|--------|--------------|
| 13a | Gradient isolation | Planned | Exp 12 results |
| 13b | Canonical encoding | Planned | Exp 13a success |
| 14 | CNN architecture | Backlog | If 13a/13b fail |
| 15 | Attention mechanism | Backlog | If 13a/13b fail |
