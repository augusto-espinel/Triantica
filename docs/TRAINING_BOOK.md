

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

---

## 🐛 Critical Bug Fix: Eval Win Rate Measurement (2026-10-21)

**Bug:** `GameSession.runGames()` swaps players every game for fairness, but `train.js` computed AI win rate as `wins[1] / evalGames`. After the swap, `wins[1]` contains Random's wins. This caused eval to always report ~50% regardless of actual AI skill.

**Fix:** Track wins by identity (AI vs opponent) rather than by slot number. `results.wins` now has `{ ai, random, draws, bySlot }`.

**Impact:** Every experiment from 1-12 reported ~50% eval win rate. The real rate was ~75% (confirmed by diagnostic). The network was learning — we just couldn't see it.

**Per-opponent tracking also added:** Self-play summary now breaks down wins by opponent type (Network/Heuristic/Random), revealing that:
- Network vs Network self-play is actually ~50/50 (fair)
- Heuristic beats AI 100% (0 wins in 111 games)
- AI beats Random ~73% in training

The "P2 dominance" in Exp 12 was entirely Heuristic-as-P2 wins inflating the P2 count.

## Experiment 12 Replicated (Fixed Eval)

**Started:** 2026-10-21 05:11 (with eval bug fix)
**Completed:** 2026-10-21 07:15
**Log:** `experiments/logs/exp12r_v2.log`
**Network:** `experiments/data/exp12r_v2_network.json`

**Config:** Same as original Exp 12:
- 20 cycles × 100 games, 100 iterations/cycle, lr=0.01
- Mixed opponents: 40% Heuristic, 20% Random, 40% Network
- Starting from exp12b_network.json
- Gradient isolation: OFF
- Exploration: 0.2

**Key difference from original:** Eval now correctly tracks AI wins regardless of player slot swap.

### Bug Fix Impact

| Metric | Original (Broken) | Fixed Eval |
|--------|-------------------|------------|
| Cycle 1 eval win rate | 49.0% | **78.5%** |
| Per-opponent wins (Cycle 1) | Not tracked | Net: 18/21 Heur: 0/40 Rand: 17/4 |

The network was performing at ~78% vs Random all along. The 49% was a measurement artifact from the player-swap bug in `GameSession.runGames()`.

### Per-Opponent Breakdown

Training self-play results decomposed by opponent type:
- **Network vs Network:** ~46-55% P1 win rate (fair game, slight P2 edge in some cycles)
- **AI vs Heuristic:** 0% win rate consistently (heuristic too strong)
- **AI vs Random:** ~70-81% win rate (real performance indicator)

### Full Results

| Cycle | Eval Win Rate | AI Med | Rand Med | Gap | vs Net | vs Heur | vs Rand |
|-------|---------------|--------|----------|-----|--------|---------|--------|
| 1 | 74.5% | 40 | 40 | ±0 | 19/17/1 | 0/37/0 | 13/3/0 |
| 2 | 78.0% | 38 | 39 | -1 | 25/19/1 | 0/39/0 | 10/4/2 |
| 3 | 79.0% | 41 | 43 | -2 | 20/19/0 | 0/38/0 | 17/6/0 |
| 4 | **81.5%** | 41 | 40 | +1 | 19/23/0 | 0/35/0 | 18/5/0 |
| 5 | 80.5% | 41 | 44 | -3 | 18/16/0 | 0/50/0 | 13/3/0 |
| 6 | 79.0% | 41 | 41 | ±0 | 18/19/1 | 0/42/0 | 12/8/0 |
| 7 | 78.0% | 42 | 39 | +3 | 17/19/2 | 0/47/0 | 9/5/1 |
| 8 | **81.5%** | 39 | 40 | -1 | 22/24/0 | 0/38/0 | 10/6/0 |
| 9 | 79.0% | 42 | 43 | -1 | 23/15/1 | 0/38/0 | 18/5/0 |
| 10 | 80.5% | 39 | 42 | -3 | 17/14/1 | 0/44/0 | 19/5/0 |
| 11 | 78.0% | 42 | 40 | +3 | 19/12/1 | 0/47/0 | 13/8/0 |
| 12 | 74.0% | 40 | 38 | +2 | 17/25/0 | 0/35/0 | 16/6/1 |
| 13 | 71.5% | 39 | 40 | -1 | 11/19/2 | 0/42/0 | 15/9/2 |
| 14 | 74.5% | 39 | 41 | -2 | 19/15/1 | 0/38/0 | 15/11/1 |
| 15 | 77.0% | 39 | 38 | +1 | 17/18/0 | 0/46/0 | 12/7/0 |
| 16 | 70.5% | 42 | 37 | +5 | 13/22/0 | 0/44/0 | 16/5/0 |
| 17 | 75.5% | 40 | 41 | -1 | 15/23/1 | 0/39/0 | 13/9/0 |
| 18 | 72.5% | 39 | 36 | +4 | 16/25/1 | 0/41/0 | 14/3/0 |
| 19 | 72.0% | 42 | 44 | -2 | 24/24/1 | 0/35/0 | 9/7/0 |
| 20 | 70.5% | 34 | 39 | -5 | 23/20/2 | 0/42/0 | 11/2/0 |

**Starting win rate:** 74.5%
**Final win rate:** 70.5%
**Peak win rate:** 81.5% (cycles 4, 8)
**Change:** -4.0%

### Per-Opponent Breakdown Trends

**vs Heuristic:**
- 0% win rate across ALL 20 cycles (heuristic dominates completely)
- Heuristic always plays as P2 in self-play, wins 100%
- Zero learning signal from heuristic opponent (pure loss gradient)

**vs Random:**
- Self-play win rate: 56-85% range
- Eval win rate peaked at 81.5%, declined to 70.5%
- Relatively stable performance against random

**vs Network (self-play):**
- Roughly 50/50 split (fair game)
- Slight P2 advantage in later cycles (16-19: P2 won more)
- Cycle 20: 23/20/2 (P1 wins 51%)

### Key Findings

1. **Did network improve past 85%?** NO — peak was 81.5%, never reached 85%. Final was 70.5%, showing regression.

2. **Eval trajectory:** Early peak (cycles 4-8), then gradual decline. Oscillation pattern similar to Exp 12 and 12b.

3. **Heuristic opponent fails:** 0% win rate means pure loss signal. Network learns "avoid this outcome" but never learns HOW to win against it.

4. **Training converges, win rate doesn't:** Error → 0.0009 every cycle, but win rate regresses. Confirms gradient dilution diagnosis.

### Conclusion

The curriculum learning approach (40% heuristic, 40% network, 20% random) fails to produce sustained improvement. The heuristic opponent provides no positive learning signal (100% loss rate). The network beats random at ~75% but cannot break through to 85%+.

**Recommendation:** Proceed to Exp 13a (gradient isolation) or Exp 13b (canonical encoding). Consider removing or weakening heuristic opponent — 0% win rate = zero positive gradient.

---


### Full 20-Cycle Results

| Cycle | Eval vs Random | Trend |
|-------|---------------|-------|
| 1 | 74.5% | |
| 2 | 78.0% | ↑ |
| 3 | 79.0% | ↑ |
| 4 | **81.5%** | ↑ Peak |
| 5 | 80.5% | ↓ |
| 6 | 79.0% | ↓ |
| 7 | 78.0% | ↓ |
| 8 | 81.5% | Spike |
| 9 | 79.0% | ↓ |
| 10 | 80.5% | |
| 11 | 78.0% | ↓ |
| 12 | 74.0% | ↓ |
| 13 | 71.5% | ↓ |
| 14 | 74.5% | |
| 15 | 77.0% | |
| 16 | 70.5% | ↓ |
| 17 | 75.5% | |
| 18 | 72.5% | ↓ |
| 19 | 72.0% | ↓ |
| 20 | 70.5% | ↓ |

**Starting win rate:** 74.5%
**Final win rate:** 70.5%
**Change:** -4.0%

### Analysis

Two distinct phases:
1. **Cycles 1-4:** Rapid improvement 74.5% → 81.5%. Fresh training signal from mixed opponents.
2. **Cycles 5-20:** Gradual decline 81.5% → 70.5%. Heuristic loss signal dominates.

**Root cause:** The 0% win rate vs Heuristic means ~36% of training data is pure loss signal. Over time, this teaches the network "I lose" more strongly than it learns "I beat random." The heuristic games act as catastrophic interference.

**Key insight:** The Exp 12b network (starting point) was already at ~78% vs Random — we just couldn't see it because of the eval bug. The brief training boost to 81.5% is real but unsustainable with the current opponent mix.

### Conclusion

Mixed curriculum with a dominant heuristic opponent is counterproductive. The network needs either:
- **Weakened heuristic** (higher exploration, ~20-30% random moves) so AI can occasionally win and learn from victories
- **Pure network self-play** or **vs-random-only** to avoid the loss-signal ceiling
- **Loss weighting** to balance heuristic loss signal against random win signal

The 81.5% peak proves the architecture can learn. The decline proves the training mix matters more than the architecture.

---

## Experiment 12s — Learn from Heuristic's Perspective

**Started:** 2026-10-21 08:05
**Log:** `experiments/logs/exp12s.log`
**Network:** `experiments/data/exp12s_network.json`
**Starting from:** `experiments/data/exp12r_network.json` (the 85% early-stop checkpoint)

### Hypothesis

The previous experiments collected training data only from the AI's perspective. When the heuristic opponent won as P2, the network learned "I lose" but never learned *how* the heuristic won.

**New feature:** `--train-on-opponent-moves` flag enables collecting training data from the opponent's perspective during vs-opponent games. This means:
- When heuristic wins as P2, we train on its winning moves (positive reward)
- The network learns what strong play looks like from P2's perspective
- Converts pure loss signal into learning signal

### Config

Same as Exp 12r but with new flag:
```bash
node src/server/train.js \
  --cycles 20 \
  --games 100 \
  --eval-games 200 \
  --iterations 100 \
  --learning-rate 0.01 \
  --exploration 0.2 \
  --self-play-mode network \
  --opponent-mix mixed \
  --heuristic-ratio 0.4 \
  --random-ratio 0.2 \
  --train-on-opponent-moves true \
  --start-from experiments/data/exp12r_network.json \
  --output experiments/data/exp12s_network.json \
  --log experiments/logs/exp12s.log
```

### Expected Behavior

- Heuristic games should now contribute positive training signal when heuristic wins
- The 81.5% → 70.5% degradation from Exp 12r should not repeat
- Win rate should stabilize or improve past cycle 4


---

## UI Update — AI vs Opponent Selection

**Added:** 2026-10-21 10:15

### New Menu Options

The Phaser game now supports selecting AI opponents:

- **AI vs Random** (`AvR`): Watch AI play against a random baseline
- **AI vs Heuristic** (`AvH`): Watch AI play against the greedy heuristic
- **AI vs AI** (`AvA`): Watch AI play against itself (network self-play)

### Implementation

Modified `game.js`:
- Added new menu buttons in `MenuScene`
- New `triggerNonHumanMove()` method handles Random and Heuristic players
- Updated `handleTurnSwitch()` to route to correct move method based on player type
- Player types: `human`, `ai`, `random`, `heuristic`

Modified `index.html`:
- Added ES6 module imports for RandomPlayer and HeuristicPlayer
- Global window exports for compatibility with game.js

Modified player files:
- Added global exports to `AIPlayer.js` and `HeuristicPlayer.js`

### Purpose

This allows observing the trained AI's play style directly in the browser, comparing:
- How AI performs against random (should win ~70-75%)
- How AI performs against heuristic (currently 0%)
- AI self-play dynamics

### Next Steps

- Run `npm run dev` to test the new menu
- Observe AI games to understand strategic patterns
- Compare pre-training and post-training AI behavior


### Final Results

| Metric | Value |
|--------|-------|
| Starting win rate | 70.0% |
| Final win rate | 63.0% |
| Change | **-7.0%** |
| Total time | 8640.8s (~144 min) |
| vs Heuristic | 0 wins / ~760 games |

**Full trajectory:**
```
70.0 → 61.5 → 71.5 → 70.0 → 74.0 → 62.5 → 70.5 → 64.0 → 68.0 → 66.5
   → 67.5 → 72.0 → 73.0 → 68.0 → 69.5 → 73.5 → 60.0 → 67.5 → 65.0 → 63.0
```

**Mean: 67.9%, Range: 60.0-74.0%**

### Conclusion: Hypothesis Failed

Learning from the opponent's perspective did NOT help. The hypothesis was:

> "If the network trains on the heuristic's winning moves, it will learn how strong players think."

**Why it failed:**
1. The heuristic's strategy is not representable in the current network architecture
2. Adding opponent-perspective data creates conflicting gradients (good for P2, bad for P1)
3. The heuristic plays a fundamentally different game than the network can encode

**vs Heuristic:** Still 0% — no improvement at all across 760+ heuristic games.

### Path Forward

Mixed curriculum with a dominant heuristic is a dead end. Options:

1. **Pure network self-play** — remove heuristic entirely, let network evolve its own strategies
2. **Weaken heuristic** — inject 25-35% random moves so AI can occasionally win
3. **Architecture change (Exp 13b)** — canonical encoding + value head
4. **Accept ceiling** — 81.5% vs Random is the limit for this architecture


---

## Game Mode Reference — Exploration & Opponent Mix

This section documents how randomness, exploration, and opponent selection work across all game modes.

### Key Concepts

**Exploration** — Probability of making a random move instead of using the trained network. Used to discover new strategies and avoid overfitting.

**Opponent Mix** — What type of opponent the AI plays against during training.

**Self-Play Mode** — When both players use the network, how the opponent makes decisions.

---

### 1. Training — Self-Play Games (Network vs Network)

**When:** `--opponent-mix network` (default) or when mixed mode selects network opponent.

**How it works:** Both players use the same network. The network is conditioned on `currentPlayer` in the input, so it predicts good moves for whoever is moving.

**Exploration:** Hardcoded in `runSelfPlayGame()` — ignores `--exploration` CLI flag.

| Game Phase | Random | Greedy Immediate | Network |
|------------|--------|-----------------|---------|
| Opening (first 5 moves) | 50% | 10% | 40% |
| Mid-game | 10% | 10% | 80% |
| Endgame | 10% | 10% | 80% |

- **Random:** Pure random valid move
- **Greedy Immediate:** Move with highest immediate reward (enclosure + cluster growth)
- **Network:** Network's top-scoring valid move

**Why hardcoded:** Self-play needs higher exploration to discover diverse strategies. Opening diversity (50% random) ensures the network sees many different game trajectories.

---

### 2. Training — vs-Opponent Games (AI vs Random/Heuristic)

**When:** `--opponent-mix random`, `--opponent-mix heuristic`, or mixed mode.

**Opponent behavior:**

| Opponent | Exploration | Strategy |
|----------|-------------|----------|
| Random | 0% | Pure random valid moves |
| Heuristic | 5% | Greedy enclosure + cluster growth (injected randomness for training variety) |

Note: During training, HeuristicPlayer is created with `explorationFactor: 0.05`.

**AI behavior:** Uses `--exploration` CLI flag (default 0.2).

| Mode | AI Exploration |
|------|----------------|
| vs Random | 20% random moves |
| vs Heuristic | 20% random moves |

---

### 3. Training — Mixed Opponent Mode

**When:** `--opponent-mix mixed`

**Per-game opponent selection:**

```
Heuristic: 40% of games (from --heuristic-ratio 0.4)
Random:    20% of games (from --random-ratio 0.2)
Network:   40% of games (remaining)
```

**Why this matters:** The ratio determines training signal composition.

- **vs Heuristic (40%):** 0% win rate → pure loss signal → teaches "you lose"
- **vs Random (20%):** 70-75% win rate → mixed signal → teaches some winning
- **vs Network (40%):** 50% win rate → neutral signal → teaches self-consistency

**Problem:** Heuristic games (40% of total) contribute 0 positive gradient. This dilutes learning.

---

### 4. Evaluation (Measuring Strength)

**When:** After each training cycle.

**Configuration:**
- Opponent: Always RandomPlayer (pure random, 0% exploration)
- AI exploration: Fixed at 5%
- Games: 200 (from `--eval-games`)

**Why 5%:** Evaluation should mostly exploit learned strategy, with minimal noise. 5% random provides slight variation without distorting the measurement.

**Why vs Random only:** We need a stable baseline to compare across cycles. Random is deterministic and provides consistent measurement. Heuristic would show 0% across all cycles (no signal).

---

### 5. Actual Gameplay (Phaser UI)

**Modes:**

| Mode | Player 1 | Player 2 | P1 Exploration | P2 Exploration |
|------|----------|----------|----------------|-----------------|
| HvH | Human | Human | N/A | N/A |
| HvAI | Human | AI | N/A | 20% (default) |
| AvR | AI | Random | 20% | 0% |
| AvH | AI | Heuristic | 20% | 0% |
| AvA | AI | AI | 20% | 20% |

**Important:** AvA in the UI does NOT use the self-play logic from training. It uses regular `getMove()` with 20% exploration for both players. This is for observation only — the training self-play uses higher, phase-dependent exploration.

---

### 6. Summary Table

| Context | P1 Strategy | P2 Strategy | Exploration Source |
|---------|-------------|-------------|-------------------|
| Training: Network self-play | Network + phase-dependent | Network + phase-dependent | Hardcoded (10-50%) |
| Training: vs Random | Network + 20% random | Pure random | CLI flag |
| Training: vs Heuristic | Network + 20% random | Heuristic + 5% random | CLI flag |
| Training: Mixed | Varies per game | Varies per game | Per-opponent-type |
| Evaluation | Network + 5% random | Pure random | Fixed |
| UI: AvR | Network + 20% random | Pure random | Default config |
| UI: AvH | Network + 20% random | Pure greedy heuristic | Default config |
| UI: AvA | Network + 20% random | Network + 20% random | Default config |

---

### Problems Identified

1. **Inconsistent exploration between training and UI:** Training self-play uses 10-50% exploration (phase-dependent), but UI AvA uses flat 20%.

2. **Mixed mode signal imbalance:** 40% heuristic games contribute 0 positive gradient, diluting learning.

3. **Evaluation doesn't measure vs Heuristic:** We only test vs Random, so we never see the 0% win rate during training.

4. **Opponent exploration during training:** Heuristic has 5% random moves, which may confuse the network about what "strong play" looks like.

---

### Recommendations

1. **Add vs-Heuristic evaluation:** Include a separate eval vs Heuristic to track progress against strong opponents.

2. **Align UI AvA with training self-play:** Either make AvA use phase-dependent exploration, or make training use flat exploration for consistency.

3. **Reduce or eliminate heuristic opponent:** If 0% win rate, either:
   - Remove heuristic from training entirely (pure self-play)
   - Weaken heuristic to 70-80% win rate (inject 25-35% random moves)
   - Accept that this opponent is beyond current architecture

4. **Increase exploration vs Random:** If the AI wins 70% vs Random, it may need more exploration to push higher. Try 30-40%.


---

## Experiment 13 — Pure Network Self-Play

**Started:** 2026-10-21 12:30
**Log:** `experiments/logs/exp13.log`
**Network:** `experiments/data/exp13_network.json`
**Starting from:** `experiments/data/exp12r_network.json` (the 85% early-stop checkpoint)

### Hypothesis

Removing all external opponents (random and heuristic) will eliminate conflicting gradients and allow the network to evolve its own strategies through pure self-play.

### Configuration

```bash
node src/server/train.js \
  --cycles 20 \
  --games 100 \
  --eval-games 200 \
  --iterations 100 \
  --learning-rate 0.01 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix network \
  --start-from experiments/data/exp12r_network.json \
  --output experiments/data/exp13_network.json \
  --log experiments/logs/exp13.log
```

**Key changes from Exp 12:**
- `--opponent-mix network` (100% network self-play, no random/heuristic)
- `--exploration 0.3` (higher exploration for discovery)
- Starting from the strongest checkpoint (85% vs Random)

### Expected Behavior

1. Win rate should start at ~85% (baseline from checkpoint)
2. Network plays itself, exploring new strategies through 30% random moves
3. May initially drop as network explores, then stabilize or improve
4. Should NOT show the degradation pattern from mixed curriculum

### Success Criteria

- Win rate stabilizes above 80% without degradation
- Opening diversity remains high (network explores different strategies)
- No 0% win rate anomalies (no dominant external opponent)

### Reference: Postponed Architecture Change

**Exp 14 (formerly "13b")** — Canonical State Encoding + Value Head Architecture

This experiment involves fundamental architecture changes:
- 4-slot per-cell encoding (canonical representation)
- Value head instead of 400-output policy head
- Fresh weights (not starting from current network)

Postponed until we understand the ceiling of the current architecture with pure self-play.

