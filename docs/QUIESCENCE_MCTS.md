# Quiescence MCTS for Enclosure-Aware Search

**Updated:** 2026-10-27  
**Status:** ✅ Completed in **TF-B12** — **Heuristic Wall Broken (`100.0%` vs Random across all 10 cycles, `63.0%` vs Heuristic)**

---

## 1. The Problem: Enclosure Cascades & The Heuristic Wall

### Observed Behavior (`TF-B10` Audit)
When the AI makes a single non-enclosing move that leaves a 3-sided "hole", `HeuristicPlayer` immediately chains multiple enclosure captures consecutively:
1. AI places a piece (`claimed = 0`) that leaves an enclosure opportunity on the board and passes the turn to the opponent.
2. `HeuristicPlayer` claims 1+ enclosed triangles → earns extra turns via `Math.floor((1 + claimed) / 2)` (even `claimed = 1` grants 1 full extra turn!).
3. More enclosures become available → claim → extra turns → cascade.
4. In `TF-B10` audit games, the AI averaged only **14 turns** vs `HeuristicPlayer`'s **96 turns** in a 110-move game.

### Why Standard MCTS Fails at Inflection Points
An inflection-point audit of `TF-B10` revealed why standard MCTS (`MCTS.cjs`) fails even when trained against a 50/50 mix of Network and Heuristic opponents:
1. **Unshaped Leaf Evaluation:** Standard MCTS evaluates leaf nodes strictly using the raw neural Value Head $V_\theta(s)$ without playing out forced enclosure cascades.
2. **Value Inversion on Single-Move Openings:** Because the self-play Value Head was trained on games where early self-play clusters correlated with winning, it evaluated a move that creates a 1-triangle hole for the opponent at **`+0.4794`**, while rating a safe move at **`+0.1060`**.
3. **Search Amplification of Blunders:** With 50 simulations across ~35 legal moves, standard MCTS amplified the blunder (`13` visits, `Q = +0.344`) over safe alternatives (`1–2` visits).

---

## 2. Game-Mechanic Invariant: Enclosing Moves vs. Single-Move Openings

A critical rule of Triantica governs enclosure risk:

$$\text{extraTurns} = \left\lfloor \frac{1 + \text{claimed}}{2} \right\rfloor$$

- **Case A — Enclosing Move (`claimed > 0`):** Any move that claims at least 1 enclosed triangle awards **at least 1 extra turn** (`floor((1+1)/2) = 1`), so the active player **retains the turn** (`nextPlayer === player`). Because the mover immediately plays again, they can either continue an **enclosing train** in their own favor or patch any newly created holes before the opponent ever gets to move. **Therefore, an enclosing move (`claimed > 0`) NEVER has opponent enclosure risk (`risk = 0`).**
- **Case B — Non-Enclosing Single Move (`claimed === 0`):** If `extraTurns === 0`, the turn passes immediately to the opponent (`nextPlayer !== player`). If this single move leaves 1 or more enclosure opportunities on the board for the opponent, the opponent can deterministically exploit them and trigger an extra-turn cascade. **Only non-enclosing moves that pass the turn can carry opponent enclosure risk (`risk > 0`).**

---

## 3. Solution: Quiescence MCTS (`QuiescenceMCTS.cjs`)

`QuiescenceMCTS` extends `MCTS` across both `search()` and `searchBatched()` (`_runBatch`) with three coordinated mechanisms—**requiring only the clean 2-head network (Policy + Value)**:

### 3.1 Root Prior & Policy Target Shaping
When expanding the root node (`_expandRootWithRisk`) and computing the MCTS training policy target (`_getRiskAdjustedPolicy`):
- Every root child records `child.claimed` (triangles enclosed by that move) and `child.enclosureRisk = calculateChildNodeRisk(child, rootPlayer)`.
- **Enclosing moves (`child.claimed > 0`):** Have `risk = 0` and receive an explicit **enclosure bonus** (`enclosureBonusWeight`), encouraging the AI to discover and chain multi-step enclosing trains.
- **Dangerous single-move openings (`child.claimed === 0` and `child.enclosureRisk.risk > 0`):** Receive a multiplicative **risk penalty** (`riskPenaltyWeight`), suppressing both UCT exploration priors and the final MCTS visit policy target stored in the Replay Buffer.

### 3.2 Pre-Cascade Leaf Expansion + Post-Cascade Quiescence Evaluation
At every non-terminal leaf node selected during `_runBatch`:
1. **True-State Leaf Expansion:** The leaf node `leaf` is always expanded using the neural policy predicted on its true pre-cascade board `leaf.grid` (`leaf.expand(policy)`), ensuring legal moves and priors match `leaf.grid`.
2. **Deterministic Quiescence Rollout (`_runQuiescenceRollout`):** Starting from `leaf.grid`, any available enclosures are played out greedily (up to `quiescenceDepth`) using an $O(1)$ local neighborhood enclosure detector (`countWouldBeEnclosed`, which checks only the moved cell and its 4 orthogonal neighbors without cloning the grid).
   - Enclosures claimed by `rootPlayer` during the leaf move or subsequent cascade add `+claimed × enclosureBonusWeight` to `rootAdjustment`.
   - Enclosures claimed by the opponent during the cascade subtract `-claimed × riskPenaltyWeight` from `rootAdjustment`.
3. **Batched GPU Evaluation:** Both the pre-cascade expansion states and the post-cascade quiescent states (`qEval.grid`) are evaluated together in a **single combined `network.batchPredict(allStates)` call**, keeping GPU utilization high (~140ms per 50-sim move).

---

## 4. Experiment Design: TF-B12

Rather than adding auxiliary network heads or artificial TD($\lambda$) reward shaping—which corrupt gradients—**TF-B12** tests the clean 2-head ConvNet (`Policy + Value`) warm-started from `TF-B10_network.json` with `QuiescenceMCTS` and a 50/50 mix of Network and Heuristic opponents:

```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001 \
  --opponent-mix mixed --network-ratio 0.5 --heuristic-ratio 0.5 --random-ratio 0.0 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.2 \
  --start-from dashboard/data/TF-B10_network.json \
  --output dashboard/data/TF-B12.json > dashboard/data/TF-B12.log 2>&1 < /dev/null &
```

### Verification (`test_quiescence.cjs`)
```bash
node test_quiescence.cjs
```
Verifies that:
- Enclosing moves (`claimed > 0`) always have `risk === 0`.
- `QuiescenceMCTS.searchBatched()` executes `_runBatch` with quiescence rollouts and selects enclosing moves even under a flat prior network.

---

## 5. Experiment `TF-B12` Final Results (2026-10-27)

- **vs Random (`100` eval games/cycle):** **`100.0%` across all 10 cycles (`1,000W / 0L / 0D`)**, with average AI cluster size climbing from `116.7` (Cycle 1) to **`166.2` vs `4.4`** (Cycle 10).
- **Training MCTS-50 vs Heuristic:** Climbed from `8.0%` in `TF-B10` to **`32.1%` (Cycle 1) → `49.0%` (Cycle 9) → `42.3%` (Cycle 10)** (`36.4%` average across all 10 cycles).
- **Eval vs Heuristic (1-Ply Policy Network, `exploration = 0`):**
  - **Cycles 1–7 (Deterministic 2-trajectory eval):** Broke the Heuristic Wall in **Cycle 6 at `50.0%`** (winning `100%` as P2, `147–24`).
  - **Cycles 8–10 (Random First 2 Plies — 100 distinct openings):** Climbed from **`50.0%` (Cycle 8: `50% P1 / 50% P2`) → `62.0%` (Cycle 9: `68% P1 / 56% P2`) → `63.0%` (Cycle 10: `66% P1 / 60% P2`, mean cluster `101.7` vs `79.5`)**!
  - By comparison, `TF-B10_network.json` evaluated under the exact same 100-game random-first-2-plies protocol scored **`0.0%` (`0W / 100L`, mean cluster `7.8` vs `175.9`)**.
