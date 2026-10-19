# Triantica AI Training Book

> A personal journal of ML/AI experiments through a strategy game

**Started:** 2026-10-18  
**Project:** Grid-based triangle placement strategy game with neural network AI  
**Repository:** `https://github.com/augusto-espinel/Triantica`

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Game Rules](#game-rules)
3. [AI Architecture](#ai-architecture)
4. [Training Methodology](#training-methodology)
5. [Experiment Log](#experiment-log)
6. [Performance Metrics](#performance-metrics)
7. [Lessons Learned](#lessons-learned)
8. [Future Experiments](#future-experiments)

---

## Project Overview

### Goals

- Learn ML/AI fundamentals through hands-on experimentation
- Understand how neural network architecture affects gameplay
- Explore reinforcement learning via self-play
- Document everything for future reference

### Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Game Engine | Phaser 3 | 3.60.0 |
| Neural Network | Brain.js | 2.0.0-beta.23 |
| Platform | Browser (vanilla JS) | - |

### Why Brain.js?

- Pure JavaScript, runs in browser without build tools
- Simple API for feedforward networks
- Easy to inspect and debug
- Good for learning fundamentals before moving to TensorFlow.js

---

## Game Rules

### Board

- 10×10 grid of cells
- Each cell can hold **up to 2 triangles** (one per diagonal orientation)
- Total capacity: 200 triangles

### Triangle Types

```
Rotation 0 (0°):   ╱  bottom-left triangle
Rotation 1 (90°):  ╲  top-left triangle  
Rotation 3 (180°): ╱  top-right triangle
Rotation 2 (270°): ╲  bottom-right triangle
```

Note: Rotations 0 and 3 share the ╱ diagonal, rotations 1 and 2 share the ╲ diagonal.

### Placement Rules

1. **First move**: Anywhere on the board
2. **Subsequent moves**: Must share at least one corner with existing triangles
3. **No overlap**: Cannot place same diagonal orientation in an occupied cell
4. **Wrap-around**: Board edges wrap (toroidal topology)

### Enclosure Mechanic

When a triangle placement completely surrounds an empty slot:
- The empty slot is automatically claimed
- Player earns an **extra turn**

Example: If a cell has one ╱ triangle, placing three ╲ triangles around it encloses the ╲ slot.

### Win Condition

When the grid is full, the player with the **largest contiguous cluster** wins.

---

## AI Architecture

### Network Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                            │
│                      401 neurons                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Cell 0,0: [P1, R1, P2, R2] × 100 cells = 400 values │   │
│  │ Current player flag: 1 value                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HIDDEN LAYER(S)                          │
│                  (varies by difficulty)                      │
│                                                             │
│  Easy:   [100 neurons]                                      │
│  Medium: [200 neurons]                                      │
│  Hard:   [267, 160 neurons] (2 layers)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                           │
│                      400 neurons                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Each output = confidence score for a specific move  │   │
│  │ Index = (row × cols + col) × 4 + rotation           │   │
│  │ Higher value = more confident this move is good     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Input Encoding (State Representation)

Each cell contributes 4 values to the input vector:

| Position | Meaning | Encoding |
|----------|---------|----------|
| 0 | Player who owns first triangle | `0` = empty, `1` = AI, `-1` = opponent |
| 1 | Rotation of first triangle | `-1` = empty, `0.0-0.75` = normalized rotation |
| 2 | Player who owns second triangle | `0` = empty, `1` = AI, `-1` = opponent |
| 3 | Rotation of second triangle | `-1` = empty, `0.0-0.75` = normalized rotation |

Final value: Current player flag (`1` if AI's turn, `-1` otherwise)

### Output Encoding (Move Selection)

Each output index maps to a specific move:
```
output[i] → move at position (x, y) with rotation r
where:
  rotation = i % 4
  cell_index = floor(i / 4)
  x = cell_index % cols
  y = floor(cell_index / cols)
```

Values are sigmoid-activated (range 0.0 to 1.0):
- `0.01` = minimum confidence (used for invalid moves)
- `0.99` = maximum confidence

---

## Training Methodology

### Self-Play Reinforcement Learning

The AI learns by playing against itself:

```
┌────────────────────────────────────────────────────────────┐
│                    SELF-PLAY LOOP                          │
│                                                            │
│  1. Initialize game state                                  │
│  2. While game not over:                                   │
│     a. Generate state representation                       │
│     b. Calculate immediate rewards for all legal moves    │
│     c. Select move (network prediction or exploration)     │
│     d. Execute move, handle claims                        │
│     e. Record (state, move, player) to history             │
│     f. Switch players (unless extra turn earned)           │
│  3. Determine winner                                       │
│  4. Process trajectory backwards, assigning rewards       │
│  5. Add training samples to buffer                         │
│  6. Train network on accumulated data                      │
└────────────────────────────────────────────────────────────┘
```

### Reward Structure

#### Immediate Rewards (calculated for each legal move)

| Reward Type | Formula | Max Value |
|-------------|---------|-----------|
| Enclosure Claims | `claims × 0.3` | ~1.5 (5 claims) |
| Connected Cluster Size | `cluster_size × 0.1` | ~20 (full board) |

#### Final Outcome Rewards

| Outcome | Base Reward |
|---------|-------------|
| Win | `+1.0` |
| Loss | `-1.0` |
| Draw | `0.0` |

#### Discounted Final Reward

The final outcome is **linearly discounted** based on move position:
```
discounted_reward = final_outcome × (move_number / total_moves) × weight

Example (50-move game, AI wins):
- Move 50 (last):  1.0 × 1.00 × 1.0 = +1.00
- Move 25 (mid):   1.0 × 0.50 × 1.0 = +0.50
- Move 1  (first): 1.0 × 0.02 × 1.0 = +0.02
```

Rationale: Moves closer to the end have more direct impact on the outcome.

#### Combined Reward Calculation

```
combined_reward = immediate_reward + discounted_final_reward

Target output is then scaled to [0.01, 0.99]:
scaled_output = (combined_reward - min_possible) / (max_possible - min_possible)
```

### Move Selection Strategy (Epsilon-Greedy)

During self-play, the AI balances exploitation and exploration:

| Mode | Probability | Behavior |
|------|-------------|----------|
| **Exploit (Network)** | 72% (90% × 80%) | Use trained network prediction |
| **Exploit (Immediate)** | 18% (90% × 20%) | Pick highest immediate reward |
| **Explore** | 10% | Random valid move |

### Training Parameters

| Parameter | Current Value | Notes |
|-----------|---------------|-------|
| `explorationFactor` | 0.10 | Probability of random move |
| `immediacyFactor` | 0.20 | Prob. of choosing by immediate reward |
| `iterations` | 500 | Training iterations per session |
| `learningRate` | 0.10 | Step size for gradient descent |
| `errorThresh` | 0.00093 | Early stopping threshold |
| `activation` | sigmoid | Output range 0-1 |

---

## Experiment Log

> ⚠️ **READING ORDER MATTERS:** This section was written chronologically as we discovered things. Many early assertions turned out to be WRONG. Key timeline:
> - **Experiments 2-5:** Written before discovering zero-exploration bug. Conclusions are outdated.
> - **"Critical Bug" section:** Discovered 2026-10-18 ~22:22 — this is the ACTUAL root cause.
> - **"Re-evaluation" section:** Written after bug discovery — this updates earlier conclusions.
> 
> For accurate analysis, skip to the "Critical Bug" and "Re-evaluation" sections. Read earlier sections as "what we thought at the time" — they show how debugging works in practice.

### Baseline (Pre-2026-10-18)

**Network:** Pre-trained (~5MB JSON file)  
**Architecture:** Medium difficulty (200 hidden neurons)  
**Training History:** Unknown (file committed to repo)

**Observations:**
- Network file present but no training logs
- Unknown number of training games or iterations
- No performance baseline established

### Experiment 2: Architecture Refactoring & Baseline

**Date:** 2026-10-18  
**Goal:** Extract game logic, build testing framework, establish quantitative baseline

**Actions:**
- [x] Extract GridManager from game.js → src/core/GridManager.js  
- [x] Build PlayerInterface + RandomPlayer + GameSession + GameLogger  
- [x] Add Node.js server + CLI runner  
- [x] Write 17 automated validation tests  
- [x] Refactor AIPlayer to extend PlayerInterface  
- [x] Fix brain.js ESM resolution (dynamic import + mod.default)  
- [x] Run baseline experiments

**Bugs Fixed:**
1. Cluster adjacency: same-cell triangles share corners, not diagonal groups
2. Coordinate convention: placeTriangle(x,y) → grid[y][x]
3. brain.js ESM: require() fails, need await import('brain.js') + mod.default fallback
4. Missing await on async _getBrain() in initialize()

**Results:**

| Experiment | Games | P1 Wins | P2 Wins | Draws | AI Win Rate | Avg Moves |
|-----------|-------|---------|---------|-------|-------------|------------|
| Random vs Random | 100 | 44 | 50 | 6 | — | 167.6 |
| AI vs Random (AI=P1) | 50 | 28 | 22 | 0 | 56% | 164.2 |
| Random vs AI (AI=P2) | 50 | 24 | 23 | 3 | 46% | 166.0 |
| AI vs AI | 10 | 6 | 4 | 0 | — | 165.0 |

**Overall AI Win Rate vs Random: ~51%** — statistically indistinguishable from coin-flip.

**Finding:** The pre-trained network (~5MB) has not learned meaningful strategy beyond random play. The network needs retraining with more self-play games and potentially adjusted hyperparameters.

**New Infrastructure:**
- `npm test` — 17 validation tests
- `npm run test:ai` — AI integration tests (AI vs Random + AI vs AI)
- `npm run experiment -- --type vs-random --count 100`
- All data saved as JSON in `experiments/data/`

---

### Experiment 3: Self-Play Training (20 Cycles)

**Date:** 2026-10-18  
**Goal:** Train AI via self-play, measure win-rate improvement vs random baseline

**Architecture:** 401 input → 200 hidden → 400 output (medium difficulty)  
**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `explorationFactor` | 0.30 |
| `immediacyFactor` | 0.90 |
| `learningRate` | 0.10 |
| `iterations` | 300/cycle |
| `errorThresh` | 0.00093 |
| `activation` | sigmoid |
| Self-play games/cycle | 50 |
| Eval games/cycle | 50 |
| Claim reward | +0.3/claim |
| Cluster reward | +0.1/triangle |
| Win/loss reward | +1.0/-1.0 |

**Results (4 complete cycles before timeout):**

| Cycle | Training Error | AI Win Rate vs Random | AI Wins | Random Wins | Draws | Avg Moves | Duration |
|-------|---------------|----------------------|---------|-------------|-------|-----------|----------|
| Baseline | — | 51% | 51 | 49 | 0 | 165.9 | — |
| 1 | 0.000930 | 50.0% | 25 | 25 | 0 | 167.9 | 701s |
| 2 | 0.008396 | 56.0% | 28 | 19 | 3 | 166.7 | 745s |
| 3 | 0.014364 | 50.0% | 25 | 24 | 1 | 166.6 | 737s |
| 4 | 0.016922 | 34.0% | 17 | 31 | 2 | 168.2 | 729s |

**Training Error Progression:**
```
Cycle 1: 0.0009 → Cycle 2: 0.0084 → Cycle 3: 0.0144 → Cycle 4: 0.0169
```
Error is *increasing* across cycles, not decreasing.

#### Analysis — ⚠️ OUTDATED (Critical Bug Unknown at Time)

> **NOTE:** This analysis was written before discovering the zero-exploration bug. While the observations are correct, the root cause analysis is wrong. The "unlearning" pattern and distribution drift are SECONDARY effects — the PRIMARY cause was lack of exploration.

**1. The "Unlearning" Pattern**

The pre-trained network had extremely low training error (0.0009) in cycle 1 because it had memorized its original (poor) self-play data. The weights encoded a near-random policy that the sigmoid activations could fit tightly. As new self-play data introduces genuine strategic signal, the optimizer must **destroy old memorized patterns** before it can build new ones. This is the "creative destruction" phase — training error rises while actual policy quality oscillates.

**2. Why Win Rate Declined After Cycle 2**

Cycle 2's 56% win rate was likely a statistical fluctuation (only 50 games, ±14% at 95% CI). The decline to 34% in cycle 4 is more concerning and suggests the network is actively *getting worse*. Probable causes:

- **Catastrophic forgetting**: Brain.js trains on the entire accumulated dataset each cycle, but new samples may overwhelm the signal from earlier cycles. Without experience replay prioritization, the network chases the latest self-play distribution.
- **Reward signal noise**: With only 50 games/cycle, the final-outcome reward is extremely noisy. A coin-flip game outcome provides almost no signal about which individual moves were good or bad. The immediate reward (cluster size, claims) is more reliable but only teaches short-term tactics.
- **Self-play distribution collapse**: After training on its own games, the AI plays differently → generates different training data → overfits to that specific style → loses generalization vs random.

**3. Architecture Limitations**

The 401→200→400 architecture has fundamental issues for this task:

- **No spatial awareness**: The flat input vector loses the 2D grid structure. A triangle at (0,0) and (9,9) are treated the same way by the fully-connected hidden layer. The network must *learn* adjacency from scratch, which is extremely sample-inefficient.
- **4× over-parameterized output**: 400 outputs (10×10×4) means most outputs are always invalid (cell full). The network wastes capacity predicting scores for impossible moves.
- **Single hidden layer**: Complex spatial strategies (enclosures, multi-turn sequences) likely require hierarchical representations that a single hidden layer cannot capture.

#### Hypotheses — ⚠️ MOSTLY INVALIDATED

> **UPDATE:** These hypotheses were tested by later experiments, but ALL missed the actual root cause. The real problem was simpler: **zero exploration during self-play**.

| # | Hypothesis | Actual Status | Evidence |
|---|-----------|---------------|----------|
| H1 | More training cycles would eventually improve win rate | ❌ INVALIDATED | Exp 3b: 12 cycles, no trend |
| H2 | Architecture is fundamentally limited | ❓ STILL UNTESTED | Need Exp 6 results first |
| H3 | Reward shaping is too noisy with small samples | ❓ SECONDARY | Exploration bug was primary |
| H4 | Learning rate is too high | ✅ CONFIRMED | Exp 4: lower LR stabilized output |
| H5 | Pre-trained weights are harmful | ❌ INVALIDATED | Exp 4: same result from scratch |

**Missing hypothesis (the actual problem):** H0 — Zero exploration during self-play training prevents any learning.

---

### Experiment 3b: Extended Training from Checkpoint (8 Complete Cycles)

**Date:** 2026-10-18  
**Goal:** Continue training from cycle-2 checkpoint to see if pattern stabilizes

**Results:**

| Cycle | Training Error | AI Win Rate | AI Wins | Random Wins | Draws | Avg Moves |
|-------|---------------|-------------|---------|-------------|-------|-----------|
| 1 | 0.001749 | 52.0% | 52 | 45 | 3 | 167.8 |
| 2 | 0.008288 | 47.0% | 47 | 53 | 0 | 168.1 |
| 3 | 0.015468 | 51.0% | 51 | 48 | 1 | 167.8 |
| 4 | 0.018356 | 49.0% | 49 | 49 | 2 | 166.7 |
| 5 | 0.022359 | **58.0%** | 58 | 40 | 2 | 167.1 |
| 6 | 0.024062 | 46.0% | 46 | 48 | 6 | 166.8 |
| 7 | 0.023938 | 49.0% | 49 | 50 | 1 | 165.7 |
| 8 | 0.024181 | 49.0% | 49 | 49 | 2 | 167.2 |

**Combined with Experiment 3 (12 cycles total):**

The pattern is now clear across **12 independent training cycles** with **1,200 evaluation games**:

- **Mean win rate: ~50%** (range 34%-58%)
- **No upward trend line** — flat at coin-flip level
- **Training error plateaued at ~0.024** — network stopped learning

---

## Deep Analysis: Why Training Failed

> ⚠️ **CRITICAL UPDATE 2026-10-18 22:22:** This entire analysis was written BEFORE discovering the zero-exploration bug. While the technical analysis is sound, it missed the actual root cause: **self-play training had 0% random exploration after move 1**. The network was never given a chance to discover new strategies. Read this section as "what we THOUGHT the problem was" — the real problem was far simpler.

### ~~The Core Problem: Credit Assignment Impossibility~~

~~The fundamental issue is that **the reward signal provides almost no actionable learning signal**.~~

**ACTUAL CORE PROBLEM:** Self-play training had **zero random exploration** after the first move of each game. The network could only reinforce its own predictions, trapped in a closed loop. All the credit assignment analysis below is technically correct but irrelevant — the network never had diverse training data.

#### Game Structure Analysis

| Property | Value | Implication |
|----------|-------|------------|
| Grid size | 10×10×4 = 200 triangles | Massive action space (200 valid positions initially) |
| Game length | ~166 moves | 166-step credit assignment problem |
| Outcome | Binary win/loss (+1/-1) | Single bit of information for 166 decisions |
| Moves per game | ~166 | Each move responsible for ~0.6% of outcome signal |
| Claims per game | ~20-40 (estimated) | Sparse intermediate rewards |

**The math is brutal:** A binary game outcome (+1/-1) distributed linearly across 166 moves means each move's "responsibility" is worth **±0.006** on average. The immediate reward (+0.3 per claim, +0.1 per triangle) provides denser signal but only teaches **short-term tactics** (grab territory now), not **long-term strategy** (build toward enclosure).

#### Information Theoretic Perspective

If we treat the game as an information channel:
- **Outcome entropy:** 1 bit (win/loss is ~50/50 at equal skill)
- **Decision points:** 166 moves × log₂(valid_moves) ≈ 166 × 7 ≈ 1,162 bits
- **Information ratio:** 1 bit outcome / 1,162 bits decisions = **0.0009**

The network receives **0.09% of the information needed** to optimize each decision. The immediate reward helps, but it's fundamentally myopic — it optimizes for local cluster growth, not winning.

### Hyperparameter Analysis

#### Learning Rate: 0.1 (Too Aggressive)

| Rate | Effect | Evidence |
|------|--------|----------|
| 0.1 (current) | Large weight updates, overshooting minima | Win rate oscillates wildly cycle-to-cycle |
| 0.01 | Slower, more stable convergence | Standard for fine-tuning |
| 0.001 | Very conservative, may be too slow | Good for late-stage training |

The 0.1 learning rate explains the **oscillation pattern:**
- Cycle 5: 58% (lucky weight configuration)
- Cycle 6: 46% (overcorrected away from that configuration)
- Cycle 7-8: 49% (oscillating around mean)

This is classic **gradient descent overshoot** — the optimizer jumps over good minima repeatedly.

#### Exploration Factor: 0.30 (Appropriate) — ⚠️ INCORRECT ASSERTION

> **CORRECTION:** This analysis is **WRONG**. It confused `explorationFactor` (used during evaluation, where it was 0.20) with **actual exploration during self-play training**, which was **ZERO**. The code path for self-play (`runSelfPlayGame`) had no random exploration after move 1.

~~The exploration settings are reasonable:~~
- ~~**10% random moves** — prevents mode collapse, maintains diversity~~
- ~~**18% immediate-reward greedy** — provides structured exploration using tactical signal~~
- ~~**72% network exploitation** — trusts learned policy~~

**Actual self-play exploration (broken):**
- **0% random moves** after move 1 — network trapped in local optimum
- **10% immediate-reward greedy** — structured but limited
- **90% network exploitation** — reinforcing only what it already knows

~~This is not the problem. The network simply hasn't learned anything worth exploiting.~~

**This WAS the problem. Without exploration, the network could never discover new strategies.**

#### Training Iterations: 300/cycle (Insufficient)

| Iterations | Samples seen | Outcome |
|-----------|--------------|----------|
| 300 | ~8,300 | Error drops from 0.03 → 0.024, then plateaus |
| 1000+ | ~27,000+ | Would reach lower error, but is it useful? |

More iterations would lower training error, but **training error ≠ policy quality**. The network is fitting its self-play data well enough; the problem is that the self-play data doesn't encode winning strategy.

#### Games per Cycle: 50 (Too Few)

| Games | Samples | Outcome Noise |
|-------|---------|---------------|
| 50 | ~8,300 | High variance (±14% win rate CI) |
| 200 | ~33,200 | Moderate variance (±7%) |
| 500 | ~83,000 | Low variance (±4.5%) |

50 games provides only **±14 percentage points** confidence in win rate. The observed 34%-58% range is entirely explainable by noise. We need 200+ games per cycle for reliable signal.

### Reward Signal Composition Analysis

The reward structure has three components:

#### 1. Immediate Reward (Claimed from simulation)
- **+0.3 per enclosure claim** — rewards territorial control
- **+0.1 per triangle in cluster** — rewards connectivity
- **Weight: immmediacyFactor = 0.90** (90% weight)

**Problem:** This reward optimizes for **local greedy tactics**, not winning. The network learns "place triangles near existing cluster" but not "build toward enclosure" or "block opponent enclosure".

**Evidence:** The immediate reward calculation simulates every legal move (~200 at start, ~50 mid-game) to find highest-value move. This is computationally expensive and only provides ~5-10 useful training signals per game (the moments when claims happen).

#### 2. Final Outcome Reward
- **+1.0 for win, -1.0 for loss, 0.0 for draw**
- **Linearly discounted: move_t gets (t/total_moves) × outcome**

**Problem:** With 166 moves, early moves get weight ~0.006. This is **below the noise floor** of neural network gradient updates.

**Evidence:** The discounting means move 1 gets 0.006 × outcome, move 80 gets 0.5 × outcome, move 166 gets 1.0 × outcome. Only the last ~30 moves get meaningful signal (>0.2). The network learns endgame tactics but not opening/midgame strategy.

#### 3. Invalid Move Penalty
- **Score = 0 for invalid moves** (implicit in softmax)

**Effect:** The network learns not to predict high scores for full cells. This works well — we don't see the AI attempting invalid moves.

**Problem:** This is a **waste of 50% of network capacity**. Half the output neurons (200 of 400) are always invalid. The network must learn which cells are full by reading the input, then suppress those outputs — using capacity that could learn strategy.

### Self-Play Distribution Drift

Each training cycle, the AI trains on games it played against itself. This creates a **distribution drift problem:**

1. Cycle 0: AI plays with random weights, generates garbage data
2. Cycle 1: AI trains on garbage, plays slightly less randomly
3. Cycle 2: AI trains on less-garbage, but distribution shifts
4. ...
5. Cycle N: AI overfits to its own play style, loses generalization

**Evidence:** The "cycle 5 spike to 58%" followed by decline. The network briefly found a weight configuration that worked against random, then self-play training pushed it toward a different style that doesn't generalize.

This is the **"alphaGo zero" problem without the solution:** AlphaGo Zero used:
- 1,600+ simulations per move (we use 1)
- Thousands of games per iteration (we use 50)
- Residual CNN architecture (we use MLP)
- 4,000,000 games total (we use ~1,000)

We're trying to do RL with **0.025% of the sample efficiency**.

### Architecture Bottleneck

The 401→200→400 MLP has ~240,000 weights. Let's analyze what it's learning:

#### Input Encoding (401 features)
- 400 cell states (empty, player1, player2, in-progress)
- 1 current player indicator

**What the network must infer:**
- Which cells are adjacent (grid topology)
- Which triangles share corners (corner topology)
- Who owns which clusters
- Which clusters are adjacent
- Enclosure threats
- Claim opportunities

**Problem:** The network must **learn spatial reasoning from scratch**. A convolutional network would have this built-in. The MLP must waste its first layer just figuring out that `(0,0)` and `(0,1)` are adjacent.

#### Hidden Layer (200 neurons)
- ReLU activations
- Must learn all spatial patterns, cluster recognition, strategy

**Problem:** Single layer cannot build hierarchical representations. Complex strategy ("if I claim here, opponent must respond there, opening up this") requires multi-step reasoning.

#### Output Layer (400 scores)
- Sigmoid activations (0-1 range)
- One score per cell/rotation

**Problem:** 50% of outputs are always invalid. Network wastes capacity learning which cells to suppress.

### Summary: Root Cause Hierarchy (OUTDATED — See Critical Bug Below)

> ⚠️ **CORRECTION:** This hierarchy was created before discovering the zero-exploration bug. The actual root cause ranking should be:
> 
> | Rank | Root Cause | Evidence |
> |------|-----------|----------|
> | **1** | **Zero exploration during training** | Code inspection: ε = 0 after move 1 |
> | 2 | Credit assignment difficulty | Still valid, but secondary |
> | 3 | No spatial inductive bias | Still valid, but secondary |
> | 4 | Learning rate too high | Fixed in Exp 4, didn't solve it |
> | 5 | Sample inefficiency | Still valid, but secondary |
>
> **The original hierarchy below was WRONG because it missed the fundamental bug.**

| Rank | Root Cause | Evidence | Fix Effort |
|------|-----------|----------|------------|
| 1 | **Credit assignment impossible** | 1 bit outcome / 1162 bits decisions | Add dense shaping rewards |
| 2 | **No spatial inductive bias** | MLP must learn topology from scratch | Switch to CNN (high effort) |
| 3 | **Learning rate too high** | Win rate oscillates 34-58% | Lower to 0.01 |
| 4 | **Sample inefficiency** | 1,000 games vs AlphaGo's 4M | Increase 10× (more time) |
| 5 | **Wasted capacity on invalid moves** | 200/400 outputs always invalid | Mask outputs |
| 6 | **Distribution drift** | Self-play without experience replay | Add replay buffer |
| 7 | **Noisy evaluation** | 50 games ±14% CI | Use 200+ games |

---

## Revised Experiment Plan

Based on the analysis, here's a prioritized plan that addresses root causes:

### Phase 1: Quick Wins (No Architecture Change)

| Exp | Change | Tests | Expected Outcome |
|-----|--------|-------|------------------|
| E1 | **Lower learning rate to 0.01** | H4 (LR too high) | More stable convergence |
| E2 | **Train from scratch** | H5 (bad init) | Faster convergence |
| E3 | **200 games/cycle, 200 eval games** | H3 (noise) | Reliable win rate signal |
| E4 | **Higher iterations (1000)** | Training completeness | Lower error plateau |

**Implementation:** Single combined experiment with all 4 changes.

### Phase 2: Reward Engineering

| Exp | Change | Expected Outcome |
|-----|--------|------------------|
| R1 | **No outcome discounting** (all moves get full outcome) | Amplifies sparse signal |
| R2 | **Opponent cluster penalty** (-0.05 per opponent triangle) | Learns to block |
| R3 | **Enclosure threat bonus** (+0.5 for completing 3/4 sides) | Learns enclosure strategy |

### Phase 3: Architecture Changes

| Exp | Change | Effort | Expected Outcome |
|-----|--------|--------|------------------|
| A1 | **Output masking** (zero invalid cells) | Low | Frees 50% capacity |
| A2 | **2 hidden layers** (200→100→400) | Medium | Hierarchical features |
| A3 | **CNN architecture** (TensorFlow.js) | High | Spatial inductive bias |

---

### Experiment 4: Train from Scratch with Better Hyperparameters

**Goal:** Test H4 + H5 simultaneously (lower LR + fresh start)

**Hyperparameters:**
| Parameter | Value | Change |
|-----------|-------|--------|
| `learningRate` | 0.01 | ↓ from 0.1 |
| `explorationFactor` | 0.20 | ↓ from 0.30 |
| `iterations` | 500 | ↑ from 300 |
| `games/cycle` | 100 | ↑ from 50 |
| `eval games` | 200 | ↑ from 100 |
| `startFrom` | null | Fresh random init |

**Results (5 complete cycles before termination):**

| Cycle | Training Error | AI Win Rate | AI Wins | Random Wins | Draws | Avg Moves |
|-------|---------------|-------------|---------|-------------|-------|-----------|
| 1 | 0.002321 | 49.0% | 98 | 96 | 6 | 168.5 |
| 2 | 0.009659 | 52.0% | 104 | 92 | 4 | 167.5 |
| 3 | 0.011783 | 45.5% | 91 | 102 | 7 | 168.4 |
| 4 | 0.014564 | 47.5% | 95 | 102 | 3 | 167.9 |
| 5 | 0.015486 | 51.5% | 103 | 90 | 7 | 167.8 |

**Statistics:**
- Mean win rate: **49.1%**
- Std deviation: **2.6%**
- Range: 45.5-52%
- Total evaluation games: **1,000**

**Conclusion:** Experiment terminated after 5 cycles. No improvement trend detected. The lower learning rate eliminated the wild 34-58% oscillations from previous experiments, but the result is still a flat line at 50%. The network has converged to predicting random-like scores for all moves.

> **UPDATE:** This conclusion was premature. The actual problem was **zero exploration during self-play**, not the architecture or hyperparameters. Exp 6 will test this hypothesis.

---

## Reward Signal Analysis: The Real Bottleneck — 2026-10-18

### User Observation

After reviewing the failure analysis, the user noted: "I had normalized to 1, but it wasn't meant to be 1 bit. If it is really 1 bit then there was probably a mistake in my variable declaration."

### Investigation: Tracing the Reward Calculation

#### Immediate Reward (Correctly Implemented)

```javascript
const claimRewardFactor = 0.3;
const connectedClusterRewardFactor = 0.1;
const maxPossibleClaimBonus = 1.5;
const maxPossibleClusterSize = rows * cols * 2;  // 200
const maxPossibleClusterBonus = 200 * 0.1;       // 20
const maxImmediateReward = 1.5 + 20;             // 21.5
const rewardRange = 21.5;

// For a typical move (1 claim, cluster size 5):
// hypotheticalReward = 0.3 + 0.5 = 0.8
// scaledReward = 0.8 / 21.5 ≈ 0.037
```

**Verdict:** Immediate rewards are **NOT** 1 bit — they're continuous values in [0.01, 0.99] with meaningful granularity. ✅

#### Final Outcome Addition (Problematic)

```javascript
const discountFactor = (t + 1) / totalMoves;
const scaledFinalReward = finalRewardValue * discountFactor * finalRewardWeight;
const combinedReward = immediateReward + scaledFinalReward;
```

### The Real Issue: Linear Discounting Kills Early-Move Signal

| Move Position | Discount Factor | Final Reward Contribution | Immediate Reward | Combined |
|---------------|-----------------|--------------------------|------------------|----------|
| Move 1/166 | 0.006 | ±0.006 | ~0.04 | ~0.04 |
| Move 50/166 | 0.30 | ±0.30 | ~0.04 | ~0.34 |
| Move 83/166 | 0.50 | ±0.50 | ~0.04 | ~0.54 |
| Move 166/166 | 1.00 | ±1.00 | ~0.04 | ~1.04 |

**Problem:** First ~50 moves (where opening strategy matters) receive less than ±0.3 contribution from the game outcome. The immediate reward (~0.04) dominates, which only teaches short-term tactics.

### Why This Matters

1. **Opening moves are critical** — Triantica is a territory control game where early positioning determines late-game options
2. **Linear discounting is too aggressive** — Standard RL uses exponential discounting γ^t where γ ≈ 0.99, not linear scaling
3. **Signal-to-noise ratio** — A ±0.006 contribution is below the gradient noise floor for sigmoid activations

### Proposed Fix

**Option A:** Remove discounting entirely — all moves get full ±1.0 outcome signal
**Option B:** Exponential discounting with γ = 0.99
**Option C:** Flat reward — all moves get same final outcome regardless of position

**Recommendation:** Start with **Option A** (remove discounting) as the simplest test. If that works, the signal was the problem. If it still fails, we know architecture is the bottleneck.

---

### Critical Bug: Zero Random Exploration During Self-Play — 2026-10-18

**Discovery:** While reviewing the heuristic rules, we discovered that **self-play training has NO random exploration** after the first move of each game.

#### Code Analysis (BEFORE FIX)

```javascript
if (this.isNetworkLoaded && movesCount > 0) {
  if (Math.random() >= this.immediacyFactor) {
    move = this._predictMoveFromRewards(...);  // 10% greedy heuristic
  } else {
    move = this._getBestNetworkMove(...);      // 90% network
  }
} else {
  move = this._getRandomValidMove(...);  // Only for first move!
}
```

**With `immediacyFactor = 0.9`:**
- 90% network moves
- 10% greedy heuristic
- **0% random moves** (after move 1)

#### Why This Breaks RL

1. **No discovery:** The network can only reinforce what it already knows
2. **Local optimum trap:** It converges to "predict scores that match my greedy heuristic"
3. **Self-reinforcing mediocrity:** If the network starts bad, it trains on bad games and stays bad

Compare to successful RL:
| System | Random Exploration |
|--------|-------------------|
| AlphaGo Zero | Dirichlet noise on MCTS policy |
| Standard Q-learning | ε-greedy with ε ≈ 0.1-0.3 |
| Our training (before fix) | **ε = 0** |

#### Fix Applied

```javascript
// NEW: Add 10% random exploration
const rand = Math.random();
if (rand < 0.10) {
  move = this._getRandomValidMove(gridManager);  // Random discovery
} else if (rand < 0.20) {
  move = this._predictMoveFromRewards(...);      // Greedy heuristic
} else {
  move = this._getBestNetworkMove(...);          // Network exploitation
}
```

**New distribution:**
- 10% random (discovery)
- 10% greedy heuristic (structured exploration)
- 80% network (exploitation)

#### Why This Matters

This single bug could explain **all** training failures:
- Exp 2: No exploration → stuck in local optimum
- Exp 3: No exploration → wild oscillations (sensitivity to initial weights)
- Exp 4: No exploration → stable flatline at 50%
- Exp 5: No exploration → same 50% plateau

**This is the most important fix so far.**

---

## Re-evaluation of Previous Experiments — 2026-10-18

### What We Can Still Trust

| Finding | Validity | Reason |
|---------|----------|--------|
| Pre-trained network is ~50% vs random | ✅ Valid | Evaluation games had 20% exploration, so measured correctly |
| Lower LR (0.01) eliminated wild oscillations | ✅ Valid | Exp 4 was smoother than Exp 2/3 — stabilization effect is real |
| Training error dropping doesn't predict win rate | ✅ Valid | Low error with no performance gain holds across all experiments |

### What's Now Questionable

| Finding | Status | Reason |
|---------|--------|--------|
| "Architecture cannot learn this game" | ❓ Inconclusive | Never tested with proper exploration — network may have been capable but starved of diverse data |
| "Reward signal too sparse" | ❓ Inconclusive | No-discounting fix (Exp 5) showed brief 55.5% spike but couldn't capitalize due to exploration bug |
| "Discounting kills early-move signal" | ❓ Untested | Fix applied simultaneously with exploration fix — effects confounded |

### The Confounding Problem

Experiments made overlapping changes without isolation:

```
Exp 1-3: Discounting ON,  Exploration OFF → 50% win rate
Exp 4:   Discounting ON,  Exploration OFF, Lower LR → 50% win rate
Exp 5:   Discounting OFF, Exploration OFF → 50% win rate (brief 55% spike)
Exp 6:   Discounting OFF, Exploration ON → ? (running now)
```

We cannot isolate:
- Whether no-discounting helps (confounded with exploration)
- Whether exploration alone would have fixed Exp 1-3
- Whether architecture is truly the bottleneck

### Decision: Don't Go Back

**Rationale:** The exploration bug is so fundamental that re-testing old configs would be academic waste.

**Correct path forward:**
1. Let Exp 6 complete — tests "both fixes" vs baseline
2. If Exp 6 succeeds (60%+ win rate): Problem solved, move on
3. If Exp 6 fails (50-55%): THEN architecture is likely the bottleneck

**Most likely outcome:** Exp 6 will show significant improvement, revealing that the exploration bug was the primary cause of all failures. Lower LR and no-discounting were secondary optimizations.

---

### Experiment 5: Training with Fixed Reward Signal (No Discounting)

**Date:** 2026-10-18
**Goal:** Test whether removing linear discounting improves learning

**Change:** `discountFactor = (t + 1) / totalMoves` → `discountFactor = 1.0`

**Expected effect:** All moves now receive full ±1.0 outcome contribution instead of ±0.006 for early moves. This amplifies the final-outcome signal by **~170×** for opening moves.

**Hyperparameters:** Same as Experiment 4
- `learningRate`: 0.01
- `explorationFactor`: 0.20
- `iterations`: 500
- `games/cycle`: 100
- `eval games`: 200

**Status:** Ready to run

### Experiments Summary

| Experiment | Cycles | Self-Play Wins | vs Random | Outcome |
|-----------|--------|----------------|-----------|----------|
| Pre-trained vs Random | — | — | 51% | Baseline: coin-flip |
| Exp 2: From checkpoint | 8 | — | 46-58% | Oscillated, no trend |
| Exp 3: From checkpoint (extended) | 4 | — | 34-50% | Wild oscillations |
| Exp 4: From scratch (lower LR) | 5 | — | 45-52% | Stable but still 50% |
| Exp 5: Fixed reward signal | 4 | — | 48-55% | Brief spike |
| **Exp 6: Exploration fix** | 10 | P1: 100% | 48.5-55.5% | **BUG: asymmetric self-play** |

**Total:** 30+ training cycles, 3,500+ evaluation games.

**Critical Discovery (2026-10-19):** Exp 1-6 all had the self-play asymmetry bug where Player 2 followed Player 1's policy. The "no improvement" conclusion was premature.

### Why Current Approach Fails

**Root cause confirmed:** The 401→200→400 MLP architecture combined with sparse final-outcome rewards **cannot solve the credit assignment problem** for this game.

| Factor | Status | Evidence |
|--------|--------|----------|
| Learning rate | ✅ Fixed | Lower LR stabilized output |
| Pre-trained weights | ✅ Ruled out | Fresh start performed same |
| Sample size | ✅ Adequate | 1,000 eval games, consistent results |
| Architecture | ❌ Problematic | No spatial bias, flat input |
| Reward signal | ❌ Problematic | Sparse outcome, weak intermediate |
| Self-play drift | ❌ Problematic | No experience replay |

### ~~Pivot Decision~~ — PREMATURE (Critical Bug Discovered)

~~**Stop training with current architecture.** After 17 cycles with zero improvement, further training has expected value of zero.~~

> **CORRECTION:** This decision was made BEFORE discovering the zero-exploration bug. We should NOT have pivoted — we should have fixed the bug first.
>
> **Lesson learned:** Always verify basic training dynamics (exploration, learning rate, data diversity) before concluding architecture is the problem.

**Current status:** Experiment 6 is now running with both fixes applied (no discounting + 10% exploration). Training should continue to completion before making any architecture decisions.

**Next step: Reward Engineering (Phase 2)** — Add dense intermediate rewards that actually teach strategy. This requires modifying `AIPlayer.js` to compute:

1. **Cluster growth delta** — Reward moves that grow connected territory
2. **Enclosure progress** — Bonus for completing 3/4 sides of an enclosure
3. **Opponent blocking** — Penalty for allowing opponent cluster growth
4. **Territory control** — Track who controls more cells over time

This provides ~50 bits of dense signal per move instead of 1 bit per game.

---

## Standard Experiment Format

> **All experiments use this JSON structure:**

```json
{
  "cycles": [{
    "cycle": 1,
    "selfPlay": {
      "games": 100,
      "wins": {"1": 4, "2": 6, "draws": 0},
      "avgMoves": 165.2,
      "samplesGenerated": 1653,
      "durationMs": 2422
    },
    "training": {
      "iterations": 500,
      "learningRate": 0.01,
      "samplesUsed": 842,
      "initialError": 0.077,
      "finalError": 0.0017,
      "durationMs": 13238
    },
    "evaluation": {
      "games": 200,
      "opponent": "random",
      "aiWins": 106,
      "opponentWins": 89,
      "draws": 5,
      "aiWinRate": 0.53,
      "avgMoves": 166.9,
      "durationMs": 9793
    }
  }]
}
```

**Key metrics to track:**
- Self-play win distribution (should be ~50-50 in fair game)
- Training error delta (initial → final, shows learning rate)
- vs Random win rate (primary performance metric)

---


---

## Performance Metrics

### What to Measure

| Metric | Description | How to Capture |
|--------|-------------|----------------|
| **Win Rate** | AI vs random opponent | Play 100 games, count wins |
| **Training Error** | Network loss per iteration | Log from brain.js train() |
| **Game Length** | Moves per game | Count during self-play |
| **Decision Time** | Ms per move | Performance.now() around predictMove() |
| **Claim Rate** | Average claims per game | Track during games |
| **Cluster Size** | Average final cluster | Score at game end |

### Current Baseline: Established ✅

| Metric | Value |
|--------|-------|
| AI Win Rate vs Random | ~51% (coin-flip) |
| Random vs Random P1 Win Rate | ~44% |
| Average Game Length | ~166 moves |
| Average Game Duration (Node.js) | ~47ms |
| Network Size | 401→200→400 (medium) |

---

## Lessons Learned

### 2026-10-19: Self-Play Asymmetry Bug (CRITICAL)

> ⚠️ **BUG DISCOVERY:** The network wasn't just playing against itself — **Player 2 was actively trying to lose**. The `currentIsSelf` variable was computed but never used, so BOTH players followed the same network predictions.

**The problem:**
```javascript
const currentIsSelf = currentPlayerNum === this.playerNumber;  // Computed but NEVER USED

// Both players use the network!
if (this.isNetworkLoaded && movesCount > 0) {
  move = this._getBestNetworkMove(gridManager);  // Player 2 follows this too!
}
```

Since the network predicts "what move benefits Player 1", Player 2 was literally being told which moves would help Player 1 win — and following that advice!

**Evidence:**
- Self-play: Player 1 wins 100% of games (statistically impossible in a fair game)
- vs Random: Only 50-55% win rate (because it learned to beat a cooperative opponent)

**The fix:** Gate network usage with `currentIsSelf`:
```javascript
if (!currentIsSelf) {
  move = this._getRandomValidMove(gridManager);  // Opponent plays random
} else if (this.isNetworkLoaded && movesCount > 0) {
  move = this._getBestNetworkMove(gridManager);  // Self uses network
}
```

**Rule:** In self-play RL, ALWAYS verify both players are playing correctly. Check win rate symmetry.

### 2026-10-19: Proper Self-Play Architecture

The fix involves two changes to support proper self-play:

**1. State representation must be currentPlayer-relative:**

When Player 2 moves, they shouldn't see themselves as "opponent" and Player 1 as "self". The network would then predict moves that help Player 1!

```javascript
// Before (broken): Always relative to training player
return p === this.playerNumber ? 1 : -1;

// After (for 'network' mode): Relative to current player
return p === currentPlayer ? 1 : -1;
```

**2. Immediate rewards must be computed from the mover's perspective:**

The reward vector tells the network "how good is each move for ME". For self-play, "ME" should be whoever is currently moving.

**Training modes:**

| Mode | Self | Opponent | Training Data |
|------|------|----------|---------------|
| `random` | Network | Random | Self moves only |
| `network` | Network | Network (shared policy) | Both players' moves |

**Which is better?**

- `random` is simpler and guarantees honest opposition
- `network` is theoretically optimal (AlphaGo approach) but requires correct implementation

Recommendation: Start with `random` for curriculum, switch to `network` once network beats random consistently.

---

## Experiment Results

> **Standard Format for All Experiments:** Each experiment logs:
> - Self-play win distribution (P1/P2/Draws)
> - Training error progression (start → end of cycle)
> - Evaluation opponent type and win rate

### Experiment 6: Training with Exploration (BROKEN SELF-PLAY)

**Date:** 2026-10-18 to 2026-10-19  
**Status:** Completed but revealed critical bug  
**Mode:** `random` (but bug affected both players)

**Configuration:**
```
--cycles 10 --games 100 --eval-games 200
--iterations 500 --learning-rate 0.01
--exploration 0.20
```

**Results:**

| Cycle | Self-Play (P1/P2/D) | vs Random | Train Error Start | Train Error End |
|-------|---------------------|-----------|-------------------|-----------------|
| 1 | 100/0/0 | 48.5% | 0.00267 | 0.00267 |
| 2 | 100/0/0 | 55.5% | 0.00917 | 0.00917 |
| 3 | 100/0/0 | 48.5% | 0.0120 | 0.0120 |
| 4 | 100/0/0 | 52.0% | 0.0136 | 0.0136 |
| 5 | 100/0/0 | 55.5% | 0.0163 | 0.0163 |
| 6 | 100/0/0 | 51.5% | 0.0161 | 0.0161 |
| 7 | 100/0/0 | 55.5% | 0.0199 | 0.0199 |
| 8 | 100/0/0 | 52.0% | 0.0187 | 0.0187 |
| 9 | 100/0/0 | 55.0% | 0.0174 | 0.0174 |
| 10 | 100/0/0 | 51.0% | 0.0184 | 0.0184 |

**Key Finding:** Player 1 wins 100% of self-play games — statistically impossible in a fair game.

**Bug Confirmed:** Both players used the same network (predicting "good moves for Player 1"), so Player 2 was actively trying to lose.

---

### 2026-10-19: Proper Self-Play Architecture

The fix involves two changes to support proper self-play:

**1. State representation must be currentPlayer-relative:**

When Player 2 moves, they shouldn't see themselves as "opponent" and Player 1 as "self". The network would then predict moves that help Player 1!

```javascript
// Before (broken): Always relative to training player
return p === this.playerNumber ? 1 : -1;

// After (for 'network' mode): Relative to current player
return p === currentPlayer ? 1 : -1;
```

**2. Immediate rewards must be computed from the mover's perspective:**

The reward vector tells the network "how good is each move for ME". For self-play, "ME" should be whoever is currently moving.

**Training modes:**

| Mode | Self | Opponent | Training Data |
|------|------|----------|---------------|
| `random` | Network | Random | Self moves only |
| `network` | Network | Network (shared policy) | Both players' moves |

**Which is better?**

- `random` is simpler and guarantees honest opposition
- `network` is theoretically optimal (AlphaGo approach) but requires correct implementation

Recommendation: Start with `random` for curriculum, switch to `network` once network beats random consistently.

---

### 2026-10-18: Critical Debugging Lesson

> ⚠️ **THE MOST IMPORTANT LESSON:** When RL training fails, **CHECK EXPLORATION FIRST**. We spent 17 training cycles analyzing architecture, hyperparameters, and reward signals before discovering the simplest possible bug: **the network had no way to discover new strategies**.

**What we did wrong:**
1. Assumed exploration was working (it had a parameter!)
2. Traced reward signals, architecture, hyperparameters
3. Never traced the actual training loop

**What we should have done:**
1. Trace the training loop FIRST — verify exploration is happening
2. Log exploration statistics (how many random moves per game?)
3. Only then analyze architecture and hyperparameters

**Rule of thumb:** In RL, if your network isn't learning, the most likely cause is **not enough exploration**. It's the RL equivalent of "is it plugged in?" — and we didn't check.

### 2026-10-18: Initial Analysis (OUTDATED — See Above)

1. **Reward shaping complexity**: The immediate reward calculation simulates every legal move, creating training labels that reflect immediate tactical value. This is computationally expensive but provides dense learning signal.

2. **Two-level exploration**: The AI has both random exploration (10%) and "immediacy-based" exploitation (20% of 90%), choosing between learned policy and greedy immediate reward maximization.

3. **Linear discounting**: Early moves get small fractions of the final reward. Alternative: exponential discounting (γ^t) or no discounting.

4. **Network size grows with difficulty**: Hard mode has 2 hidden layers, suggesting the problem benefits from deeper representations.

### 2026-10-18: Training Results & Architecture Analysis

5. **Rising training error ≠ worsening policy**: During the "unlearning" phase, training error increases as the optimizer destroys memorized patterns from previous (poor) self-play. This is expected but makes it hard to gauge progress.

6. **Win rate is the only reliable metric**: Training error in RL is deceptive. A low error just means the network fits its training data, not that the data represents good play. Always evaluate against a fixed opponent.

7. **Small evaluation sets are misleading**: 50 games at ~50% win rate has a 95% CI of ±14%. The apparent 56%→34% decline could partly be noise. Use 200+ games for evaluation.

8. **brain.js is CPU-bound in Node.js**: Training 8K samples × 300 iterations takes ~12 min/cycle on M2 Ultra. This is feasible but slow for extensive hyperparameter search. A Web Worker or child_process pool would help.

9. **Pre-trained weights can be harmful**: The existing network had memorized near-random play with very low error. Starting from scratch may converge faster than unlearning bad habits.

10. **Spatial structure matters**: A flat 401-element input vector removes the 2D grid topology that's essential for understanding adjacency, enclosure, and cluster growth. This is the single biggest architectural limitation.

11. **Random self-play produces constant reward signal (CRITICAL)**: In `random` mode, P1 wins ~100% of games vs random P2. This means the reward signal is always +1 for P1 — the network cannot distinguish good moves from bad moves because every move is followed by a win. This is equivalent to trying to learn chess by only playing opponents who resign on move 1. **Random mode is fundamentally broken for RL training.**

12. **Self-play win rate is a misleading metric in random mode**: A 100% self-play win rate in `random` mode does NOT mean the AI is playing well. It means the opponent is trivially weak. Always validate against a separate evaluation opponent.

13. **Catastrophic forgetting without experience replay**: Training on new self-play data each cycle without mixing in old data causes the network to completely forget previous learning. Initial error resets to ~0.09 each cycle regardless of training progress. An experience replay buffer is essential.

14. **Training time scales linearly with sample count in brain.js**: No GPU acceleration, no batching, no parallelism. Each sample is processed sequentially. 24K samples takes 3× as long as 8K samples.

15. **Network mode doubles training data**: In `random` mode, only P1's moves become training data (~8.7K). In `network` mode, both players use the network, so both players' moves have meaningful signals (~16.5K). This alone doubles the data.

16. **Replay buffer further increases sample count**: A 50% replay ratio adds 8K old samples on top of 16K new ones, bringing total to ~24.6K. Combined with network mode, this is 2.8× the data of random mode.

17. **Over-training risk**: 500 iterations × 24K samples = each sample seen 500 times. AlphaGo-style training typically does 1-3 passes over data, not 500. Consider reducing to 50-100 iterations. The network will fit training data faster but may not generalize — balance convergence speed vs overfitting.

19. **Training/inference state mismatch (CRITICAL)**: In `network` mode, training data uses `_getStateRelativeToCurrentPlayer` (current player = +1) but `_getBestNetworkMove` and `getMove` use `_getStateRepresentation` (P1 always = +1). This means P2's training data is encoded differently than P2's inference, causing the network to see a "mirrored" board at inference time. The fix: use the same state function for both training and inference.

20. **Conflicting gradients from mismatched state**: When the network is trained on P1-relative AND P2-relative states but only infers with P1-relative, the P2 training data creates gradients that fight the P1 training data, preventing learning entirely. This explains why Exp 7 and Exp 8 showed zero improvement.
    - Random mode, no replay: ~43 min/cycle (8.7K × 500 iter)
    - Network mode, 50% replay: ~128 min/cycle (24.6K × 500 iter)
    - Network mode, 25% replay, 50 iter: ~30 min/cycle (estimated)
    - Network mode, 25% replay, 100 iter: ~50 min/cycle (estimated)

---

## Experiment 7 Results (2026-10-19)

**Config:** 10 cycles, 100 games/cycle, 200 eval games, `--self-play-mode random`, lr=0.01, exploration=0.2
**Duration:** 7 hours 4 minutes

### The Numbers

| Cycle | Eval Win Rate | Train Error (init→final) |
|-------|--------------|--------------------------|
| 1 | 50.5% | 0.079 → 0.002 |
| 2 | **56.0%** (peak) | 0.093 → 0.007 |
| 3 | 50.5% | 0.092 → 0.010 |
| 4 | 50.5% | 0.096 → 0.011 |
| 5 | 53.0% | 0.094 → 0.013 |
| 6 | 48.5% | 0.097 → 0.015 |
| 7 | 48.5% | 0.096 → 0.016 |
| 8 | 50.0% | 0.094 → 0.017 |
| 9 | 53.0% | 0.094 → 0.019 |
| 10 | 52.5% | 0.098 → 0.020 |

### Diagnosis

**Three fatal problems identified:**

1. **Constant reward signal**: In `random` mode, P1 wins 100% of self-play games. Every move gets labeled as "good" (reward ≈ +1). The network has zero gradient to distinguish good from bad moves.

2. **Catastrophic forgetting**: Initial error resets to ~0.09 every cycle. The network trains down to low error within a cycle, but the next cycle's data overwrites everything. No experience replay buffer.

3. **Network plays randomly**: Verified by running the final v7 network with 0% exploration vs random — 51% win rate (statistical noise). The network learned nothing in 7 hours.

### Verification

- Random vs Random (500 games): P1=47.8%, P2=49.0% — **no first-mover advantage** in Triantica
- v7 network (0% exploration) vs Random (200 games): 51% — **coin flip**
- Self-play `random` mode: P1 wins 100% — but this is meaningless (opponent is trivially weak)
- Self-play `network` mode: P1=40%, P2=60% — **balanced**, meaningful signal

### Conclusion

**`random` self-play mode is fundamentally broken for RL.** The opponent is too weak to produce informative win/loss signals. Must use `network` mode where both players compete.

---

## Potential Improvements

### Opening Diversity (Phase 3)

**Problem:** With 20% exploration, ~80% of games start from similar positions. Network may overfit to specific openings and be weak elsewhere.

**Proposed fix:** Higher random rate for early moves:

```javascript
const openingMoves = 5;  // First N moves
const openingExploration = 0.50; // 50% random for openings
const normalExploration = 0.20;

const explorationRate = movesCount < openingMoves 
  ? openingExploration 
  : normalExploration;
```

**Decision:** Implement for Experiment 8 alongside network mode.

### Experience Replay Buffer (Phase 2 — NOW)

**Problem:** Catastrophic forgetting — new training data overwrites old learning each cycle.

**Proposed fix:** Maintain a replay buffer of past training samples:

```javascript
class ReplayBuffer {
  constructor(maxSize = 50000) { this.buffer = []; this.maxSize = maxSize; }
  add(samples) {
    this.buffer.push(...samples);
    if (this.buffer.length > this.maxSize) {
      this.buffer = this.buffer.slice(-this.maxSize);
    }
  }
  sample(n) {
    // Mix 50% old + 50% new
    const fromBuffer = Math.min(Math.floor(n/2), this.buffer.length);
    const oldSamples = shuffle(this.buffer).slice(0, fromBuffer);
    return oldSamples; // caller mixes with new samples
  }
}
```

**Decision:** Implement for Experiment 8.

---

## Future Experiments

### Experiment 8 — In Progress (2026-10-19)

**Config:** 10 cycles, 100 games/cycle, 200 eval games, `--self-play-mode network`, lr=0.01, exploration=0.3, replay-ratio=0.5, replay-size=50000
**Log:** `experiments/logs/exp8.log`
**ETA:** ~09:00 CET 2026-10-20

**Key changes from Exp 7:**
1. Network mode (competitive self-play with meaningful reward signal)
2. Experience replay buffer (prevent catastrophic forgetting)
3. Per-cycle log file for real-time monitoring

**Cycle 1 result:** Self-play P1=32 P2=68 (competitive!), Eval 48.5%, ~128 min/cycle

**Pending decision after Exp 8 results:**
- Reduce iterations (500→50-100) to speed up future experiments
- Reduce replay ratio (0.5→0.25) for faster cycles
- Opening diversity (50% random for first 5 moves)

### Priority 1: Establish Baseline ✅

- [x] Extract game logic into platform-agnostic GridManager
- [x] Build testing framework (PlayerInterface, GameSession, RandomPlayer)
- [x] Refactor AIPlayer to work with new architecture
- [x] Run 100 games vs random opponent → 51% (needs retraining)
- [x] Measure average decision time → ~47ms/game in Node.js
- [x] Document current network performance → coin-flip

### Priority 2: Architecture Experiments

| Experiment | Hypothesis | Method |
|------------|------------|--------|
| Smaller network | 50-neuron hidden layer sufficient for basics? | Train from scratch, compare win rate |
| Larger network | Does 400→400 improve over 200? | Double hidden size, same training |
| Deeper network | Does 3-layer help? | Add third hidden layer |
| No hidden layers | Can linear separator work? | Single-layer perceptron |

### Priority 3: Training Experiments

| Experiment | Hypothesis | Method |
|------------|------------|--------|
| Pure RL | Remove immediate reward shaping | Zero out immediate rewards, train longer |
| Different discount | Exponential (γ=0.99) better than linear? | Modify reward processing |
| Higher exploration | 30% exploration improve learning? | Increase explorationFactor |
| Curriculum learning | Start with small board, grow to 10×10? | Progressive board sizes |

### Priority 4: Performance Optimization

- [ ] Web Worker for background training
- [ ] Cache immediate reward calculations
- [ ] Prune invalid moves before simulation
- [ ] Batch training updates

---

## Appendix

### File Structure

```
triantica/
├── index.html                 # Entry point
├── game.js                    # GridManager + GameScene
├── AIPlayer.js                # Neural network AI
├── trainedAI_network.json     # Pre-trained weights
├── trainedAI_network_backup.json
└── docs/
    └── TRAINING_BOOK.md       # This document
```

### Key Classes

- `GridManager`: Rules engine, state tracking, cluster computation
- `AIPlayer`: Neural network, training, prediction
- `GameScene`: Phaser rendering and input

### Brain.js API Reference

```javascript
// Create network
const net = new brain.NeuralNetwork({
  inputSize: 401,
  hiddenLayers: [200],
  outputSize: 400,
  activation: 'sigmoid'
});

// Train
const result = net.train(trainingData, {
  iterations: 500,
  learningRate: 0.1,
  errorThresh: 0.00093
});

// Predict
const output = net.run(stateVector);

// Save/Load
const json = net.toJSON();
net.fromJSON(json);
```

---

## Session 2026-10-19 Evening: The Shallow Copy Bug

### Discovery
After fixing the state mismatch bug, experiments still showed no learning. Deep investigation revealed cluster sizes exploding to 10^42 for 200 triangles.

### Root Cause
`GridManager.clone()` used shallow copy for `this.clusters`:
```javascript
// BUG:
cloned.clusters = { ...this.clusters };
```

When `calculateImmediateRewardVector()` simulated moves on cloned grids, it modified the **original** grid's clusters via shared references.

### Impact
- Every simulation (400+ per move) corrupted original clusters
- Scores became astronomical (10^20 vs 10^25)
- Reward signal was pure noise
- Training could not possibly learn anything

### Fix
```javascript
// DEEP COPY:
cloned.clusters = Object.fromEntries(
  Object.entries(this.clusters).map(([k, v]) => [k, { ...v }])
);
```

### Verification
```
Before fix: P1=10^44, P2=10^42, Total=200 (corrupted)
After fix:  P1=35, P2=90, Total=200 ✓
```

### Lesson
**Simulation isolation is critical.** Always deep-copy nested objects. A shallow copy is a subtle bug that corrupts state silently.

---

**Next Session:** Run Experiment 9 with all fixes applied.

---

## Session 2026-10-20 00:35: Bug #3 — Eval State Mismatch

### Bug Found
`AIPlayer.getMove()` used different state representations based on `selfPlayMode`:
```javascript
const state = this.selfPlayMode === 'network'
  ? this._getStateRelativeToCurrentPlayer(gridManager)  // relative (correct)
  : this._getStateRepresentation(gridManager);           // absolute (wrong!)
```

Eval player didn't set `selfPlayMode`, so it defaulted to 'random' and used **absolute state**.
Network was trained on **relative state** — receiving absolute state meant garbage input.

### Fix
Always use relative state for network inference:
```javascript
const state = this._getStateRelativeToCurrentPlayer(gridManager);
```

### Impact
- Before: 45.5% win rate vs random (below baseline)
- After: 50.0% win rate vs random (at baseline)

### Remaining Puzzle
Self-play scores (73/72) are double eval scores (37/37). Possible causes:
1. Strategic play creates larger clusters
2. Random play fragments clusters
3. Different exploration rates (self-play: 30%, eval: 5%)

---

---

## Session 2026-10-20 00:15: Experiment 9 Preparation

### Fixes Implemented

| Fix | Status | Description |
|-----|--------|-------------|
| Deep copy in `clone()` | ✅ | Prevents cluster corruption during simulation |
| Opening diversity injection | ✅ | 50% random moves for first 5 moves |
| Score metrics tracking | ✅ | Median, variance, min, max for scores |
| Score gap tracking | ✅ | AI vs Random score difference in eval |

### New Metrics Added

| Metric | Purpose |
|--------|---------|
| **Self-play score median** | Robust central tendency (not skewed by outliers) |
| **Self-play score variance** | Diversity of game outcomes |
| **Self-play score min/max** | Sanity check: scores in valid range |
| **Eval AI score median** | AI scoring performance |
| **Eval Random score median** | Opponent scoring performance |
| **Eval score gap** | AI domination metric (should grow over training) |

### Opening Diversity Injection

```javascript
const isOpeningPhase = movesCount < 5;
const openingRandomFactor = isOpeningPhase ? 0.50 : 0;

if (rand < openingRandomFactor) {
  // Opening diversity: random move
  move = this._getRandomValidMove(gridManager);
}
```

### Cycle Summary Format (Enhanced)

```
📋 Cycle 3 Summary | Self-play: P1=52 P2=48 D=0 | 
P1 med=66 (σ²=121, 49-83) | P2 med=97 (σ²=118, 72-102) | 
Train: 0.002290 → 0.000975 (234 replay + 783 new) | 
Eval: 60.0% (AI med=28, Rand med=30, gap=+2) | 
Diversity: Open=2.32 Board=1.25 | Buf: 783 | 15s
```

### Experiment 9 Config

```bash
node src/server/train.js \
  --cycles 20 \
  --games 100 \
  --eval-games 200 \
  --iterations 100 \
  --learning-rate 0.01 \
  --exploration 0.3 \
  --self-play-mode network \
  --start-from trainedAI_network.json \
  --output trainedAI_network_v9.json \
  --save-every 5 \
  --replay-ratio 0.3 \
  --replay-size 50000 \
  --log experiments/logs/exp9.log
```

### Success Criteria

| Metric | Target |
|--------|--------|
| Eval win rate | >55% |
| AI-Random score gap | Growing over training |
| Opening diversity | >1.5 bits |
| Initial error | Decreasing cycle-over-cycle |
| Score variance | >0 (diverse games) |
