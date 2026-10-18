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

### 2026-10-18: Initial Analysis

1. **Reward shaping complexity**: The immediate reward calculation simulates every legal move, creating training labels that reflect immediate tactical value. This is computationally expensive but provides dense learning signal.

2. **Two-level exploration**: The AI has both random exploration (10%) and "immediacy-based" exploitation (20% of 90%), choosing between learned policy and greedy immediate reward maximization.

3. **Linear discounting**: Early moves get small fractions of the final reward. Alternative: exponential discounting (γ^t) or no discounting.

4. **Network size grows with difficulty**: Hard mode has 2 hidden layers, suggesting the problem benefits from deeper representations.

---

## Future Experiments

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

**Next Session:** Add metrics instrumentation and establish baseline performance numbers.
