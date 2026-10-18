# Triantica

A strategy game of triangular territory control with AI training capabilities.

## Quick Start

```bash
cd triantica
npm install
node tests/validate.js              # Validate architecture (17 tests)
npm run experiment -- --type vs-random --count 100
```

See [README_QUICKSTART.md](README_QUICKSTART.md) for full instructions and [CONTEXT.md](CONTEXT.md) for session handoff context.

## Architecture Validation

All 17 automated tests pass ✅

| Component | Tests | Status |
|-----------|-------|--------|
| GridManager | 8 | ✅ Pass |
| RandomPlayer | 2 | ✅ Pass |
| GameSession | 4 | ✅ Pass |
| GameLogger | 2 | ✅ Pass |
| Baseline (20 games) | 1 | ✅ Pass |

Baseline data (100 Random vs Random games): P1=44 wins, P2=50 wins, 6 draws, avg 167.6 moves/game.
Saved at `experiments/data/baseline-random-100.json`.

## 📚 Documentation

| Document | Purpose |
|----------|--------|
| **[CONTEXT.md](CONTEXT.md)** | Session handoff + current state |
| **[README_QUICKSTART.md](README_QUICKSTART.md)** | How to run experiments |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Technical architecture |
| **[docs/TRAINING_BOOK.md](docs/TRAINING_BOOK.md)** | Experiment journal |
| **[docs/AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md)** | Neural network reference |
| **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** | Game rules cheat sheet |

---

### Game Overview

**Triantica** is a grid-based strategy game built with **Phaser 3**.

* **Goal:** Players place triangles on a 10×10 grid to form the largest contiguous cluster of their color.
* **Mechanic:** Each grid cell can hold up to two triangles. Connecting triangles merges "clusters." Enclosing an empty space claims it and grants an extra turn.
* **AI:** Neural network (Brain.js) trained via self-play RL.

---

### File Structure

| File | Primary Responsibility |
| --- | --- |
| **`game.js`** | Phaser Scenes + visual rendering + rules logic |
| **`AIPlayer.js`** | Neural network, move prediction, self-play training |
| **`src/core/GridManager.js`** | Extracted pure game logic (no Phaser deps) |
| **`src/core/GameSession.js`** | Game orchestration for experiments |
| **`src/players/RandomPlayer.js`** | Baseline random opponent |
| **`src/data/GameLogger.js`** | Unified experiment logging |
| **`src/server/runExperiment.js`** | CLI experiment runner |
| **`tests/validate.js`** | 17 automated validation tests |

---

### AI Architecture

* **Input (401):** Flattened 10×10 grid + current rotation
* **Hidden (200):** Fully connected
* **Output (400):** One score per (cell × rotation) move
* **Training:** Self-play RL with epsilon-greedy exploration
* **Reward:** Immediate cluster/enclosure gains + discounted final outcome

### Critical Functions

| Function | Location | What it does |
| --- | --- | --- |
| **`GridManager.updateClustersForNewTriangle`** | `GridManager.js` | Core mechanic: merges clusters on placement |
| **`AIPlayer.calculateImmediateRewardVector`** | `AIPlayer.js` | Simulates every legal move to find immediate reward |
| **`AIPlayer.trainNetwork`** | `AIPlayer.js` | Feeds self-play history into Brain.js |
