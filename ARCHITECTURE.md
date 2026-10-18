# Triantica Unified Architecture

**Goal**: Single codebase that runs in browser (GitHub Pages) and locally (Node.js) with full experiment capabilities.

---

## Design Principles

1. **Single Source of Truth** - Game logic lives in one place, used by all modes
2. **Player Interface** - Standardized interface for all player types
3. **Platform Abstraction** - Core logic works in browser and Node.js
4. **Data Portability** - Training data format is consistent across modes
5. **Local Acceleration** - Leverage M2 Ultra for parallel experiments

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAME LOGIC (Platform-Agnostic)               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     GridManager                           │  │
│  │  - Pure JS, no DOM/Phaser dependencies                   │  │
│  │  - Exported as ES module + global for browser            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│    BROWSER    │     │   NODE.JS     │     │   SPAWNED     │
│     MODE      │     │    SERVER     │     │    AGENT      │
├───────────────┤     ├───────────────┤     ├───────────────┤
│ GameScene     │     │ Experiment    │     │ AgentPlayer   │
│ (Phaser UI)   │     │ Runner        │     │ (OpenClaw)    │
│               │     │               │     │               │
│ BrowserPlayer │     │ Worker Pool   │     │ LLM Strategy  │
│ (Human UI)    │     │ (Parallel)    │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PLAYER INTERFACE                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  getMove(gridManager, playerNumber) → {x, y, rotation}    │  │
│  │  isReady() → boolean                                       │  │
│  │  recordResult(won) → void                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GameLog {                                                │  │
│  │    id, timestamp, players, winner, moves[],               │  │
│  │    trainingSamples: [{state, move, reward}]               │  │
│  │  }                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Browser: localStorage + JSON download                          │
│  Server: JSON files + SQLite aggregation                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
triantica/
├── src/
│   ├── core/
│   │   ├── GridManager.js      # Pure game logic (extracted)
│   │   ├── GameSession.js      # Orchestrates game between 2 players
│   │   └── constants.js        # Board size, rotations, etc.
│   │
│   ├── players/
│   │   ├── AIPlayer.js         # Neural network (refactored)
│   │   ├── RandomPlayer.js     # Random valid moves
│   │   ├── BrowserPlayer.js    # Human via Phaser UI
│   │   └── AgentPlayer.js      # Spawned OpenClaw agent
│   │
│   ├── data/
│   │   ├── GameLogger.js       # Unified logging
│   │   ├── TrainingData.js     # Training sample extraction
│   │   └── Export.js           # JSON/CSV export utilities
│   │
│   └── server/
│       ├── server.js           # Express server
│       ├── experimentRunner.js # Batch experiments
│       ├── workerPool.js       # Parallel game execution
│       └── api.js              # REST endpoints
│
├── browser/
│   ├── game.js                 # Phaser scenes (visual only)
│   ├── index.html              # Entry point
│   └── browserEntry.js         # Connects UI to core
│
├── experiments/
│   └── data/                   # Generated logs
│
├── package.json                # Node.js deps + scripts
├── vite.config.js              # Browser bundler
└── ARCHITECTURE.md             # This file
```

---

## Player Interface

```javascript
// All players implement this interface
class PlayerInterface {
  constructor(playerNumber, config = {}) {}
  
  // Called when it's this player's turn
  // Returns a Promise resolving to {x, y, rotation}
  async getMove(gridManager) {
    throw new Error('Must implement getMove()');
  }
  
  // Check if player is ready to play
  isReady() {
    return true;
  }
  
  // Called after game ends
  // winner: 0 (draw), 1, or 2
  recordResult(winner) {}
  
  // Optional: called before game starts
  async initialize() {}
  
  // Optional: for trainable players
  getTrainingData() {
    return [];
  }
}
```

---

## Experiment Types

| Type | Player 1 | Player 2 | Purpose |
|------|----------|----------|---------|
| **self-play** | Trained AI | Trained AI (clone) | Generate training data |
| **vs-random** | Trained AI | RandomPlayer | Quality benchmark |
| **vs-agent** | Trained AI | AgentPlayer (me) | Qualitative test |
| **human** | BrowserPlayer | Trained AI | Human gameplay |

---

## Local Acceleration Strategy (M2 Ultra)

### Option 1: Node.js Worker Threads
```javascript
// Run N games in parallel
const pool = new WorkerPool(8); // 8 parallel games
const results = await pool.runGames(100, config);
```

### Option 2: Brain.js Training Options
```javascript
// Use all CPU cores
network.train(data, {
  iterations: 1000,
  learningRate: 0.1,
  // Brain.js handles parallelization internally
});
```

### Option 3: Future GPU Acceleration
- Migrate to TensorFlow.js for Metal GPU support
- Requires model architecture change
- Phase 2 consideration

---

## Data Format

### Game Log
```json
{
  "id": "game-20261018-120000-001",
  "timestamp": "2026-10-18T12:00:00Z",
  "experiment": "EXP-001",
  "type": "self-play",
  "players": {
    "1": { "type": "ai", "difficulty": "medium" },
    "2": { "type": "ai", "difficulty": "medium" }
  },
  "winner": 1,
  "moves": 156,
  "duration_ms": 5420,
  "finalScores": { "1": 67, "2": 54 },
  "trainingSamples": [
    {
      "moveIndex": 0,
      "player": 1,
      "state": [0, -1, 0, -1, ...],
      "move": { "x": 5, "y": 5, "rotation": 0 },
      "immediateReward": 0,
      "finalReward": 0.64
    }
  ]
}
```

### Training Export
```json
{
  "samples": 15600,
  "games": 100,
  "format": "brainjs",
  "data": [
    { "input": [...], "output": [...] }
  ]
}
```

---

## API Design (Local Server)

```
POST /api/experiment
  body: { type: "self-play", count: 100 }
  response: { experimentId, estimatedTime }

GET /api/experiment/:id/status
  response: { completed, total, eta_seconds }

GET /api/experiment/:id/results
  response: { metrics, downloadUrl }

POST /api/train
  body: { experimentId, iterations: 500 }
  response: { trainingId, status }

GET /api/export/:experimentId/training-data
  response: JSON download
```

---

## Migration Path

### Phase 1: Extract Core Logic ✅ (Next)
1. Move GridManager to `src/core/GridManager.js`
2. Remove Phaser dependencies from GridManager
3. Create ES module exports + global for browser

### Phase 2: Create Players
1. Refactor AIPlayer to use interface
2. Create RandomPlayer
3. Create BrowserPlayer wrapper

### Phase 3: Add Server
1. Simple Express server
2. Worker pool for parallel games
3. File-based logging

### Phase 4: Agent Integration
1. AgentPlayer class
2. OpenClaw session spawning
3. LLM-based move decisions

### Phase 5: Training Pipeline
1. Automatic training data generation
2. Batch training from accumulated data
3. Model versioning

---

## Backward Compatibility

- `game.js` becomes thin Phaser wrapper
- Browser mode works exactly as before
- All existing functionality preserved
- Visual rendering unchanged
