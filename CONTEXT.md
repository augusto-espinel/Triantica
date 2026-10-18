# Triantica Project Context

**Last Updated:** 2026-10-18 12:30 CET
**Status:** ✅ AIPlayer Integrated — All tests pass, AI vs Random/ Self-play working

## Quick Start (for Augusto)

```bash
cd ~/.openclaw/workspace/triantica
npm test                  # Run 17 validation tests
npm run test:ai           # Run AI integration tests (AI vs Random, AI vs AI)
npm run experiment -- --type vs-random --count 100
npm run serve             # Start API server on :8080
```

See `README_QUICKSTART.md` for full instructions.

## Architecture

```
┌─────────────────────────────────────────────────┐
│            GAME LOGIC (src/core/)                │
│  GridManager.js - Pure JS, no dependencies       │
│  GameSession.js - Orchestrates games            │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│    BROWSER    │           │   NODE.JS     │
│ (Phaser UI)   │           │  (Server)    │
├───────────────┤           ├───────────────┤
│ game.js       │           │ server.js     │
│ (visual only) │           │ runExperiment │
│               │           │               │
│ BrowserPlayer │           │ RandomPlayer  │
│ (Human UI)    │           │ AIPlayer ✅   │
└───────────────┘           └───────────────┘
        │                           │
        └─────────────┬─────────────┘
                      ▼
              ┌───────────────┐
              │  GameLogger   │
              │  (Unified)    │
              └───────────────┘
```

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/core/GridManager.js` | Pure game logic | ✅ Validated |
| `src/core/GameSession.js` | Game orchestration | ✅ Validated |
| `src/players/RandomPlayer.js` | Baseline opponent | ✅ Validated |
| `src/players/AIPlayer.js` | Neural network AI | ✅ Integrated |
| `src/data/GameLogger.js` | Unified logging | ✅ Validated |
| `src/server/runExperiment.js` | CLI runner | ✅ Created |
| `src/server/server.js` | Express API | ✅ Created |
| `tests/validate.js` | 17 automated tests | ✅ All pass |
| `tests/test_ai_integration.js` | AI integration tests | ✅ Working |
| `AIPlayer.js` (original) | Legacy neural net | 📦 Preserved (not deleted) |
| `game.js` (original) | Phaser scenes | ✅ Still works |

## Baseline Results

| Experiment | P1 Wins | P2 Wins | Draws | AI Win Rate |
|-----------|---------|---------|-------|-------------|
| Random vs Random (100) | 44 | 50 | 6 | — |
| AI vs Random (50) | 28 | 22 | 0 | 56% |
| Random vs AI (50) | 24 | 23 | 3 | 46% |
| AI vs AI (10) | 6 | 4 | 0 | — |

**Finding:** The pre-trained AI network wins ~51% vs random — barely above coin-flip. The network needs retraining to develop meaningful strategy.

## What's Been Done

### Phase 1-3: Complete ✅
- Documentation, architecture refactoring, GridManager extraction
- All 17 validation tests pass
- Cluster adjacency bug fixed
- Coordinate convention documented

### Phase 4: AIPlayer Integration ✅
- Refactored `AIPlayer.js` to extend `PlayerInterface`
- Uses `GridManager` directly (no Phaser dependency)
- `brain.js` resolved via dynamic import (ESM-safe)
- Three experiment modes working: AI vs Random, Random vs AI, AI vs AI
- `calculateImmediateRewardVector` uses `GridManager.clone()` instead of JSON.parse/stringify
- `runSelfPlayGame` method preserved for training

## Known Issues / Lessons

1. **brain.js ESM resolution:** `require()` doesn't work in ESM context. Use `await import('brain.js')` with `mod.default || mod` fallback.
2. **async _getBrain():** Must `await` the result — missing `await` was the cause of "NeuralNetwork is not a constructor".
3. **Coordinate convention:** `placeTriangle(x, y)` → `grid[y][x]` (row-major).
4. **Cluster adjacency:** Same-cell triangles share a corner, not a diagonal group.
5. **AI performance:** Pre-trained network is barely above random. This is the baseline to improve.

## Next Steps

1. **Retrain the AI** — The current network hasn't learned meaningful strategy
2. **Create AgentPlayer** — For spawned agent opponents (me playing your AI)
3. **Worker pool** — Parallel games on M2 Ultra for faster training
4. **Training automation** — Script to run N self-play games + train cycles
