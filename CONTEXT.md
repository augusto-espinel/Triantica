# Triantica Project Context

**Last Updated:** 2026-10-24 23:08 CET
**Status:** ✅ Dashboard Working — 9 experiments with full plots, TF.js baselines complete

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
- `brain.js` resolved via dynamic import (ESM-safe) for browser inference
- Training uses `@tensorflow/tfjs-node` with Metal GPU acceleration (Node v20 LTS)
- Three experiment modes working: AI vs Random, Random vs AI, AI vs AI
- `calculateImmediateRewardVector` uses `GridManager.clone()` instead of JSON.parse/stringify
- `runSelfPlayGame` method preserved for training

## Known Issues / Lessons

1. **brain.js ESM resolution:** `require()` doesn't work in ESM context. Use `await import('brain.js')` with `mod.default || mod` fallback. (Browser inference only — training uses TF.js.)
2. **async _getBrain():** Must `await` the result — missing `await` was the cause of "NeuralNetwork is not a constructor".
3. **TF.js Node version:** Requires Node v20 LTS. Node v25's V8 engine breaks `tfjs-node` native bindings. Use `fnm use v20.20.2`.
3. **Coordinate convention:** `placeTriangle(x, y)` → `grid[y][x]` (row-major).
4. **Cluster adjacency:** Same-cell triangles share a corner, not a diagonal group.
5. **AI performance:** Pre-trained network is barely above random. This is the baseline to improve.

## Next Steps

1. **Retrain the AI** — The current network hasn't learned meaningful strategy
2. **Create AgentPlayer** — For spawned agent opponents (me playing your AI)
3. **Worker pool** — Parallel games on M2 Ultra for faster training
4. **Training automation** — Script to run N self-play games + train cycles

---

## Dashboard Data Management

### Architecture Overview

The project uses a **dual-branch deployment**:

```
main       → Source code + experiments.json + data files
              ↓
gh-pages   → GitHub Pages deployment (auto-sync required)
```

**GitHub Pages URL:** `https://augusto-espinel.github.io/Triantica/docs/`

### How to Add New Experiment Data

When running a new training experiment:

1. **Run the experiment** (creates data files in `docs/data/`):
   ```bash
   cd triantica
   node src/server/train_tfjs.cjs --cycles 20 --games 100 --output docs/data/TF-B4_new.json
   ```

2. **Update experiments.json** with the new entry:
   ```bash
   # Edit docs/experiments.json to add:
   # - id: unique experiment ID
   # - name: descriptive name with key parameters
   # - logFile: path to data file (data/XXX.json)
   # - dataFile: filename only (XXX.json)
   # - summary: one-line description
   # - parameters: all training parameters
   # - result: final metrics
   ```

3. **Commit to main branch:**
   ```bash
   git add docs/experiments.json docs/data/TF-B4_new.json docs/data/TF-B4_new_network.json
   git commit -m "Add TF-B4 experiment results"
   git push origin main
   ```

4. **Sync to gh-pages branch:**
   ```bash
   git checkout gh-pages
   git checkout main -- docs/experiments.json docs/data/TF-B4_new.json docs/data/TF-B4_new_network.json
   git add docs/
   git commit -m "Add TF-B4 experiment data"
   git push origin gh-pages
   git checkout main
   ```

5. **Wait for deployment** (~2 minutes):
   ```bash
   gh run list --branch gh-pages --limit 1
   ```

6. **Verify with cache-busting:**
   ```bash
   curl -sH "Cache-Control: no-cache" \
     "https://augusto-espinel.github.io/Triantica/docs/data/TF-B4_new.json?nc=$(date +%s)" \
     | jq '.cycles | length'
   ```

### Critical Mistakes to Avoid

#### ❌ Mistake 1: File Truncation During Write
**What happened:** Large JSON files got cut off mid-write, leaving invalid JSON.

**Symptom:** `jq` parse errors, dashboard shows fewer experiments than expected.

**Prevention:**
- Use `jq '.' < file.json` to verify the file is valid JSON after writing
- Check file size is reasonable (not suspiciously small)
- Always verify with `jq 'length'` for arrays

#### ❌ Mistake 2: Forgetting gh-pages Sync
**What happened:** Updated `experiments.json` in `main`, but `gh-pages` still had old version.

**Symptom:** Dashboard shows 3 experiments instead of 9; only "baseline" experiments work.

**Prevention:**
- ALWAYS sync data files to `gh-pages` branch after updating `main`
- Verify raw GitHub URL: `https://raw.githubusercontent.com/augusto-espinel/Triantica/gh-pages/docs/experiments.json`
- GitHub Pages caches for ~2 minutes; use cache-busting query params to verify

#### ❌ Mistake 3: Missing Data Files in gh-pages
**What happened:** `experiments.json` pointed to `TF-B1_v3.json`, but that file didn't exist in `gh-pages`.

**Symptom:** Dashboard shows experiment card but plots don't render.

**Prevention:**
- Check `ls -la docs/data/TF-B*.json` in BOTH branches
- Use `git checkout main -- docs/data/FILENAME.json` to copy from main

### Dashboard Data Format

Experiment data files (`docs/data/XXX.json`) must have this structure:

```json
{
  "experimentId": "TF-B1_v3",
  "startedAt": "2026-10-24T19:30:00Z",
  "endedAt": "2026-10-24T19:50:00Z",
  "options": { /* training parameters */ },
  "cycles": [
    {
      "cycle": 1,
      "selfPlay": {
        "gamesPlayed": 100,
        "opponentTypeCounts": { "network": 70, "random": 30 },
        "winsByOpponent": { "network": {"1": 35}, "random": {"1": 15} },
        "scoresByOpponent": {
          "network": { "p1": {"median": 25, "max": 80}, "p2": {...} },
          "random": { ... }
        }
      },
      "evaluation": {
        "vsRandom": { "aiWinRate": 42, "aiScores": [...], "opponentScores": [...] },
        "vsHeuristic": { "aiWinRate": 0, "aiScores": [...], "opponentScores": [...] }
      },
      "training": {
        "initialError": 0.012,
        "finalError": 0.008,
        "iterations": 10
      }
    },
    // ... more cycles
  ],
  "finalResult": {
    "vsRandom": 37,
    "vsHeuristic": 0
  }
}
```

---

## Training Book Updates

### Purpose

`docs/TRAINING_BOOK.md` serves as the **comprehensive RL learning guide** for non-ML engineers. It must stay current with experimental findings.

### When to Update

Update the training book when:

1. **New experiment reveals a finding** — positive or negative results
2. **Parameter tuning shows clear patterns** — learning rate, batch size, iterations
3. **Bug fixes change interpretation** — role alternation, gradient isolation
4. **Architecture changes planned** — Value Head, MCTS, Conv layers
5. **New heuristics discovered** — opening moves, territory control

### How to Update

1. **Add experiment results** to the relevant section:
   ```markdown
   #### TF-B4: Mixed Opponents with Value Head (2026-10-25)
   
   | Metric | Cycle 1 | Cycle 10 | Cycle 20 |
   |--------|---------|----------|----------|
   | vs Random | 40% | 52% | 48% |
   | vs Heuristic | 0% | 0% | 0% |
   | Training Error | 0.012 | 0.005 | 0.008 |
   
   **Key Finding:** Value Head alone doesn't break the heuristic wall.
   ```

2. **Update the Key Takeaways** section with lessons learned

3. **Add to the Bug Reports** section if a bug was discovered:
   ```markdown
   ### Bug: Heuristic Player Number Hardcoded
   
   **Discovered:** 2026-10-24 in TF-B2
   **Symptom:** Both AI and Heuristic thought they were P2
   **Root Cause:** `heuristicEvalP2` hardcoded player number
   **Fix:** Pass player number to HeuristicPlayer constructor
   ```

4. **Update parameter analysis** if new insights emerged:
   ```markdown
   #### Learning Rate (LR)
   
   **Updated insight:** After TF-B4, we found that LR=0.01 with momentum=0.9
   outperforms LR=0.05 by 15% in final win rate.
   ```

5. **Commit with descriptive message:**
   ```bash
   git add docs/TRAINING_BOOK.md
   git commit -m "Update TRAINING_BOOK: TF-B4 results, Value Head analysis"
   git push origin main
   ```

### Quality Checklist

Before committing training book updates, verify:

- [ ] All experiment IDs match the actual data files
- [ ] Numerical results match the JSON data files
- [ ] Bug descriptions include: symptom, root cause, fix
- [ ] Parameter explanations are accessible to non-ML readers
- [ ] Diagrams render correctly (ASCII art in code blocks)
- [ ] Links to external resources are valid

---

## Quick Reference: Experiment Workflow

```bash
# 1. Run training experiment
cd triantica
node src/server/train_tfjs.cjs --cycles 20 --games 100 --output docs/data/TF-B4.json

# 2. Verify output is valid
jq '.cycles | length' docs/data/TF-B4.json  # Should be 20
jq '.cycles[0].evaluation.vsRandom.aiWinRate' docs/data/TF-B4.json

# 3. Update experiments.json
# (manual edit or script)

# 4. Verify experiments.json is valid
jq 'length' docs/experiments.json

# 5. Commit to main
git add docs/experiments.json docs/data/TF-B4.json docs/data/TF-B4_network.json
git commit -m "Add TF-B4: Mixed opponents with gradient isolation"
git push origin main

# 6. Sync to gh-pages
git checkout gh-pages
git checkout main -- docs/experiments.json docs/data/TF-B4*.json
git add docs/
git commit -m "Add TF-B4 experiment data"
git push origin gh-pages
git checkout main

# 7. Wait for deployment (~2 min)
gh run list --branch gh-pages --limit 1

# 8. Verify deployment
curl -sH "Cache-Control: no-cache" \
  "https://augusto-espinel.github.io/Triantica/docs/data/TF-B4.json?nc=$(date +%s)" \
  | jq '.cycles | length'

# 9. Update TRAINING_BOOK.md with findings
# (manual edit)

# 10. Commit training book update
git add docs/TRAINING_BOOK.md
git commit -m "Update TRAINING_BOOK: TF-B4 results and analysis"
git push origin main
```
