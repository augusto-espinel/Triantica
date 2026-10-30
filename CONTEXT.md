# Triantica Project Context

**Last Updated:** 2026-10-28 18:35 CET
**Status:** ✅ `TF-B14` Completed — **New Record `90.0%` Peak (`88%` P1 / `92%` P2) & `80.1%` 10-Cycle Avg vs Heuristic (`100%` vs Random)**; `TF-B14_best_network.json` promoted to `trainedAI_network.json`
**Official Execution Plan:** [`docs/EXECUTION_ROADMAP.md`](./docs/EXECUTION_ROADMAP.md)

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
| `src/data/GameLogger.js` | Unified & `HumanGameLogger` telemetry logging | ✅ Validated |
| `src/server/runExperiment.js` | CLI runner | ✅ Created |
| `src/server/server.js` | Express API | ✅ Created |
| `tests/validate.js` | 20 automated tests (incl. `HumanGameLogger`) | ✅ All pass |
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

### Phase 4: AIPlayer Integration & Human-vs-AI Telemetry ✅
- Refactored `AIPlayer.js` to extend `PlayerInterface`
- Uses `GridManager` directly (no Phaser dependency)
- `brain.js` resolved via dynamic import (ESM-safe) for browser inference
- Training uses `@tensorflow/tfjs-node` with Metal GPU acceleration (Node v20 LTS)
- Three experiment modes working: AI vs Random, Random vs AI, AI vs AI
- `calculateImmediateRewardVector` uses `GridManager.clone()` instead of JSON.parse/stringify
- `runSelfPlayGame` method preserved for training
- **`HumanGameLogger` (`game.js` & `src/data/GameLogger.js`):** Captures pre-move `10×10×18` canonical one-hot state (`1,800` floats) and `401`-float legacy state for every human and AI ply in `HvAI` mode, persists up to 100 games in a `localStorage` FIFO ring buffer (`triantica_human_games`), and provides a one-click **📤 Export & Flush** button next to the scoreboard.

## Known Issues / Lessons

1. **brain.js ESM resolution:** `require()` doesn't work in ESM context. Use `await import('brain.js')` with `mod.default || mod` fallback. (Browser inference only — training uses TF.js.)
2. **async _getBrain():** Must `await` the result — missing `await` was the cause of "NeuralNetwork is not a constructor".
3. **TF.js Node version:** Requires Node v20 LTS. Node v25's V8 engine breaks `tfjs-node` native bindings. Use `fnm use v20.20.2`.
3. **Coordinate convention:** `placeTriangle(x, y)` → `grid[y][x]` (row-major).
4. **Cluster adjacency:** Same-cell triangles share a corner, not a diagonal group.
5. **AI performance:** Pre-trained network is barely above random. This is the baseline to improve.

## Next Steps (See [`docs/EXECUTION_ROADMAP.md`](./docs/EXECUTION_ROADMAP.md))

1. **Phase 1 (Active — `TF-B14`):** Fix Bottleneck 1 (`45,000` sliding-window replay buffer + Cosine LR `0.0004 → 0.0001` + automatic `*_best_network.json` checkpointing) and evaluate the ceiling of **100% pure network self-play** (`Dual-Side QuiescenceMCTS-50`) under the current 2D hybrid topology (`Bounded Placement / Toroidal Enclosure`).
2. **Phase 2 (Official 2D Release — `2d-variant` branch & `v1.0.0-2d` tag):** Finalize `TF-B14` metrics, promote the best checkpoint to `trainedAI_network.json`, commit, push, tag `v1.0.0-2d`, and publish the completed 2D game to its own dedicated **`2d-variant`** GitHub branch (so GitHub Pages can be pointed to `2d-variant` anytime for A/B comparison), while continuing 3D development under `main`.
3. **Phase 3 (3D Capability & Full Edge Symmetry):** Enable full toroidal corner placement (`hasCorner` wrapping), implement Bottleneck 2 (Toroidal Circular Padding in `TFConvNetwork`) and Bottleneck 3 (Inference-Time Tactical Safety Filter in `ConvNetwork.predictMove`), and generalize `GridManager` to arbitrary 3D triangulated manifolds $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$.
4. **Phase 4 (Manifold Symmetry Augmentation):** Tackle Bottleneck 4 ($D_4$ dihedral / toroidal / gauge symmetry exploitation) after all 3D-compatible training upgrades are verified.

---

## Dashboard Data Management

### Architecture Overview

The project uses a **dual-branch deployment**:

```
main       → Source code + dashboard/experiments.json + dashboard/data/ files
              ↓
GitHub Pages deploys directly from `main` branch root (`/`)
```

**GitHub Pages URL:** `https://augusto-espinel.github.io/Triantica/dashboard/`

### How to Add New Experiment Data

When running a new training experiment:

1. **Run the experiment** (creates data files in `dashboard/data/`):
   ```bash
   cd triantica
   node src/server/train_tfjs.cjs --cycles 20 --games 100 --output dashboard/data/TF-B4_new.json
   ```

2. **Update experiments.json** with the new entry:
   ```bash
   # Edit dashboard/experiments.json to add:
   # - id: unique experiment ID
   # - name: descriptive name with key parameters
   # - logFile: path to data file (data/XXX.json)
   # - dataFile: filename only (XXX.json)
   # - summary: one-line description
   # - parameters: all training parameters
   # - result: final metrics
   ```

3. **Verify and clean `dashboard/data/`:**
   ```bash
   ./scripts/publish-dashboard.sh
   ```

4. **Commit and push to `main` branch:**
   ```bash
   git add dashboard/experiments.json dashboard/data/TF-B4_new.json dashboard/data/TF-B4_new_network.json
   git commit -m "Add TF-B4 experiment results"
   git push origin main
   ```

5. **Wait for deployment** (~1-2 minutes):
   ```bash
   gh run list --branch main --limit 1
   ```

6. **Verify with cache-busting:**
   ```bash
   curl -sH "Cache-Control: no-cache" \
     "https://augusto-espinel.github.io/Triantica/dashboard/data/TF-B4_new.json?nc=$(date +%s)" \
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

#### ❌ Mistake 2: Missing Data Files in `dashboard/data/`
**What happened:** `dashboard/experiments.json` pointed to a `dataFile`, but that file wasn't committed in `dashboard/data/`.

**Symptom:** Dashboard shows experiment card but plots don't render.

**Prevention:**
- Run `./scripts/publish-dashboard.sh` to verify every `dataFile` in `dashboard/experiments.json` exists in `dashboard/data/`
- Verify raw GitHub URL: `https://raw.githubusercontent.com/augusto-espinel/Triantica/main/dashboard/experiments.json`
- GitHub Pages caches for ~2 minutes; use cache-busting query params to verify

### Dashboard Data Format

Experiment data files (`dashboard/data/XXX.json`) must have this structure:

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
node src/server/train_tfjs.cjs --cycles 20 --games 100 --output dashboard/data/TF-B4.json

# 2. Verify output is valid
jq '.cycles | length' dashboard/data/TF-B4.json  # Should be 20
jq '.cycles[0].evaluation.vsRandom.aiWinRate' dashboard/data/TF-B4.json

# 3. Update experiments.json
# (manual edit or script)

# 4. Verify experiments.json and dashboard/data/
jq 'length' dashboard/experiments.json
./scripts/publish-dashboard.sh

# 5. Commit and push to main (GitHub Pages deploys directly from main /)
git add dashboard/experiments.json dashboard/data/TF-B4.json dashboard/data/TF-B4_network.json
git commit -m "Add TF-B4: Mixed opponents with gradient isolation"
git push origin main

# 6. Wait for deployment (~1-2 min)
gh run list --branch main --limit 1

# 7. Verify deployment
curl -sH "Cache-Control: no-cache" \
  "https://augusto-espinel.github.io/Triantica/dashboard/data/TF-B4.json?nc=$(date +%s)" \
  | jq '.cycles | length'

# 8. Update TRAINING_BOOK.md with findings
# (manual edit)

# 10. Commit training book update
git add docs/TRAINING_BOOK.md
git commit -m "Update TRAINING_BOOK: TF-B4 results and analysis"
git push origin main
```
