# Triantica Quick Start

## Prerequisites

- **Node.js v20 LTS** (required for TF.js Metal GPU bindings — Node v25 breaks native bindings)
- [fnm](https://github.com/Schniz/fnm) (recommended for Node version management)
- npm

## Installation

```bash
cd ~/.openclaw/workspace/triantica
fnm use v20.20.2       # Switch to Node v20 LTS
npm install
npm rebuild @tensorflow/tfjs-node  # Rebuild Metal GPU bindings if needed
```

## Training the AI

### Quick Test (2 minutes)

```bash
fnm use v20.20.2
node src/server/train_tfjs.cjs \
  --cycles 3 \
  --games 50 \
  --eval-games 50 \
  --output dashboard/data/quick_test.json
```

### Full Baseline Experiment (~35 minutes)

```bash
fnm use v20.20.2
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 30 \
  --learning-rate 0.1 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix network \
  --audit true \
  --output dashboard/data/baseline.json
```

### Mixed Opponent Training (recommended — best results)

```bash
fnm use v20.20.2
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 30 \
  --learning-rate 0.1 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix mixed \
  --network-ratio 0.7 \
  --random-ratio 0.3 \
  --audit true \
  --output dashboard/data/mixed_baseline.json
```

### Continue from Checkpoint

```bash
fnm use v20.20.2
node src/server/train_tfjs.cjs \
  --cycles 10 \
  --start-from dashboard/data/TF-B10_network.json \
  --output dashboard/data/continued.json
```

### All CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cycles` | 10 | Number of train-eval cycles |
| `--games` | 100 | Self-play games per cycle |
| `--eval-games` | 100 | Evaluation games per cycle |
| `--iterations` | 30 | Training iterations per cycle |
| `--learning-rate` | 0.1 | SGD learning rate |
| `--exploration` | 0.3 | Epsilon-greedy exploration rate |
| `--self-play-mode` | network | `network` or `random` |
| `--opponent-mix` | network | `network`, `random`, or `mixed` |
| `--network-ratio` | 1.0 | Fraction of network opponents (mixed mode) |
| `--random-ratio` | 0.0 | Fraction of random opponents (mixed mode) |
| `--enable-cluster-reward` | false | Enable TD(λ) cluster reward |
| `--audit` | false | Enable game auditing |
| `--audit-rate` | 0.05 | Fraction of games to audit |
| `--start-from` | null | Path to checkpoint JSON to resume |
| `--output` | dashboard/data/training.json | Output path |

## Why TensorFlow.js on Mac Studio M2 Ultra

The training pipeline uses **TensorFlow.js with Metal GPU acceleration**, which is specifically optimized for Apple Silicon:

1. **Metal GPU acceleration**: TF.js uses Apple's Metal framework via native bindings in `@tensorflow/tfjs-node`. Tensor operations run directly on the GPU, yielding ~100x speedup over CPU-only brain.js.

2. **Unified Memory Architecture**: The M2 Ultra's unified memory (up to 192GB) means GPU and CPU share the same memory space. No CPU↔GPU data copies over a PCIe bus — training data is accessible by the GPU at full bandwidth.

3. **Performance comparison** (M2 Ultra, 24-core GPU):
   - Brain.js (CPU): ~8 minutes per 20-cycle experiment
   - TF.js (Metal GPU): ~35 seconds per 20-cycle experiment
   - **Effective speedup: ~100x per training step**

4. **Node v20 LTS requirement**: Node v25's V8 engine breaks the `tfjs-node` native bindings. Always use `fnm use v20.20.2` before training.

5. **Browser compatibility**: Trained TF.js weights can be exported to brain.js-compatible JSON for the Phaser browser game, which uses brain.js for inference (lightweight, no GPU needed).

## Viewing Results

### Dashboard

Training results are visualized on the dashboard:
**https://augusto-espinel.github.io/Triantica/dashboard/**

### Local Analysis

```bash
# List training data
ls dashboard/data/*.json

# Quick summary with jq
cat dashboard/data/TF-B12.json | jq '.cycles[-1] | {cycle, winRate: .evaluation.aiWinRate, error: .training.finalError}'
```

### Output Format

Each training run produces JSON with full per-cycle metrics:

```json
{
  "options": { /* CLI parameters */ },
  "cycles": [
    {
      "cycle": 1,
      "selfPlay": {
        "games": 100,
        "wins": {"1": 48, "2": 48, "draws": 4},
        "openingDiversity": 6.15,
        "boardDiversity": 1.95
      },
      "training": {
        "initialError": 0.014,
        "finalError": 0.003,
        "errorSnapshots": [0.014, 0.008, 0.005, 0.003],
        "iterations": 30,
        "samples": 17000
      },
      "evaluation": {
        "opponent": "random",
        "aiWinRate": 50,
        "games": 100
      }
    }
  ]
}
```

## Deploying to Dashboard

```bash
# Verify and clean dashboard data
./scripts/publish-dashboard.sh

# Push to main (GitHub Pages serves directly from main /)
git add dashboard/
git commit -m "Update dashboard data"
git push origin main
```

## Playing the Game

### Browser (GitHub Pages)

Open **https://augusto-espinel.github.io/Triantica/**

### Local Development

```bash
# Start dev server with hot reload
npm run dev
# Opens at http://localhost:5173
```

### Game Controls

| Key | Action |
|-----|--------|
| Click | Place triangle |
| R / Space | Rotate preview |
| Mouse wheel | Rotate preview |

### AI Opponent Modes

| Mode | Description |
|------|-------------|
| HvAI | Human vs trained AI (default, includes persistent FIFO move logging & JSON export) |
| AvR | AI vs Random baseline |
| AvH | AI vs Heuristic (greedy enclosure) |
| AvA | AI vs AI (self-play) |

### Exporting Human vs. AI Games for Training

During **Human vs. AI (`HvAI`)** matches, every human and AI ply is automatically logged to `localStorage` (`triantica_human_games`) in a rotating 100-game FIFO buffer:

1. **In-Game Indicator:** The **📤** button to the right of the scoreboard displays the number of completed games currently stored in the buffer.
2. **Download & Flush:** Click **📤** to download `triantica_human_games_YYYY-MM-DD.json` and automatically clear the local buffer.
3. **Training-Ready Schema (`schemaVersion: "2.0-canonical"`):**
   - `move.state`: Pre-move **10×10×18 canonical one-hot state** (`1,800` floats, player-relative) directly compatible with `TFConvNetwork` in `src/server/train_tfjs.cjs`.
   - `move.legacyState`: Pre-move **401-float vector** (`r / 4.0` rotation normalization + player flag) for legacy feedforward networks.
   - `move.actionIndex`: Flat policy index `(y * 10 + x) * 4 + rotation` (`0..399`).
   - `move.valueTarget`: Terminal outcome $+1$ (win), $0$ (draw), or $-1$ (loss) from the moving player's perspective (respecting forfeits).

## Documentation

| Document | Purpose |
|----------|---------|
| [TRAINING_BOOK.md](docs/TRAINING_BOOK.md) | Main experiment journal |
| [AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) | Neural network technical reference |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Game rules cheat sheet |
| [METRICS_INTEGRATION.md](docs/METRICS_INTEGRATION.md) | Metrics tracking guide |

## Troubleshooting

**`tfjs-node` build errors:**
```bash
fnm use v20.20.2
npm rebuild @tensorflow/tfjs-node
```

**Port 5173 already in use:**
```bash
PORT=3000 npm run dev
```

**Module not found errors:**
```bash
npm install
```

**GPU not detected:**
Check that Metal backend is active:
```bash
node -e "const tf = require('@tensorflow/tfjs-node'); console.log(tf.getBackend())"
# Should print: "tensorflow" (native Metal)
```
