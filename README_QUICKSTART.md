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
  --output data/quick_test.json
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
  --output data/baseline.json
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
  --output data/mixed_baseline.json
```

### Continue from Checkpoint

```bash
fnm use v20.20.2
node src/server/train_tfjs.cjs \
  --cycles 10 \
  --start-from data/exp20_mixed.json \
  --output data/continued.json
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
| `--output` | data/training.json | Output path |

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
**https://augusto-espinel.github.io/Triantica/docs/**

### Local Analysis

```bash
# List training data
ls data/*.json

# Quick summary with jq
cat data/baseline.json | jq '.cycles[-1] | {cycle, winRate: .evaluation.aiWinRate, error: .training.finalError}'
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
# Copy data to dashboard
cp data/experiment.json docs/data/

# Push to gh-pages for dashboard update
git checkout gh-pages
git checkout main -- docs/data/experiment.json docs/index.html
git add docs/
git commit -m "Update dashboard data"
git push origin gh-pages
git checkout main
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
| HvAI | Human vs trained AI (default) |
| AvR | AI vs Random baseline |
| AvH | AI vs Heuristic (greedy enclosure) |
| AvA | AI vs AI (self-play) |

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
