> **[📚 Training Book Index](./README.md)** | [← Chapter 12: Toroidal 3D Visualization & Human Telemetry](./12-toroidal-3d-and-telemetry.md) | [Book Index →](./README.md)

---

## Appendix A: Command Reference

```bash
# Phase 1: Basic Feedforward Training (pure self-play)
node src/server/train_tfjs.cjs --cycles 20 --games 100 --architecture dense

# Phase 2–3: 2-Head Canonical ConvNet Actor-Critic (TF-B5 / TF-B6)
node src/server/train_tfjs.cjs --cycles 20 --games 100 \
  --architecture conv --learning-rate 0.001 --batch-size 256

# Phase 4: Consolidated AlphaZero MCTS-50 (TF-B9 / TF-B10)
node src/server/train_tfjs.cjs --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001

# Phase 5: Quiescence MCTS + Sliding-Window Pure Self-Play (TF-B14 — 2D Champion)
node src/server/train_tfjs.cjs --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.0004 --lr-decay cosine --min-lr 0.0001 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.25 --enclosure-bonus 0.15 \
  --replay-buffer-size 45000 --start-from dashboard/data/TF-B13_network.json \
  --output dashboard/data/TF-B14.json

# Phase 6: Full T² Toroidal Manifold + Circular Conv + Safety Filter (TF-B15 — 100% Champion)
node src/server/train_tfjs.cjs --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --toroidal-pad true --toroidal-placement true \
  --learning-rate 0.0003 --lr-decay cosine --min-lr 0.00008 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.25 --enclosure-bonus 0.15 \
  --safety-filter true --replay-buffer-size 45000 \
  --start-from dashboard/data/TF-B14_best_network.json \
  --output dashboard/data/TF-B15.json
```

### Evaluation Commands

```bash
# Evaluate champion network against HeuristicPlayer (100 random-2-ply openings, P1/P2 split)
node src/server/evaluate.cjs --network trainedAI_network.json \
  --opponent heuristic --games 100

# Evaluate against RandomPlayer
node src/server/evaluate.cjs --network trainedAI_network.json \
  --opponent random --games 100

# Full cross-checkpoint tournament
node src/server/evaluate.cjs --network dashboard/data/TF-B15_best_network.json \
  --tournament --games 50
```

---

## Appendix B: Project Structure

<details>
<summary>📐 Diagram: triantica/ (click to expand ASCII)</summary>

```
triantica/
├── src/
│   ├── core/
│   │   ├── GridManager.js         # T² toroidal game engine, enclosures & cluster BFS
│   │   ├── ToroidalViewport.js    # 2D toroidal panning & wrap-around coordinate projection
│   │   └── Torus3DView.js         # Synchronized 3D interactive parametric ring torus view
│   ├── players/
│   │   ├── AIPlayer.js            # Browser/server ConvNet + 3 difficulty tiers (Easy/Med/Hard)
│   │   ├── HeuristicPlayer.js     # Deterministic 2-ply tactical baseline opponent
│   │   └── RandomPlayer.js        # Uniform random baseline opponent
│   ├── data/
│   │   └── HumanGameLogger.js     # Telemetry logger for human-vs-AI matches (/api/human-games)
│   └── server/
│       ├── ConvNetwork.cjs        # 10×10×18 2-Head ConvNet + ToroidalPad2D + Safety Filter
│       ├── MCTS.cjs               # Batched PUCT Monte Carlo Tree Search
│       ├── QuiescenceMCTS.cjs     # Quiescence cascade rollouts + root prior risk shaping
│       ├── ReplayBuffer.cjs       # Sliding-window FIFO experience replay buffer
│       ├── train_tfjs.cjs         # Unified TF.js Metal/AMX self-play training pipeline
│       └── evaluate.cjs           # 2-ply random-opening evaluation harness (P1/P2 split)
├── dashboard/
│   ├── index.html                 # Interactive training telemetry dashboard UI
│   ├── experiments.json           # Experiment catalog & milestone summaries
│   └── data/                      # Published training JSON logs & model checkpoints
├── docs/
│   ├── TRAINING_BOOK.md           # Training Book entry-point index
│   ├── training-book/             # Modular Training Book chapters (01..12 + Appendices)
│   ├── AI_ARCHITECTURE.md         # Deep-dive tensor & network reference
│   ├── EXECUTION_ROADMAP.md       # 2D release & 3D manifold generalization roadmap
│   └── QUIESCENCE_MCTS.md         # Quiescence MCTS mathematical specification
└── trainedAI_network.json         # Production champion weights (TF-B15 Cycle 7: 99% Raw / 100% +Safety)
```

</details>

---

## Appendix C: CLI & Dashboard Telemetry Parameters

### 1. Self-Play Training Pipeline CLI Flags (`src/server/train_tfjs.cjs`)

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--cycles` | `10` | Number of self-play + gradient descent cycles |
| `--games` | `100` | Self-play games generated per cycle |
| `--eval-games` | `100` | Evaluation games per opponent (`RandomPlayer` & `HeuristicPlayer`, 50 P1 / 50 P2) |
| `--architecture` | `conv` | Network architecture (`conv` 18-channel Actor-Critic or `dense` 401-input MLP) |
| `--toroidal-pad` | `true` | Circular toroidal padding (`ToroidalPad2D`) across all `3×3` `Conv2D` layers |
| `--toroidal-placement` | `true` | Full $T^2$ modulo vertex corner legality in `GridManager.hasCorner` |
| `--mcts-sims` | `50` | MCTS simulations per move during self-play training (`mctsBatchSize = 8`) |
| `--quiescence-mode` | `greedy` | Quiescence rollout mode at MCTS leaves (`none`, `greedy`, `hybrid`) |
| `--quiescence-depth` | `5` | Maximum forced enclosure-cascade depth simulated at MCTS leaves |
| `--risk-penalty` | `0.25` | Leaf value & root prior penalty weight for moves leaving opponent enclosures |
| `--enclosure-bonus` | `0.15` | Leaf value & root prior bonus weight for moves claiming enclosures |
| `--safety-filter` | `true` | Enable inference-time top-$K$ ($K=12$) tactical safety filter during evaluation |
| `--replay-buffer-size` | `45000` | Sliding-window FIFO experience replay capacity (`~3.5` cycles of dual-side MCTS) |
| `--learning-rate` | `0.0003` | Initial Adam learning rate |
| `--lr-decay` | `cosine` | Learning rate schedule (`none` or `cosine` decay down to `--min-lr`) |

### 2. Dashboard Telemetry Schema (`dashboard/data/TF-*.json`)

The training dashboard (`dashboard/index.html`, hosted at `/dashboard/`) visualizes these experiment parameters and per-cycle telemetry fields:

| Dashboard Field | Description |
|-----------------|-------------|
| `cycles` / `games` / `evalGames` | Total training cycles, self-play games/cycle, and evaluation games/opponent |
| `learningRate` / `lrSchedule` | Initial learning rate ($\alpha$) and schedule (`constant` vs `cosine` decay) |
| `mctsSims` / `quiescenceMode` | MCTS simulations per move (`50`) and leaf cascade mode (`greedy`, depth `5`) |
| `toroidalPad` / `toroidalPlacement` | Whether `ToroidalPad2D` circular convs and full $T^2$ corner symmetry are active |
| `eval.vsRandom` / `eval.vsHeuristic` | Win rates (`0–100%`) across 100 random-2-ply openings (`P1` and `P2` role splits) |
| `eval.vsHeuristicRaw` | Unfiltered 1-ply Policy Head win rate vs `HeuristicPlayer` (`99.0%` peak in `TF-B15`) |
| `eval.vsHeuristicSafe` | 1-ply Policy Head + Top-12 Tactical Safety Filter win rate (`100.0%` across all 10 cycles) |
| `scoreStats` (`mean`, `median`, `max`) | Contiguous triangle cluster sizes for AI vs Opponent (`174.2 vs 14.7` in `TF-B15`) |
| `initialError` / `finalError` | Combined Policy CE + Value MSE loss at Epoch 1 vs Epoch 10 of each cycle |
| `openingDiversity` / `boardDiversity` | Shannon entropy (in bits) of opening moves (`~4.4b`) and final board states (`~1.90b`) |

---

*Last updated: 2026-10-31*
*Version: 4.0 (Authoritative Complete Edition — Phase 1 to Phase 6 Full Torus $T^2$ Mastery)*
*Framework: TensorFlow.js (`@tensorflow/tfjs-node`) on Apple Silicon (AMX / Accelerate)*
*Hardware: Mac Studio M2 Ultra (24-core CPU, 60-core GPU, 64GB Unified Memory)*

---

> [← Chapter 12: Toroidal 3D Visualization & Human Telemetry](./12-toroidal-3d-and-telemetry.md) | **[📚 Training Book Index](./README.md)** | [Book Index →](./README.md)
