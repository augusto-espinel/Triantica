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

## Appendix D: AI-Assisted Engineering Telemetry — Token Economics, Model Selection (`GLM-5` vs. `Argon`) & Context Architecture {#appendix-d-token-economics}

Alongside the reinforcement learning telemetry captured by the training pipeline (`src/server/train_tfjs.cjs`), the engineering of Triantica itself was conducted through an agentic coding runtime (OpenClaw on Apple Mac Studio M2 Ultra) backed by OpenRouter foundation models (`z-ai/glm-5` and `google/gemini-4-argon`) and a local Ollama compaction model (`ollama/gemma4:e4b`).

By correlating the **17,928 deduplicated LLM API calls (`1.18 billion` billed prompt + completion tokens)** recorded in the local agent session transcripts (`~/.openclaw/agents/main/sessions/*.jsonl`) against the **190 Git commits** and **6 architectural phases (`exp12` → `TF-B15`)** from October 18 to November 1, 2026, we can evaluate the token economics of building a zero-to-mastery AlphaZero system.

### 1. Model Efficiency Comparison: `z-ai/glm-5` vs. `google/gemini-4-argon`

| Metric | `z-ai/glm-5` (Oct 18–25 + Sub-Agent Runs) | `google/gemini-4-argon` (Oct 26 – Nov 1) | `ollama/gemma4:e4b` (Local M2 Ultra) |
| :--- | :--- | :--- | :--- |
| **Total API Calls** | **13,312 calls** (`11,879` tool calls / `1,417` stops) | **4,540 calls** (`3,973` tool calls / `567` stops) | **79 calls** (local compaction/utility) |
| **Total Prompt Volume (In + Cache)** | **859.33M tokens** | **315.79M tokens** | **5.10M tokens** |
| **Uncached Input Tokens** | **44.62M tokens** (`5.2%` of prompt) | **73.68M tokens** (`23.3%` of prompt) | **5.10M tokens** |
| **Cached Input Tokens (`cacheRead`)** | **805.48M tokens** (**`94.8%` cache hit rate**) | **241.51M tokens** (**`76.7%` cache hit rate**) | `0` (local unified memory) |
| **Output Tokens** | **3.12M tokens** | **1.79M tokens** | **0.08M tokens** |
| **Total Logged Cost** | **$129.65** | **$189.50** | **$0.00** (On-device Metal GPU) |
| **Average Cost per Turn / Tool Call** | **$0.0097 / call** (`< 1¢`) | **$0.0417 / call** (`~4.3×` higher per call) | **$0.0000 / call** |
| **Effective Blended Rate** | **$0.150 / 1M tokens** | **$0.597 / 1M tokens** (`~4.0×` higher per token) | **$0.000 / 1M tokens** |
| **Mean Tool Calls per User Turn** | **8.38 tool calls / stop** | **7.01 tool calls / stop** | **0.72 tool calls / stop** |

#### Architectural & Economic Findings (`GLM-5` vs. `Argon`)
1. **Unit-Token Economics Favor `GLM-5` by 4× (`$0.150/1M` vs. `$0.597/1M`):**
   - `z-ai/glm-5` achieved an extraordinary **94.8% prompt cache hit rate** (`805.5M` cached vs. `44.6M` uncached input tokens), keeping the average cost of an agentic tool-call step under **one cent (`$0.0097`)**.
   - During Phases 1–4 (`brain.js` exploration `exp12`–`exp22`, the `@tensorflow/tfjs-node` migration `TF-B1`–`TF-B7`, and the interactive telemetry dashboard), `glm-5` delivered 135 commits across 8 days (`Oct 18–25`) for **$110.20 total** (`~$13.77/day`).
2. **Reasoning Depth & Algorithmic Convergence Favor `Argon` on High-Entropy Math/Search Bugs:**
   - Between Oct 23 and Oct 25 (`TF-B5` through `TF-B8`), `glm-5` executed **7,045 API calls (`440M` tokens, `$64.69`)** while struggling to isolate coupled mathematical bugs across MCTS tree backup sign conventions (`Bug 5` & `Bug 6`), `1/400` policy cross-entropy dilution (`Bug 11`), and non-quiet leaf horizon blindness.
   - Switching the primary architect model to `google/gemini-4-argon` on **Oct 26** immediately resolved the MCTS perspective/loss coupling (`TF-B9` → `84%` vs. Random), designed **Quiescence MCTS** (`TF-B12`–`TF-B14` → breaking the `0%` Heuristic Wall up to `90%`), formulated **Full $T^2$ Toroidal Circular Padding (`ToroidalPad2D`) + Top-12 Safety Filtering** (`TF-B15` → **`100.0%` mastery across `1,000` evaluation games**), and built the synchronized 2.5D/3D Torus UI (`Torus3DView.js`).
3. **Where Over-Spend Occurred (`Argon` on High-Frequency Mechanical Edits):**
   - While `argon` was dramatically more cost-effective per *algorithmic breakthrough*, using it as the default model for high-iteration UI/CSS tweaking, icon geometry adjustments, and multi-file documentation sweeps (e.g., **Oct 30**: `1,261` `argon` calls costing **$48.11** in a single day) incurred a `4.3×` per-call premium where `glm-5` would have cost `~$11`.

---

### 2. Daily Token Spend vs. Git Commits & Engineering Milestones (`Oct 18 – Nov 1`)

| Date | Commits | `GLM-5` Calls / Cost | `Argon` Calls / Cost | Uncached In | Cached In (`Hit %`) | Compactions | Daily Total | Key Triantica Milestone Achieved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Oct 18** | 9 | `447c` / `$5.62` | — | `3.30M` | `28.38M` (`89.6%`) | 3 | **$5.62** | Initial `brain.js` reward engineering & trajectory analyzer |
| **Oct 19** | 4 | `616c` / `$7.01` | — | `3.67M` | `38.00M` (`91.2%`) | 3 | **$7.01** | Experience replay prototype & state mismatch bugfix |
| **Oct 20** | 12 | `760c` / `$8.90` | — | `4.85M` | `47.09M` (`90.7%`) | 9 | **$8.90** | `HeuristicPlayer`, $TD(\lambda)$ cluster shaping, `exp10`–`exp13a` |
| **Oct 21** | 12 | `1,412c` / `$13.80` | — | `6.86M` | `83.62M` (`92.4%`) | 4 | **$13.80** | Opponent-perspective learning (`exp12s`) & Phaser UI debug |
| **Oct 22** | 14 | `949c` / `$10.18` | — | `4.62M` | `60.43M` (`92.9%`) | 6 | **$10.18** | Reward normalization fix (`exp16`/`17b`) & GitHub Pages Dashboard |
| **Oct 23** | 36 | `2,440c` / `$23.82` | — | `8.17M` | `150.79M` (`94.9%`) | 13 | **$23.82** | `exp20` (`91%` vs Random) + `@tensorflow/tfjs-node` migration (`TF-B1`–`B3`) |
| **Oct 24** | 27 | `2,397c` / `$21.31` | — | `5.52M` | `144.78M` (`96.3%`) | 22 | **$21.31** | Phase 1–4 `ReplayBuffer.cjs` + `10×10×18` `ConvNetwork.cjs` (`TF-B4`–`B7`) |
| **Oct 25** | 21 | `2,208c` / `$19.56` | — | `5.94M` | `128.97M` (`95.6%`) | 11 | **$19.56** | Phase 5 `MCTS.cjs` implementation & `TF-B8` sign-flip debugging |
| **Oct 26** | 12 | `445c` / `$3.78` | `690c` / `$35.39` | `14.95M` | `64.51M` (`81.2%`) | 15 | **$39.17** | **Switch to `Argon`**: `Bug 6–11` fixes (`400×` CE sum) & `TF-B9` (`84%` peak) |
| **Oct 27** | 4 | `840c` / `$7.77` | `723c` / `$26.19` | `10.89M` | `96.28M` (`89.8%`) | 11 | **$33.97** | **Heuristic Wall Broken!** `QuiescenceMCTS.cjs` (`TF-B10` → `TF-B12` `63%` vs Heur) |
| **Oct 28** | 9 | `32c` / `$0.25` | `825c` / `$33.54` | `13.89M` | `46.20M` (`76.9%`) | 13 | **$33.79** | `TF-B13` (`89%`), `TF-B14` (`90%` Planar Champion, `v1.0.0-2d`), `ToroidalPad2D` |
| **Oct 29** | 18 | `19c` / `$0.12` | `405c` / `$20.14` | `8.62M` | `21.52M` (`71.4%`) | 12 | **$20.26** | **Neon Manifold UI**, 3D D20 intro & interactive 4-step Tutorial |
| **Oct 30** | 14 | `521c` / `$5.18` | `1,261c` / `$48.11` | `21.13M` | `97.08M` (`82.1%`) | 23 | **$53.29** | **`TF-B15` (`100%` Champion)**, PWA + `HumanGameLogger.js`, mobile UI polish |
| **Oct 31** | 13 | `226c` / `$2.34` | `575c` / `$23.92` | `10.04M` | `47.95M` (`82.7%`) | 22 | **$26.26** | Training Book v4.1 modularization (`18` chapters), 7 SVGs & Zero-Build SPA |
| **Nov 01** | 2 | — | `63c` / `$2.33` | `1.02M` | `1.28M` (`55.6%`) | 0 | **$2.33** | Late-game human playtesting analysis & token economics telemetry |
| **Total** | **207** | **13,312c / $129.65** | **4,540c / $189.50** | **118.30M** | **1,046.99M (`89.8%`)** | **167** | **$319.15** | *(Note: OpenRouter account total will be cross-reconciled when CSV is added)* |

---

### 3. OpenClaw Context Overhead & Why the Artificial ~100K Window Was Counterproductive

#### A. Dissecting the ~26.9K Token Startup Baseline
Across 43 fresh sessions in the telemetry logs, **Turn 1 begins with an average of `26,907` uncached input tokens** before a single project file is read. That baseline consists of:
1. **Tool Schema & Runtime System Prompt (`~8,500` tokens):** JSON schemas for `20+` tools (`exec`, `read`, `edit`, `write`, `sessions_spawn`, `cron`, `memory_*`, etc.) plus behavioral directives.
2. **Workspace Identity & Long-Term Memory (`~3,500–8,000` tokens):** `AGENTS.md`, `SOUL.md`, `USER.md`, `IDENTITY.md`, `TOOLS.md`, `HEARTBEAT.md`, and `MEMORY.md` (which had grown to `~19 KB` prior to the Oct 31 archival compaction).
3. **Rolling Daily Memory Injection (`startupContext.dailyMemoryDays: 3`, `~10,000–14,000` tokens):** Automatically injecting the last 3 days of raw `memory/YYYY-MM-DD.md` logs into the prompt prefix on every session start.

Because agentic coding averages **7.5 tool calls per user message**, any bloat in the static startup prefix is multiplied across thousands of turns.

#### B. The "Compaction Thrashing" Penalty of an Artificial ~108K Context Ceiling
Prior to expanding `agents.defaults.contextTokens` to `400,000` on October 31, the runtime operated under an effective **`~108,000`-token ceiling** (visible in the daily max prompt column: `108,352` on Oct 18, `108,583` on Oct 22–23, `107,008` on Oct 30). That smaller artificial window proved **actively counterproductive** for both cost and engineering velocity:

1. **167 Compaction Checkpoints in 14 Days (`12–23` per day during peak coding):**
   - Reading monolithic files (such as the 3,546-line `TRAINING_BOOK.md` at `~45K` tokens or `game.js` + `train_tfjs.cjs`) pushed sessions from the `27K` baseline past the `108K` threshold in just **15–25 tool calls**.
2. **KV-Cache Destruction (`cacheRead` Drop from `95%` to `71–81%`):**
   - Provider prompt caching relies on an **immutable token prefix**. Every time compaction summarized the middle of a transcript or a session reset occurred, the token prefix changed—invalidating the KV cache and forcing the next turn to bill `50K–80K` tokens at **100% uncached input rates**.
3. **Rediscovery Amnesia Loop:**
   - After each compaction, the agent lost exact line numbers, tensor shapes, or CSS selectors from earlier in the hour, forcing it to re-run `read` and `grep` commands—re-filling the context window with the exact same file contents and triggering yet another compaction cycle.

---

### 4. Architectural Playbook: Making Future RL & Software Projects 65–75% Less Token-Expensive

1. **Modularize Large Documentation & Source Files Early (`< 500` lines per file):**
   - Splitting `TRAINING_BOOK.md` (`3,546` lines) into 18 modular chapter files under `docs/training-book/` reduced the per-edit file read footprint from `~45,000` tokens to `~2,500` tokens—a **94% reduction in context growth per edit**.
2. **Tiered Model Routing (Architect vs. Specialist Sub-Agents):**
   - Reserve `google/gemini-4-argon` (`$0.60/1M` blended) strictly for **high-entropy architectural design, mathematical derivations, and subtle bug audits** (where its 1-shot accuracy prevents multi-hour debugging loops).
   - Delegate **low-entropy mechanical execution** (CSS layout adjustments, wrapping 59 ASCII diagrams in `<details>`, syncing JSON files, writing unit tests) to `z-ai/glm-5` sub-agents (`$0.15/1M` blended, `< 1¢/call`).
3. **Keep Startup Context Lean (`< 10K` tokens) & Use On-Demand Recall:**
   - Keep `MEMORY.md` curated under `5 KB` (archiving historical tables to `memory/archive-*.md`) and set `startupContext.dailyMemoryDays` to `1` (relying on `memory_search` + `memory_get` only when prior session details are needed).
4. **Wide Context Ceiling (`400K`) + Surgical Tool Reads (`offset`/`limit` & Python Inspection):**
   - Pairing a wide `400K` context ceiling (to eliminate mid-task compaction thrashing and preserve `90%+` KV-cache hit rates) with surgical line-bounded reads (`read` with `offset`/`limit` or compact `python3`/`grep` summaries) prevents raw log dumps from polluting the cached prefix.

---

*Last updated: 2026-11-01*
*Version: 4.2 (Authoritative Complete Edition — Phase 1 to Phase 6 Full Torus $T^2$ Mastery + Engineering Token Telemetry)*
*Framework: TensorFlow.js (`@tensorflow/tfjs-node`) on Apple Silicon (AMX / Accelerate)*
*Hardware: Mac Studio M2 Ultra (24-core CPU, 60-core GPU, 64GB Unified Memory)*

---

> [← Chapter 12: Toroidal 3D Visualization & Human Telemetry](./12-toroidal-3d-and-telemetry.md) | **[📚 Training Book Index](./README.md)** | [Book Index →](./README.md)
