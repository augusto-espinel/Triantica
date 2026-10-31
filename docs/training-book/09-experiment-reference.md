> **[📚 Training Book Index](./README.md)** | [← Chapter 8: The Heuristic Wall](./08-the-heuristic-wall.md) | [Chapter 10: Key Takeaways →](./10-key-takeaways.md)

---

## 9. Experiment Reference {#9-experiment-reference}

### Complete Results Table

| ID | Name | Framework | Mix | LR | Iter | Batch | Grad Iso | Cluster | vs Random | vs Heuristic | Time |
|----|------|-----------|-----|-----|------|-------|-----------|---------|-----------|--------------|------|
| TF-B1_pure | Pure Self-Play | TF.js | 100% net | 0.05 | 10 | - | ✗ | ✗ | **49%** | 0% | 39m |
| TF-B1_v3 | Pure (v3 fixes) | TF.js | 100% net | 0.05 | 10 | 256 | ✓ | ✗ | 37% | 0% | 40m |
| TF-B1_opt | Optimized | TF.js | 100% net | 0.01 | 100 | 32 | ✓ | ✗ | **47%** | 0% | 35m |
| TF-B2_mixed | Mixed Opponents | TF.js | 50/30/20 | 0.05 | 10 | - | ✗ | ✗ | 44% | 0% | 81m |
| TF-B2_v3 | Mixed (v3 fixes) | TF.js | 50/30/20 | 0.05 | 10 | 256 | ✓ | ✗ | 43% | 0% | 81m |
| TF-B3_cluster | Cluster Reward | TF.js | 70/30 | 0.05 | 10 | - | ✗ | ✓ | **52%** | 0% | 40m |
| TF-B3_v3 | Cluster (v3 fixes) | TF.js | 70/30 | 0.05 | 10 | 256 | ✓ | ✓ | 36% | 0% | 40m |
| TF-ONLINE | Online SGD | TF.js | alt | 0.01 | 10 | - | ✓ | ✗ | 27% | 0% | - |
| exp20 | brain.js Best | brain.js | mixed | 0.01 | 200 | - | ✗ | ✗ | **91%** | 0% | 8m |
| exp22 | Cluster (brain.js) | brain.js | mixed | 0.01 | 200 | - | ✗ | ✓ | 84% | 0% | 8m |
| TF-B5_conv | Phase 2: Conv+Value | TF.js | 100% net | 0.01 | 10 | 256 | ✓ | ✗ | 50% | 0% | 42m |
| TF-B6_optimized | Hyperparameter Tuning | TF.js | 100% net | 0.02 | 10 | 256 | ✓ | ✗ | **50%** | 0% | 42m |
| TF-B8-V2 | MCTS-50 (Pre-Audit) | TF.js | 100% net | 0.05 | 10 | 256 | ✓ | ✗ | 38% (54% pk) | 0% | 432m |
| TF-B8 | MCTS-50 (Bugs 6–10 Fixed) | TF.js | 100% net | 0.05 | 10 | 256 | ✓ | ✗ | 47% | 0% | 158m |
| TF-B9 | MCTS-50 (CE Sum + Adam) | TF.js | 100% net | 0.001 | 10 | 256 | ✓ | ✗ | **79%** (84% pk) | 0% | 145m |
| TF-B10 | MCTS-50 Mixed (50/50 Heur) | TF.js | 50/0/50 | 0.001 | 10 | 256 | ✓ | ✗ | **87%** | 0% | 161m |
| TF-B12 | Quiescence MCTS + Mixed | TF.js | 50/0/50 | 0.001 | 10 | 256 | ✓ | ✗ | **100%** | **63%** | 271m |
| TF-B13 | Dual-Side Quiescence + 75/25 | TF.js | 75/0/25 | 0.0005 | 10 | 256 | ✓ | ✗ | **100%** | **73% (89% pk)** | 341m |
| TF-B14 | Sliding-Window Pure SP + CosLR | TF.js | 100% net | 4e-4→1e-4 | 10 | 256 | ✓ | ✗ | **100%** | **78% (90% pk, 80.1% avg)** | 397m |
| **TF-B15** | **Full $T^2$ Torus + Circular Conv + Safety** | **TF.js** | **100% net** | **3e-4→8e-5** | **10** | **256** | **✓** | **✗** | **100%** | **99% Raw (95.9% avg) / 100% +Safety** | **585m** |

**Key:**
- Mix: opponent distribution (network/random/heuristic)
- LR: learning rate (constant or cosine decay schedule)
- Grad Iso: isolated gradient updates on active actions
- Cluster: TD(λ) intermediate cluster reward shaping

### Reading the Telemetry Across All 5 Stages

To see *why* our telemetry metrics (introduced in [Chapter 2](./02-rl-fundamentals.md)) were so essential, compare the diagnostic signatures of key milestone runs across the five architectural stages:

| Experiment | Stage & Architecture | Win Rate vs Random (P1 / P2 Split) | Win Rate vs Heuristic (P1 / P2 Split) | Avg Cluster Score vs Heuristic (AI : Heur) | Cycle Error Curve ($e_{\text{start}} \rightarrow e_{\text{end}}$) | Opening / Board Diversity (bits) | Diagnostic Interpretation |
|---|---|---|---|---|---|---|---|
| **`exp20`** | Stage 1: `brain.js` MLP + 50% Replay | **91.0%** (`88%` / `94%`) | **0.0%** (`0%` / `0%`) | `6.2 : 168.4` | `0.0061 → 0.0049` (Full-400 MSE) | `3.8b` / `1.72b` | Brute-forced legal move suppression beats `Random`, wiped out by Heuristic cascades. |
| **`TF-B1_v3`** | Stage 1: `TF.js` MLP (Grad Iso, No Replay) | **37.0%** (`32%` / `42%`) | **0.0%** (`0%` / `0%`) | `4.1 : 171.2` | `0.246 → 0.0000` ⚠️ (Overfit!) | `2.9b` / `1.65b` | Error crashing to `0.0000` exposed catastrophic overfitting on current-cycle games. |
| **`TF-B5_conv`** | Stage 2: `10×10×18` ConvNet Actor-Critic | **50.0%** (`46%` / `54%`) | **0.0%** (`0%` / `0%`) | `14.8 : 154.0` | `1.910 → 1.380` (Healthy sawtooth) | `4.6b` / `1.89b` | Spatial features learned, but 1-ply policy cannot foresee 2-ply cavity traps. |
| **`TF-B8-V2`** | Stage 3: Pre-Audit MCTS-50 (Bugs #6–#11 active) | **38.0%** (`15%–54%` osc.) | **0.0%** (`0%` / `0%`) | `8.5 : 162.1` | `1.312 → 1.299` ⚠️ (Micro-drop!) | `5.3b` / `1.96b` | `0.013` error drop exposed `1/400` MSE policy dilution + dual-perspective cancellation. |
| **`TF-B10`** | Stage 3: Fixed MCTS-50 (Masked CE Sum + Adam) | **87.0%** (`84%` / `90%`) | **0.0%** (`0%` / `0%`) | `48.2 : 118.5` | `3.820 → 2.410` (Strong gradient) | `5.8b` / `1.97b` | Cluster score vs Heuristic jumped $10\times$ (`4.1 → 48.2`), proving MCTS works—but Horizon Effect still blocks wins. |
| **`TF-B12`** | Stage 4: Quiescence MCTS + 50/50 Mixed | **100.0%** (`100%` / `100%`) | **63.0%** (`58%` / `68%`) | `89.4 : 76.2` | `3.450 → 2.180` | `5.9b` / `1.97b` | **Heuristic Wall Broken!** Quiescence cascade rollouts expose cavity blunders at MCTS leaves. |
| **`TF-B14`** | Stage 4: Dual-Side Quiescence + 45K Sliding Replay | **100.0%** (`100%` / `100%`) | **78.0% / 90.0% pk** (`80.1%` avg) | `106.3 : 63.8` | `2.890 → 1.940` (Cosine LR decay) | `6.1b` / `1.98b` | Pure self-play with sliding replay + cosine LR reaches `90%` peak; limited only by zero-padded border blind spots. |
| **`TF-B15`** | Stage 5: Full $T^2$ `ToroidalPad2D` + Safety | **100.0%** (`100%` / `100%`) | **99.0% Raw / 100.0% +Safety** | **`138.4 : 31.8`** | `2.610 → 1.720` (Smooth convergence) | `6.2b` / `1.98b` | **Total Mastery:** Circular torus convolutions achieve `95.9%` avg raw 1-ply (`99%` peak) and `1,000W / 0L` with safety filter! |

### Complete Parameter Glossary (All Stages)

| Parameter | CLI Flag / Config Key | Symbol | Range / Default | Architectural Stage & Effect |
|-----------|-----------------------|--------|-----------------|------------------------------|
| Learning Rate | `--learning-rate` | $\alpha$ | `8e-5` – `0.05` | Step size (`0.01–0.05` for SGD; `1e-3` or Cosine `3e-4 → 8e-5` for Adam on ConvNet) |
| Cosine Min LR | `--lr-min` | $\alpha_{\min}$ | `8e-5` – `1e-4` | Final learning rate at Cycle $C$ under `--lr-schedule cosine` (`TF-B14`, `TF-B15`) |
| Momentum | `momentum` | $\beta$ | `0.0` – `0.99` | Velocity smoothing for Stage 1 SGD (`0.9`); superseded by Adam $(\beta_1=0.9, \beta_2=0.999)$ |
| Exploration | `--exploration` | $\varepsilon$ | `0.0` – `1.0` | Random move probability in Stage 1; superseded by MCTS Dirichlet noise in Stages 3–5 |
| Epochs / Iters | `--epochs` | $E$ | `1` – `200` | Training passes per cycle (`10` epochs over replay buffer in `TF.js` ConvNet) |
| Batch Size | `--batch-size` | $B$ | `1` – `512` (`256`) | Samples per gradient update (`256` saturates M2 Ultra AMX matrix coprocessor) |
| Replay Buffer | `--replay-buffer-size` | $|D|$ | `45,000` – `500,000` | Sliding-window FIFO experience memory (`45,000` = last 3 cycles in `TF-B14`/`TF-B15`) |
| Replay Ratio | `--replay-ratio` | $r_{\text{replay}}$ | `1.0` – `4.0` (`2.0`) | Ratio of historical replay samples mixed with current-cycle self-play samples |
| MCTS Sims | `--mcts-sims` | $N_{\text{sims}}$ | `25` – `200` (`50`) | Tree search simulations per move during self-play generation (`50` sims, batched `8`) |
| PUCT Constant | `--c-puct` | $c_{\text{puct}}$ | `1.0` – `2.5` (`1.41`) | Balances MCTS exploitation $Q(s,a)$ vs neural prior exploration $P(s,a)$ |
| Quiescence Mode | `--quiescence-mode` | - | `none \| greedy` | Plays out forced opponent enclosure cascades at MCTS leaves before $V(s)$ evaluation |
| Quiescence Depth| `--quiescence-depth` | $d_q$ | `1` – `10` (`5`) | Maximum cascade plies simulated during leaf quiescence extension |
| Risk Penalty | `--risk-penalty` | $w_{\text{risk}}$ | `0.0` – `0.5` (`0.20`) | Value penalty per cascaded opponent claim when a move passes turn with open cavities |
| Enclosure Bonus | `--enclosure-bonus` | $w_{\text{enc}}$ | `0.0` – `0.5` (`0.15`) | Value bonus for moves that immediately capture enclosures and earn bonus turns |
| Toroidal Pad | `--toroidal-pad` | - | `true \| false` | Wraps all $3\times 3$ `Conv2D` layers in `ToroidalPad2D` for circular $T^2$ continuity (`TF-B15`) |
| Safety Filter | `--safety-filter` | - | `true \| false` | Top-$K$ ($K=8$) 1-ply tactical blunder check during fast inference (`100%` vs Heuristic) |

---

---

> [← Chapter 8: The Heuristic Wall](./08-the-heuristic-wall.md) | **[📚 Training Book Index](./README.md)** | [Chapter 10: Key Takeaways →](./10-key-takeaways.md)
