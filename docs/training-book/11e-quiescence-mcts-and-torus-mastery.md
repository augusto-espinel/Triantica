> **[📚 Training Book Index](./README.md)** | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | [← 11D: Monte Carlo Tree Search (MCTS)](./11d-mcts-theory-and-implementation.md) | [Chapter 12: Toroidal 3D Visualization & Human Telemetry →](./12-toroidal-3d-and-telemetry.md)

---

### Phase 5.5: Quiescence MCTS — Why It Improves Over Simple MCTS

**Status:** ✅ Completed (Quiescence MCTS Engine, `src/server/QuiescenceMCTS.cjs`, and Mathematical Specification, `docs/QUIESCENCE_MCTS.md`) — **Broke the Heuristic Wall in `TF-B12` (`63%` vs Heuristic), Reached `89%` Peak in `TF-B13`, Achieved `90.0%` Peak (`80.1%` Avg) in `TF-B14`, and Culminated in `99.0%` Peak (`95.9%` Avg) Raw / `100.0%` (`1,000W/0L`) +Safety in `TF-B15` (`100%` vs Random across all 40 cycles)**

#### 1. The Horizon Effect & Why Simple MCTS Fails Against Deterministic Cascades
In `TF-B10` (Simple MCTS-50 + 50% Heuristic opponents), the AI reached a record **87% vs Random**, yet remained at **0% vs `HeuristicPlayer`** (averaging only ~14 turns to Heuristic's ~96 turns per game). An inflection-point audit of `TF-B10` exposed two structural blind spots in the standard Monte Carlo Tree Search engine (`MCTS.cjs`):

1. **The Horizon Effect at Unquiet Leaves:**
   Simple MCTS expands a leaf node $s_L$ after a single candidate move and immediately queries the neural Value Head $V_\theta(s_L)$. If $s_L$ is a **"noisy" (tactically unstable) position**—specifically, a single move (`claimed = 0`) that passes the turn to the opponent while leaving a 3-sided enclosure hole—`HeuristicPlayer` will deterministically claim that enclosure with probability $P(\text{capture}) = 1.0$, earn an extra turn via $\lfloor (1 + \text{claimed})/2 \rfloor$, and chain a 5–10 move capture cascade.
2. **Self-Play Value Head Inversion on Single-Move Openings:**
   Because the Value Head was trained primarily on states where local cluster density correlated with winning against non-cascading play, $V_\theta(s_L)$ rated a hole-creating blunder at **`+0.4794`** while rating a safe move at **`+0.1060`**. With 50 simulations spread across ~35 legal moves, simple MCTS never searched deep enough down the forced opponent cascade to see the collapse—instead, UCT **amplified** the `+0.4794` blunder (`13` visits, `Q = +0.344` vs `1–2` visits for safe moves).

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Simple MCTS vs. Quiescence MCTS at a Leaf Node              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SIMPLE MCTS (Blind to Forced Cascades):                                │
│  Root ──(Move A: claimed=0, leaves hole)──► Leaf s_L                    │
│    • Evaluates raw V_θ(s_L) = +0.4794 ("cluster grew, looks great!")    │
│    • Backpropagates +0.4794 → UCT visits Move A 13 times (Q = +0.344)   │
│    • Reality: Heuristic chains 6 enclosures + extra turns → AI loses    │
│                                                                         │
│  QUIESCENCE MCTS (Resolves Tactical Cascades Before Evaluation):        │
│  Root ──(Move A: claimed=0, leaves hole)──► Leaf s_L                    │
│    1. Expands s_L using pre-cascade policy P_θ(s_L)                     │
│    2. Plays out greedy enclosure cascade: s_L → s_1 → s_2 → s_quiet     │
│    3. Evaluates V_θ(s_quiet) - Σ(oppClaimed × riskPenalty) = -0.65      │
│    4. Backpropagates -0.65 → UCT abandons Move A and selects safe move  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2. Game-Mechanic Invariant: Enclosing Trains vs. Single-Move Openings
A foundational rule of Triantica dictates when a move is safe versus exploitable:

$$\text{extraTurns} = \left\lfloor \frac{1 + \text{claimed}}{2} \right\rfloor$$

- **Enclosing Moves (`claimed > 0`) Are NEVER Risky (`risk = 0`):**
  Claiming even a single enclosed triangle (`claimed = 1`) awards $\lfloor 2/2 \rfloor = 1$ full extra turn, meaning **the active player retains the turn (`nextPlayer === mover`)**. Even if placing that triangle creates a new 3-sided pocket elsewhere on the board, the AI immediately moves again—allowing it to either **chain an enclosing train in its own favor** or patch the hole before the opponent ever gets a turn. Therefore, the enclosure risk calculators (`calculateEnclosureRisk` and `calculateChildNodeRisk` in `src/server/QuiescenceMCTS.cjs`) strictly enforce `risk = 0` whenever `claimed > 0 || nextPlayer === mover`.
- **Single-Move Openings (`claimed === 0`) Carry All Opponent Risk:**
  Only when `claimed === 0` (and `extraTurns === 0`) does the turn pass to the opponent. If that single move leaves enclosure opportunities $\mathcal{O}_{\text{opp}}$ for the opponent, each opportunity $o \in \mathcal{O}_{\text{opp}}$ contributes both a direct triangle-loss weight (`0.3` per enclosed triangle) and a turn-snowball weight (`0.5` per bonus turn earned):
  $$\text{risk}(s, a) = \begin{cases} 0 & \text{if } \text{claimed}(s, a) > 0 \;\lor\; \text{nextPlayer} = \text{mover} \\[6pt] \displaystyle\sum_{o \in \mathcal{O}_{\text{opp}}(s')} \Big(0.3 \cdot o.\text{claimed} + 0.5 \cdot \Big\lfloor \frac{1 + o.\text{claimed}}{2} \Big\rfloor\Big) & \text{otherwise} \end{cases}$$

#### 3. Four Mechanisms That Make `QuiescenceMCTS` a Superior Teacher
Recall that during evaluation (`getAIMove`) and browser/mobile play (via the AI player controller, `AIPlayer.js`), the AI uses a **single forward pass of the Policy Head** (no MCTS at test time). MCTS is the **training teacher** that generates target distributions $\pi_{\text{MCTS}}$ and value targets for the 2-head ConvNet. The Quiescence MCTS engine (`src/server/QuiescenceMCTS.cjs`) directly upgrades the teacher in four ways without needing a 3rd network head or artificial TD($\lambda$) reward shaping:

1. **Root Prior Shaping (`_expandRootWithRisk`):**
   At the root node, every legal child move records `child.claimed` and `child.enclosureRisk`.
   - Enclosing moves (`child.claimed > 0`) receive both a multiplicative and additive prior boost, ensuring that even if an early-cycle Policy Head assigns $P_0(a) \approx 0$ to an enclosing move, the additive floor (`+0.05 × claimed`) guarantees the 50 MCTS simulations immediately explore offensive **enclosing trains**:
     $$P'(s_0, a) \propto \begin{cases} P_0(s_0, a)\cdot\big(1 + 1.5\cdot\text{claimed}\big) + 0.05\cdot\text{claimed} & \text{if } \text{claimed} > 0 \\[4pt] P_0(s_0, a)\cdot\max\!\big(0.05,\; 1 - 2\cdot w_{\text{risk}}\cdot\text{risk}\big) & \text{if } \text{claimed} = 0 \;\land\; \text{risk} > 0 \\[4pt] P_0(s_0, a) & \text{otherwise} \end{cases}$$
   - Hole-creating single moves (`child.claimed === 0` and `risk > 0`) have their prior scaled down by $\max(0.05, 1 - 2 \cdot \text{risk} \cdot w_{\text{risk}})$, preventing wasted simulations on obvious blunders.
2. **Concatenated Dual-State GPU Batching for Pre-Cascade Expansion + Post-Cascade Quiescent Value (`_runBatch`):**
   During batched MCTS (`searchBatched`), every non-terminal leaf `leaf` requires two distinct evaluations when a tactical cascade occurs (`qEval.depth > 0`):
   - **Pre-Cascade Expansion (`expandStates`):** `leaf` must be expanded using the neural policy $P_\theta(\text{leaf.grid})$ of the true **pre-cascade** board so child moves and priors match `leaf.grid`.
   - **Post-Cascade Quiescent Value (`quietStates`):** `leaf` must be evaluated by running a deterministic greedy cascade (`_runQuiescenceRollout`, up to `quiescenceDepth = 5`, configurable across `quiescenceMode ∈ {'none', 'greedy', 'heuristic', 'hybrid'}`) to reach a quiet board `qEval.grid`, then combining the perspective-signed neural value $\pm V_\theta(\text{qEval.grid})$ with tactical cascade adjustments (`+claimed × enclosureBonusWeight` for AI claims, `-claimed × riskPenaltyWeight` for opponent claims):
     $$V_{\text{leaf}} = \text{clip}_{[-1, 1]}\!\left(\text{sign}(p_{\text{quiet}}, p_{\text{root}})\cdot V_\theta(s_{\text{quiet}}) + w_{\text{bonus}}\sum \text{claimed}_{\text{AI}} - w_{\text{risk}}\sum \text{claimed}_{\text{opp}}\right)$$
     *(Note: If the quiescence cascade fills the board or leaves zero legal moves, `_getGridTerminalValue(qEval.grid, rootPlayer)` computes the exact terminal cluster winner $\in \{-1, 0, +1\}$ directly without a neural call.)*
   - **Zero-Extra-Roundtrip GPU Packing:** Rather than issuing two separate GPU calls per simulation batch, `_runBatch` concatenates `allStates = expandStates.concat(quietStates)` (up to `8 + 8 = 16` canonical `10×10×18` tensors) into a **single combined `network.batchPredict(allStates)` call**, slicing `policies[0 .. expandStates.length - 1]` for leaf expansion and `values[quietOffset + k]` for quiescent leaf evaluation!
3. **Direct Policy Target Distillation (`_getRiskAdjustedPolicy`):**
   After all 50 simulations finish, the raw root visit distribution $N(s_0, a)/N(s_0)$ is further refined before being returned as the training target $\pi_{\text{MCTS}}$:
   $$\pi_{\text{MCTS}}(a) \propto \begin{cases} \dfrac{N(s_0, a)}{N(s_0)}\big(1 + w_{\text{bonus}}\cdot\text{claimed}\big) + 0.02\cdot\text{claimed} & \text{if } \text{claimed} > 0 \\[8pt] \dfrac{N(s_0, a)}{N(s_0)}\max\!\big(0.02,\; 1 - 2\cdot w_{\text{risk}}\cdot\text{risk}\big) & \text{if } \text{claimed} = 0 \;\land\; \text{risk} > 0 \\[8pt] \dfrac{N(s_0, a)}{N(s_0)} & \text{otherwise} \end{cases}$$
   Because the ConvNet Policy Head trains directly via cross-entropy against $\pi_{\text{MCTS}}$, **the 1-ply Policy Head internalizes both offensive enclosing trains and defensive hole avoidance**—which is verified in the standalone quiescence regression suite (`test_quiescence.cjs`), where `searchBatched()` selects enclosing chains even when initialized with a completely flat (`0.5` uniform) mock network prior.
4. **$O(1)$ Local Neighborhood Enclosure Detection (`countWouldBeEnclosed`):**
   Because `isTriangleEnclosed(cx, cy, rot)` only depends on cell `(cx, cy)` and its 4 orthogonal neighbors, placing a triangle at `(mx, my)` can only enclose `(mx, my)` itself or one of its 4 orthogonal neighbors:
   $$\mathcal{N}_5(x, y) = \Big\{(x, y),\; \big((x+1)\bmod 10, y\big),\; \big((x+9)\bmod 10, y\big),\; \big(x, (y+1)\bmod 10\big),\; \big(x, (y+9)\bmod 10\big)\Big\}$$
   Pushing the candidate triangle onto `grid[my][mx].triangles`, checking just those 5 cells in-place via `mapping = [2, 3, 0, 1]` (the complementary rotation in a 1-triangle cell), and popping the triangle back off avoids cloning the $10 \times 10$ grid (`80×` fewer checks than scanning all 400 slots) and runs **3.2× faster** (~140ms per 50-sim move on M2 Ultra).

#### 4. Experiment `TF-B12` Results: Breaking the Heuristic Wall (`100%` vs Random, `63%` vs Heuristic)
Warm-started from `TF-B10_network.json`, `TF-B12` ran 10 cycles (100 games/cycle, 50% `QuiescenceMCTS-50` self-play / 50% `HeuristicPlayer`, `quiescenceMode = greedy`, `quiescenceDepth = 5`, `riskPenalty = 0.2`, `enclosureBonusWeight = 0.15`, `LR = 0.001` Adam, runtime `270.5 min`).

| Cycle | Loss (`Start → End`) | Training MCTS-50 vs Heuristic (`W/L/D`, `P1`, `P2`, `Med AI–Opp`) | Eval vs Random (`100 games`) | Mean Score vs Random (`AI – Rand`) | Eval vs Heuristic (Dashboard) | Eval vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`) | Eval Mode vs Heuristic |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0 (`TF-B10`)** | — | `8.0%` (`4W/46L`, `med 4–177`) | `87.0%` | `79.2 – 26.8` | `0.0%` | `0% / 0%` | `7.8 – 175.9` | Both Det & Random 2-Ply (`0%`) |
| **1** | `5.204 → 3.146` | **`32.1%`** (`18W/38L`, P1:`10/27`, P2:`8/29`, `med 69.5–118`) | **100.0%** | `116.7 – 16.0` | `0.0%` | `0% / 0%` | `12.0 – 169.0` | Deterministic (2 trajectories) |
| **2** | `4.827 → 3.116` | **`31.3%`** (`15W/32L/1D`, P1:`8/26`, P2:`7/22`, `med 74.5–109.5`) | **100.0%** | `130.0 – 11.3` | `0.0%` | `0% / 0%` | `45.0 – 141.5` | Deterministic (2 trajectories) |
| **3** | `4.418 → 3.029` | **`36.0%`** (`18W/32L`, P1:`8/22`, P2:`10/28`, `med 76–105.5`) | **100.0%** | `144.2 – 8.8` | `0.0%` | `0% / 0%` | `55.0 – 141.0` | Deterministic (2 trajectories) |
| **4** | `4.378 → 3.013` | **`42.6%`** (`20W/27L`, P1:`10/29`, P2:`10/18`, `med 89–99`) | **100.0%** | `150.3 – 8.1` | `0.0%` | `0% / 0%` | `36.5 – 147.0` | Deterministic (2 trajectories) |
| **5** | `4.149 → 3.002` | **`29.5%`** (`13W/31L`, P1:`5/17`, P2:`8/27`, `med 79–109`) | **100.0%** | `152.2 – 7.7` | `0.0%` | `0% / 0%` | `79.0 – 111.0` | Deterministic (2 trajectories) |
| **6** | `4.033 → 2.961` | **`40.0%`** (`22W/33L`, P1:`10/26`, P2:`12/29`, `med 79–105`) | **100.0%** | `157.6 – 5.8` | **🏆 `50.0%`** | `0% / 100%` | **`91.0 – 77.5`** | Deterministic (`P2 won 147–24!`) |
| **7** | `4.007 → 2.961` | **`35.7%`** (`15W/27L`, P1:`9/22`, P2:`6/20`, `med 79.5–102.5`) | **100.0%** | `162.1 – 5.4` | `0.0%` | `0% / 0%` | `58.0 – 115.5` | Deterministic (2 trajectories) |
| **8** | `3.984 → 2.947` | **`35.8%`** (`19W/34L`, P1:`9/29`, P2:`10/24`, `med 75–108`) | **100.0%** | `163.7 – 5.1` | **🏆 `50.0%`** | **`50% / 50%`** | **`90.3 – 88.4`** | **Random First 2 Plies (100 openings)** *(Det: `50%`)* |
| **9** | `3.892 → 2.951` | **`49.0%`** (`24W/25L`, P1:`12/27`, P2:`12/22`, `med 88–93`) | **100.0%** | `163.1 – 5.1` | **🚀 `62.0%`** | **`68% / 56%`** | **`100.8 – 78.8`** | **Random First 2 Plies (100 openings)** *(Det: `0%`)* |
| **10** | `3.720 → 2.899` | **`42.3%`** (`22W/29L/1D`, P1:`12/26`, P2:`10/26`, `med 79–108`) | **100.0%** | **`166.2 – 4.4`** | **🚀 `63.0%`** | **`66% / 60%`** | **`101.7 – 79.5`** | **Random First 2 Plies (100 openings)** *(Det: `50%`)* |

#### 5. Methodological Discovery: Why Random First 2 Plies Are Essential for `Eval vs Heuristic`
During `TF-B12`, we discovered a critical flaw in how `Eval vs Heuristic` had been measured since `TF-B1`:
- Because both `getAIMove(..., exploration=0)` ( greedy argmax ) and `new HeuristicPlayer(..., { explorationFactor: 0 })` (with deterministic cell-order tie-breaking in `HeuristicPlayer.getMove`) are **100% deterministic**, running 100 evaluation games from an empty board was actually replaying **only 2 unique game trajectories** (50 identical copies of 1 game with AI as P1, and 50 identical copies of 1 game with AI as P2).
- Consequently, `Eval vs Heuristic` could only ever output `0%`, `50%`, or `100%`! For example, in Cycle 9, the network lost its 2 canonical trajectories (`0%` deterministic), even though its raw 1-ply policy was actually winning **62.0% of games (`68%` as P1, `56%` as P2)** across diverse openings!
- **Protocol Upgrade (Cycle 8 Onwards & Permanent Default in the Self-Play Training Pipeline, `train_tfjs.cjs`):**
  Starting at Cycle 8 of `TF-B12` (and now permanently baked into the self-play training pipeline, `src/server/train_tfjs.cjs`, and the 2-ply evaluation harness, `scripts/eval_random_2ply.cjs`), `Eval vs Heuristic` plays **random moves for the first 2 plies (`m < 2`, i.e., 1 random opening triangle per player)** before switching to `exploration = 0` greedy policy vs deterministic `HeuristicPlayer`. This tests the trained network across **100 genuinely distinct opening board configurations** while keeping all mid-game and endgame play strictly deterministic.
- As validation, we also ran the exact same 100-game random-2-ply audit on `TF-B10_network.json` (Cycle 0 baseline): `TF-B10` still scored **`0.0%` (`0W / 100L`, mean cluster `7.8` vs `175.9`)**, proving that `TF-B12`'s jump from **`0.0%` → `50.0%` (C8) → `62.0%` (C9) → `63.0%` (C10)** is a genuine, massive architectural breakthrough driven by `QuiescenceMCTS`.

#### 6. Experiment `TF-B13` Results: Scaling to `89.0%` Peak & `73.0%` Final vs Heuristic (`100%` vs Random)
Warm-started from `TF-B12_network.json`, `TF-B13` ran 10 cycles (100 games/cycle, 75% dual-side `QuiescenceMCTS-50` self-play / 25% `HeuristicPlayer`, `quiescenceMode = greedy`, `quiescenceDepth = 5`, `riskPenalty = 0.25`, `enclosureBonusWeight = 0.15`, `LR = 0.0005` Adam, runtime `340.8 min`, `115,892` total replay buffer samples). All 10 cycles were evaluated across 100 distinct random-first-2-plies openings against both `RandomPlayer` and `HeuristicPlayer`.

| Cycle | Loss (`Start → End`) | Buffer Samples (`Util`) | Training MCTS-50 vs Heuristic (`W/L`, `P1`, `P2`, `Med AI–Opp`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic (`100 games`) | Eval vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B12`)** | `3.720 → 2.899` | `60,961` (`12.2%`) | `42.3%` (`22W/29L/1D`, `med 79–108`) | **100.0%** (`166.2 – 4.4`) | `63.0%` | `66% / 60%` | `101.7 – 79.5` | `3.78 / 1.84` |
| **1** | `4.046 → 2.934` | `12,036` (`2.4%`) | **`75.0%`** (`12W/4L`, P1:`6/7`, P2:`6/9`, `med 127.5–57`) | **100.0%** (`165.2 – 4.4`) | **`52.0%`** (`52W/47L/1D`) | `44% / 60%` | `91.8 – 87.7` | `4.13 / 1.84` |
| **2** | `4.002 → 2.912` | `23,507` (`4.7%`) | **`52.2%`** (`12W/11L`, P1:`4/15`, P2:`8/8`, `med 95–94`) | **100.0%** (`167.8 – 3.7`) | **`61.0%`** (`61W/39L/0D`) | `54% / 68%` | `98.0 – 78.2` | `3.96 / 1.82` |
| **3** | `3.829 → 2.874` | `34,816` (`7.0%`) | **`79.3%`** (`23W/6L`, P1:`10/11`, P2:`13/18`, `med 124–62`) | **100.0%** (`167.7 – 3.9`) | **`66.0%`** (`66W/34L/0D`) | `70% / 62%` | `102.8 – 75.0` | `4.21 / 1.81` |
| **4** | `3.723 → 2.881` | `46,558` (`9.3%`) | **`77.3%`** (`17W/5L`, P1:`9/11`, P2:`8/11`, `med 132.5–59.5`) | **100.0%** (`171.8 – 3.2`) | **`72.0%`** (`72W/28L/0D`) | `74% / 70%` | `107.8 – 72.4` | `4.33 / 1.81` |
| **5** | `3.838 → 2.880` | `57,556` (`11.5%`) | **`71.0%`** (`22W/9L`, P1:`10/14`, P2:`12/17`, `med 113–84`) | **100.0%** (`171.6 – 3.3`) | **`71.0%`** (`71W/29L/0D`) | `78% / 64%` | `108.2 – 69.8` | `4.19 / 1.79` |
| **6** | `3.724 → 2.886` | `69,363` (`13.9%`) | **🔥 `90.9%`** (`20W/2L`, P1:`9/9`, P2:`11/13`, `med 114.5–71.5`) | **100.0%** (`172.0 – 3.2`) | **`75.0%`** (`75W/25L/0D`) | `74% / 76%` | `117.8 – 62.3` | `4.07 / 1.82` |
| **7** | `3.723 → 2.883` | `81,090` (`16.2%`) | **`73.7%`** (`14W/5L`, P1:`7/10`, P2:`7/9`, `med 109–71`) | **100.0%** (`172.6 – 3.4`) | **`79.0%`** (`79W/19L/2D`) | `78% / 80%` | `117.9 – 61.6` | `3.94 / 1.82` |
| **8** | `3.777 → 2.908` | `93,218` (`18.6%`) | **`81.3%`** (`13W/3L`, P1:`10/10`, P2:`3/6`, `med 122.5–69`) | **100.0%** (`173.6 – 3.0`) | **🏆 `89.0%`** (`89W/11L/0D`) | **`90% / 88%`** | **`126.7 – 52.6`** | `4.51 / 1.84` |
| **9** | `3.718 → 2.894` | `104,119` (`20.8%`) | **`62.5%`** (`20W/12L`, P1:`9/18`, P2:`11/14`, `med 107–84`) | **100.0%** (**`173.9 – 3.0`**) | **`78.0%`** (`78W/22L/0D`) | `86% / 70%` | `121.9 – 57.3` | `4.27 / 1.81` |
| **10** | `3.645 → 2.905` | `115,892` (`23.2%`) | **`58.8%`** (`10W/7L`, P1:`7/11`, P2:`3/6`, `med 99–80`) | **100.0%** (`172.9 – 3.5`) | **🚀 `73.0%`** (`73W/27L/0D`) | **`80% / 66%`** | **`114.1 – 65.8`** | `4.63 / 1.83` |

**Key Takeaways from `TF-B13`:**
- **Sustained Dominance Over `HeuristicPlayer`:** The standalone 1-ply ConvNet (`exploration = 0`, no MCTS at eval time) climbed steadily from `52.0%` (Cycle 1) to an all-time peak of **`89.0%` (`89W / 11L`, `90%` as P1, `88%` as P2, mean cluster `126.7` vs `52.6`)** in Cycle 8, finishing at **`73.0%` (`80%` as P1, `66%` as P2)** in Cycle 10 and averaging **`71.6%`** across all 10 cycles.
- **MCTS-50 Self-Play vs Heuristic:** Across all 227 training games played against `HeuristicPlayer`, `QuiescenceMCTS-50` won **`71.8%` (`163W / 64L`)**, peaking at **`90.9%` (`20W / 2L`)** in Cycle 6.
- **Flawless vs Random:** Achieved **`100.0%` (`1,000W / 0L / 0D`)** across all 10 cycles (`2,000` consecutive wins across `TF-B12` + `TF-B13`), with mean AI cluster size reaching **`173.9` vs `3.0`** in Cycle 9.
- **Dual-Side MCTS Efficiency:** Collecting both players' MCTS policy targets in network-vs-network games nearly doubled the replay buffer growth (`115,892` samples vs `60,961` in `TF-B12`) while maintaining balanced 50/50 self-play parity (`503` P1 wins vs `494` P2 wins, `3` draws across 1,000 games).

#### 7. Post-`TF-B13` Forensic Audit: Architectural Ceiling, Remaining Bottlenecks & Toroidal Topology

While `TF-B13` reached **`89.0%` vs Heuristic** in Cycle 8 (using a raw 1-ply ConvNet forward pass with `exploration = 0`), Cycles 9 and 10 dipped slightly to `78.0%` and `73.0%`. A forensic audit of the training dynamics, network padding, and game rules identified **five key bottlenecks** that explain both the `89%` peak and the path to `95%–99%+` mastery—as well as how to generalize Triantica to **true 3D toroidal and arbitrary manifold surfaces**:

1. **Bottleneck 1 — Replay Buffer Staleness ("Ghost of Cycle 1"):**
   - `TF-B13` used an unbounded `500,000`-capacity replay buffer (`115,892` samples accumulated by Cycle 10 at `~11.6K` samples/cycle).
   - Because early samples were never evicted, by Cycles 9–10 roughly **`70%` of training mini-batches were drawn from Cycles 1–6** (when the policy was weaker: `52%–75%` vs Heuristic). Training 10 epochs at a constant `LR = 0.0005` over stale trajectories pulled Cycle 8's converged weights (`loss = 2.883`) back toward earlier MCTS visit distributions (`loss = 2.908` in C8, `2.905` in C10).
   - **Next Step:** Implement a **sliding-window replay buffer** (`35,000–50,000` samples = last 3–4 cycles, AlphaZero style) + **cosine/step LR decay** (`0.0005 → 0.0001`) + automatic **`best_network.json` checkpointing** whenever `eval.vsHeuristic` sets a new high-water mark.

2. **Bottleneck 2 — Edge Asymmetry in Corner Placement (`hasCorner`) vs. Toroidal Enclosures & Clusters:**
   - In the core board & rules engine (`src/core/GridManager.js` and `game.js`), **edge adjacency** (`getAdjacentTriangles`) and **enclosure detection** (`isTriangleEnclosed`) wrap around the `10×10` grid toroidally via `(x ± 1 + cols) % cols` and `(y ± 1 + rows) % rows`.
   - However, **corner placement legality** (`hasCorner(x, y)` + `getTriangleCorners(x, y, rotation)`) uses **unwrapped planar vertex coordinates** `cx ∈ [0..cols]`, `cy ∈ [0..rows]` and checks `if (cellX >= 0 && cellY >= 0 && cellX < this.cols && cellY < this.rows)`.
   - **Consequence:** A triangle placed on border `x = 0` places corners on vertex line `cx = 0`, which does **not** wrap to `cx = cols` (`10`). Thus, players **cannot** branch legal placements across the outer boundary based on corner contact alone, even though enclosures and cluster merges **do** wrap across the boundary!
   - **Topological Fix (True Torus $T^2$):** In a true $10 \times 10$ torus, vertex `(cols, y)` is identical to `(0, y)` and `(x, rows)` is identical to `(x, 0)` (there are exactly `rows × cols = 100` unique vertices, not `11 × 11 = 121`). Wrapping vertex coordinates modulo `(cols, rows)` inside `hasCorner`:
     ```js
     hasCorner(x, y) {
       const vx = ((x % this.cols) + this.cols) % this.cols;
       const vy = ((y % this.rows) + this.rows) % this.rows;
       for (let dy = -1; dy <= 0; dy++) {
         for (let dx = -1; dx <= 0; dx++) {
           const cellX = (vx + dx + this.cols) % this.cols;
           const cellY = (vy + dy + this.rows) % this.rows;
           const cell = this.grid[cellY][cellX];
           if (cell.triangles.some(t =>
             this.getTriangleCorners(cellX, cellY, t.rotation).some(([tx, ty]) =>
               (tx % this.cols) === vx && (ty % this.rows) === vy
             )
           )) {
             return true;
           }
         }
       }
       return false;
     }
     ```
     eliminates the artificial planar boundary completely, making every cell topologically identical—essential for a **3D rotatable torus UI** where there is no "edge" of the board.

3. **Bottleneck 3 — Zero-Padding (`padding: 'same'`) vs. Circular Toroidal Convolutions:**
   - `TFConvNetwork` currently applies `padding: 'same'` (zero-padding) in its three `3×3` `Conv2D` layers (`receptive field = 7×7`).
   - Because zero-padding fills out-of-bounds neighbors with `0` instead of wrapping to the opposite side of the board, the ConvNet sees artificial walls at `x = 0, 9` and `y = 0, 9`, splitting wrap-around enclosures across opposite sides of the tensor.
   - **Next Step:** Once `hasCorner` is wrapped toroidally, replace zero-padding with **Toroidal Circular Padding** (wrapping 1 row/col around each border before each `3×3` conv, or circularly padding the input tensor), giving the `[32, 64, 64]` filters exact translation invariance across the entire torus.

4. **Bottleneck 4 — 1-Ply Raw Policy Inference vs. Inference-Time Hole-Risk Filtering:**
   - Evaluation and browser play currently select moves using the raw 1-ply policy head (`predictMove()`, `exploration = 0`) with zero search lookahead.
   - Applying `QuiescenceMCTS`'s $O(1)$ local neighborhood risk detector (`countWouldBeEnclosed`) as a lightweight **inference-time safety filter** on the top-$K$ policy candidates (or running light `QuiescenceMCTS-25` at play time) eliminates residual 1-ply hole blunders immediately without changing network weights.

5. **Bottleneck 5 — 8-Fold Dihedral Symmetry ($D_4$) & Future 3D Manifold Surfaces (Graph / Mesh Generalization):**
   - On the $10 \times 10$ square torus, exploiting the 8 symmetries of $D_4$ (4 rotations × 2 reflections, permuting triangle rotations `[0, 1, 2, 3]`) yields an **8× sample efficiency multiplier** (and full $10 \times 10 \times 8 = 800$-fold symmetry once toroidal translation is enabled).
   - **Future Vision — Arbitrary 3D Surfaces:** When Triantica expands beyond a regular $10 \times 10$ torus to **arbitrary closed 3D triangulated meshes** (e.g., spheres, higher-genus surfaces, icosahedral geodesic grids, or irregular 3D manifolds rotated interactively by the player), a planar 2D grid `(x, y)` no longer applies. Instead, the board is naturally represented as a **Dual Face-Adjacency Graph $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$**:
     - Each triangle slot is a node $v \in V_T$ (with state channels: empty, my piece, opponent piece).
     - **Enclosure & Cluster Contiguity** operate strictly over the **3 edge-neighbors** $\mathcal{N}_{\text{edge}}(v)$ of each triangle (`isTriangleEnclosed` = all 3 edge-neighbors occupied; clusters = connected components over $\mathcal{N}_{\text{edge}}$).
     - **Placement Legality** operates strictly over the **3 vertices** $\mathcal{V}(v)$ of each triangle (`isValidPlacement` = at least one vertex in $\mathcal{V}(v)$ is already touched by an existing triangle).
     - **Neural Architecture:** On any 3D triangle mesh, the `3×3` grid `Conv2D` generalizes directly to a **Mesh / Graph Convolutional Network (GCN / Gauge-Equivariant MeshCNN)** aggregating features over $\mathcal{N}_{\text{edge}}(v)$ and $\mathcal{N}_{\text{vertex}}(v)$, while `QuiescenceMCTS` remains **100% unchanged** because it only queries legal moves and enclosure cascades!

#### 8. Experiment `TF-B14` Results: Bottleneck 1 Fixed + 100% Pure Network Self-Play (`90.0%` Peak / `80.1%` 10-Cycle Avg vs Heuristic)

Warm-started from `TF-B13_network.json`, **`TF-B14`** tested the resolution of **Bottleneck 1** (`45,000`-sample sliding-window `ReplayBuffer` + Cosine LR decay `0.0004 → 0.0001` with in-place Adam moment retention + automatic `*_best_network.json` peak checkpointing) under **100% pure network self-play** (`0%` Heuristic teacher, `Dual-Side QuiescenceMCTS-50`, runtime `397.2 min`, `129,962` total samples generated across 10 cycles):

| Cycle | Cosine LR | Buffer (`45K` Window, `Total Added`) | Loss (`Start → End`) | Self-Play (`P1 / P2 / D`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic (`100 games`) | Role Split vs Heuristic (`P1 / P2`) | Mean Score vs Heuristic (`AI – Heur`, `Margin`) | Win/Loss Anatomy (`Blowout ≥+50 / CloseW / D / CloseL / CascadeL <-30`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B13` C10)** | `0.000500` | `115,892 / 500K` | `3.645 → 2.905` | — | **100.0%** (`172.9 – 3.5`) | `73.0%` (`73W/27L/0D`) | `80% / 66%` | `114.1 – 65.8` (`+48.3`) | — | `4.63 / 1.83` |
| **1** | `0.000400` | `12,973 / 45K` (`12,973`) | `3.933 → 2.829` | `54 / 46 / 0` | **100.0%** (`171.3 – 3.5`) | **`83.0%`** (`83W/17L/0D`) | `82% / 84%` | `120.4 – 58.3` (`+62.0`) | `58 / 25 / 0 / 8 / 9` | `4.06 / 1.86` |
| **2** | `0.000391` | `25,939 / 45K` (`25,939`) | `3.710 → 2.823` | `49 / 51 / 0` | **100.0%** (`171.7 – 3.5`) | **`76.0%`** (`76W/24L/0D`) | `86% / 66%` | `113.5 – 65.8` (`+47.7`) | `53 / 23 / 0 / 9 / 15` | `4.59 / 1.87` |
| **3** | `0.000365` | `38,972 / 45K` (`38,972`) | `3.743 → 2.863` | `53 / 47 / 0` | **100.0%** (`173.5 – 2.9`) | **`80.0%`** (`80W/20L/0D`) | `82% / 78%` | `117.4 – 63.3` (`+54.1`) | `61 / 19 / 0 / 7 / 13` | `4.67 / 1.88` |
| **4** | `0.000325` | **`45,000 / 45K`** (`52,062`) | `3.820 → 2.901` | `47 / 52 / 1` | **100.0%** (`172.7 – 3.2`) | **`79.0%`** (`79W/21L/0D`) | `80% / 78%` | `113.8 – 65.2` (`+48.6`) | `50 / 29 / 0 / 5 / 16` | `4.40 / 1.88` |
| **5** | `0.000276` | **`45,000 / 45K`** (`65,123`) | `3.817 → 2.963` | `43 / 56 / 1` | **100.0%** (`173.5 – 3.2`) | **`82.0%`** (`82W/18L/0D`) | `84% / 80%` | `122.2 – 59.0` (`+63.2`) | `63 / 19 / 0 / 7 / 11` | `4.38 / 1.87` |
| **6** | `0.000224` | **`45,000 / 45K`** (`78,036`) | `3.800 → 2.970` | `46 / 53 / 1` | **100.0%** (`173.2 – 2.9`) | **`76.0%`** (`76W/24L/0D`) | `78% / 74%` | `116.0 – 63.8` (`+52.3`) | `53 / 23 / 0 / 12 / 12` | `4.33 / 1.87` |
| **7** | `0.000175` | **`45,000 / 45K`** (`90,998`) | `3.886 → 2.977` | `67 / 33 / 0` | **100.0%** (`175.0 – 2.8`) | **`78.0%`** (`78W/22L/0D`) | `80% / 76%` | `118.7 – 60.6` (`+58.1`) | `62 / 16 / 0 / 8 / 14` | `3.99 / 1.86` |
| **8** | `0.000135` | **`45,000 / 45K`** (`104,004`) | `3.804 → 3.017` | `44 / 56 / 0` | **100.0%** (`174.2 – 3.0`) | **`79.0%`** (`79W/20L/1D`) | `82% / 76%` | `114.8 – 63.2` (`+51.7`) | `57 / 22 / 1 / 7 / 13` | `4.02 / 1.87` |
| **9** | `0.000109` | **`45,000 / 45K`** (`117,017`) | `3.916 → 3.081` | `41 / 59 / 0` | **100.0%** (`174.6 – 2.9`) | **🏆 `90.0%`** (`90W/9L/1D`) | **`88% / 92%`** | **`123.8 – 54.2` (`+69.7`)** | **`62 / 28 / 1 / 2 / 7`** | `4.43 / 1.87` |
| **10** | `0.000100` | **`45,000 / 45K`** (`129,962`) | `3.860 → 3.089` | `48 / 51 / 1` | **100.0%** (**`175.3 – 2.6`**) | **`78.0%`** (`78W/21L/1D`) | `78% / 78%` | `118.4 – 62.0` (`+56.5`) | `51 / 27 / 1 / 9 / 12` | **`5.03 / 1.85`** |

**Key Takeaways from `TF-B14` (Official 2D Baseline Champion — `v1.0.0-2d`):**
1. **New All-Time Record Peak (`90.0%` vs Heuristic in Cycle 9):**
   - In Cycle 9 (`LR = 0.000109`), the 1-ply ConvNet achieved **`90.0%` (`90W / 9L / 1D`, `88.0%` as P1 and `92.0%` as P2)** with an all-time record mean cluster score of **`123.8` vs `54.2` (`+69.7` margin)**, reducing close positional losses to just `2` games and cascade blunders to `7` games out of `100`.
   - Automatic peak checkpointing captured this exact model in **`dashboard/data/TF-B14_best_network.json`** (`MD5: f2eb302a23dc4a38397a7ed34bd27b2a`), which has been promoted to **`trainedAI_network.json`** as the official 2D Triantica AI.
2. **Massive Gain in 10-Cycle Stability (`80.1%` Avg vs `71.6%` in `TF-B13`):**
   - Across all 1,000 evaluation games vs `HeuristicPlayer`, `TF-B14` won **`801W / 196L / 3D` (`80.1%`)**—an **+8.5 percentage point increase** over `TF-B13` (`71.6%`), never dropping below `76.0%` in any cycle (`76%–90%` range vs `52%–89%` in `TF-B13`).
   - Against `RandomPlayer`, `TF-B14` went **`1,000W / 0L / 0D` (`100.0%`)**, reaching a record **`175.3` vs `2.6`** mean cluster score in Cycle 10 (`3,000` consecutive wins across `TF-B12`–`TF-B14`).
3. **Pure Network Self-Play Requires Zero Heuristic Teacher:**
   - Even with `0%` `HeuristicPlayer` games in training, `Dual-Side QuiescenceMCTS-50` (`riskPenalty = 0.25`, `enclosureBonus = 0.15`) retained and improved anti-Heuristic mastery purely from self-play.
4. **Confirming the 1-Ply `3×3` ConvNet Architectural Ceiling:**
   - The win/loss anatomy confirms that when the 1-ply ConvNet avoids blunders, it dominates `HeuristicPlayer` (`57.0%` of all games are `+50` point blowouts, plus `23.1%` close wins = `80.1%`). The remaining `~19.6%` losses consist of `~7.4%` close positional races and `~12.2%` sporadic 1-ply cascade blunders (where the `7×7` receptive field and zero-padded borders miss a 1-move hole over a 40-move game, $0.995^{40} \approx 82\%$). Breaking into `95%–99%` requires the Phase 3 upgrades (Inference-Time Safety Filter, Toroidal Circular Padding, and 3D Mesh/Graph Global Context).

---

#### 9. Experiment `TF-B15` Results: Phase 3 Full Torus ($T^2$) + Circular Convolutions + Tactical Safety Filter (`99.0%` Peak Raw / `95.9%` Avg Raw / **`100.0%` Across All 10 Cycles with Safety Filter**)

Following the official release of the 2D baseline (`v1.0.0-2d`), **Phase 3 (`TF-B15`)** unified the board's topology and resolved **Bottlenecks 2, 3, and 4** simultaneously. Let's visualize how **Toroidal Circular Padding (`ToroidalPad2D`)** and the **Inference-Time Tactical Safety Filter** work under the hood before examining the 10-cycle telemetry:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     TOROIDAL TOPOLOGY — Seamless edge connections on T²

![Toroidal Wrap-Around — Seamless edge connections on T²](./assets/diagrams/toroidal-wrap.svg)

WHY ZERO-PADDING BLINDED TF-B14 AT THE BORDER vs. TOROIDALPAD2D (TF-B15) │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Suppose an enclosure cavity straddles the top/bottom seam (y = 0 ↔ y = 9): │
│                                                                              │
│  1. Standard `padding: 'same'` (Zero-Padding in TF-B5 through TF-B14):       │
│     ┌───────────────────────────┐                                            │
│     │  0   0   0   0   0   0    │  ◄── Out-of-bounds row y = -1 filled       │
│     ├───────────────────────────┤      with ZEROS! The 3×3 filter at y = 0   │
│     │ .   .  [▲] [?] [▼]  .    │      cannot see the enclosing pieces at    │
│     │ ...     (Interior)    ... │      row y = 9! (~19% border blunder rate) │
│     │ .   .  [◀] [█] [▶]  .    │                                            │
│     ├───────────────────────────┤                                            │
│     │  0   0   0   0   0   0    │  ◄── Out-of-bounds row y = 10 filled       │
│     └───────────────────────────┘      with ZEROS!                           │
│                                                                              │
│  2. `ToroidalPad2D` (`[B, 10, 10, C] ──► [B, 12, 12, C]` in TF-B15):         │
│     ┌───────────────────────────┐                                            │
│     │ .   .  [◀] [█] [▶]  .    │  ◄── Top pad copies Row y = 9!             │
│     ├───────────────────────────┤                                            │
│     │ .   .  [▲] [?] [▼]  .    │  ◄── Row y = 0 now sees its true           │
│     │ ...     (Interior)    ... │      toroidal neighbors at y = 9 and       │
│     │ .   .  [◀] [█] [▶]  .    │      x = 9! Followed by `padding: 'valid'` │
│     ├───────────────────────────┤      3×3 Conv2D ──► exact `[B, 10, 10, C]` │
│     │ .   .  [▲] [?] [▼]  .    │  ◄── Bottom pad copies Row y = 0!          │
│     └───────────────────────────┘                                            │
│                                                                              │
│     Mathematical Invariant Verified by Unit Test:                            │
│     Conv(Shift_{(dx,dy)}(X)) == Shift_{(dx,dy)}(Conv(X)) (diff = 5.96e-8)    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│     HOW THE O(1) INFERENCE-TIME TACTICAL SAFETY FILTER CLOSES THE LAST 1%    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  During fast 1-ply play (~0.8 ms/move), the raw Policy Head P_θ(a|s) already │
│  wins 99.0% of games (Cycle 7 & 10). Why does it ever lose that 1% game?     │
│  Because over a 40-move game, a 99.95%-accurate softmax distribution can     │
│  occasionally rank a subtle 1-move cavity blunder at #1 (p = 0.34) just      │
│  ahead of a completely safe territorial move at #2 (p = 0.31).               │
│                                                                              │
│  The Top-K Safety Filter (`topK = 12`, `safetyFilter = true`):               │
│  ─────────────────────────────────────────────────────────────               │
│  1. Extract the top K = 12 legal moves ranked by neural probability P_θ(a|s).│
│  2. For each candidate move a_k (k = 1..12):                                 │
│     • Check immediate AI enclosure claims (`claimed_AI`) in the 5-cell       │
│       orthogonal neighborhood (`countWouldBeEnclosed` — O(1) in-place!).     │
│     • If `claimed_AI > 0`, the AI retains the turn (`extraTurns >= 1`),      │
│       so opponent reply risk is ZERO (`oppRisk = 0`).                        │
│     • If `claimed_AI === 0` (turn passes to opponent), scan the 5×5 local    │
│       toroidal neighborhood for any immediate opponent enclosure replies     │
│       (`oppRisk = max opponent claims on next turn`).                        │
│  3. Score each top-K candidate:                                              │
│       S(a_k) = log P_θ(a_k|s) + β_enc · claimed_AI - β_risk · oppRisk        │
│  4. Result: If Move #1 leaves an unguarded cavity (`oppRisk > 0`), the       │
│     filter seamlessly promotes Move #2 (`oppRisk = 0`), turning a 99.0%      │
│     raw network into a 1,000W / 0L / 0D (100.0%) undefeated champion!        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

In summary, **Phase 3 (`TF-B15`)** combined three synergistic mechanisms:
1. **Full Toroidal Edge Symmetry ($T^2$):** Wrapped vertex coordinates modulo `(cols, rows)` in `hasCorner(x, y)` (`toroidalPlacement = true`), giving the $10 \times 10$ grid exactly $100$ unique topological vertices (each shared by $12$ triangle slots) so both corner placement and enclosure detection are 100% translation-invariant.
2. **Toroidal Circular Convolutions (`ToroidalPad2D` / `ToroidalPad3D`):** Replaced standard zero-padding (`padding: 'same'`) across all three `3×3` `Conv2D` layers with custom circular padding that wraps 1 row/column around all four borders (`10×10 → 12×12`) prior to `valid` convolution, achieving exact 2D periodic translation equivariance (`max abs diff = 5.96e-8`).
3. **Inference-Time Tactical Safety Filter (`topK = 12`):** Integrated `ConvNetwork.evaluateOpponentReplyRisk` (`5×5` toroidal neighborhood scan for 1-ply hole blunders + immediate AI enclosure bonuses) directly into `predictMove()`.

Warm-started from `TF-B14_best_network.json`, **`TF-B15`** ran 10 cycles of 100% pure network self-play (`Dual-Side QuiescenceMCTS-50`, Cosine LR `0.0003 → 0.00008`, `45,000`-sample sliding replay window, runtime `585.2 min`, `129,628` total samples generated):

| Cycle | Cosine LR | Buffer (`45K` Window, `New Samples`) | Loss (`Start → End`) | Self-Play (`P1 / P2 / D`) | Eval vs Random (`100 games`, `Mean AI–Rand`) | Eval vs Heuristic **Raw Policy** (`Role P1 / P2`) | Raw Mean Score (`AI – Heur`, `Margin`) | Eval vs Heuristic **+Safety Filter** (`Mean AI–Heur`) | Diversity (`Open / Board`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (`TF-B14` Best)** | `0.000109` | `45,000 / 45K` | `3.916 → 3.081` | — | **100.0%** (`174.6 – 2.9`) | `87.0%` on $T^2$ | `131.2 – 48.4` (`+82.8`) | **`99.0%`** (`173.8 – 15.1`) | `4.43 / 1.87` |
| **1** | `0.000300` | `12,976 / 45K` (`12,976`) | `4.007 → 2.992` | `55 / 44 / 1` | **100.0%** (`175.8 – 2.8`) | **`87.0%`** (`84% / 90%`) | `133.9 – 46.5` (`+87.4`) | **🛡️ `100.0%`** (`174.8 – 14.3`) | `4.41 / 1.89` |
| **2** | `0.000293` | `25,975 / 45K` (`12,999`) | `3.856 → 2.912` | `50 / 49 / 1` | **100.0%** (`177.7 – 2.5`) | **`93.0%`** (`92% / 94%`) | `144.8 – 39.1` (`+105.7`) | **🛡️ `100.0%`** (`176.2 – 13.8`) | `4.12 / 1.88` |
| **3** | `0.000274` | `38,983 / 45K` (`13,008`) | `3.759 → 2.923` | `50 / 50 / 0` | **100.0%** (`175.5 – 2.7`) | **`95.0%`** (`92% / 98%`) | `146.8 – 35.3` (`+111.5`) | **🛡️ `100.0%`** (`175.4 – 13.6`) | `4.61 / 1.90` |
| **4** | `0.000245` | **`45,000 / 45K`** (`12,868`) | `3.716 → 2.916` | `55 / 44 / 1` | **100.0%** (`177.6 – 2.5`) | **`96.0%`** (`96% / 96%`) | `149.7 – 32.2` (`+117.5`) | **🛡️ `100.0%`** (`175.1 – 14.2`) | `4.40 / 1.89` |
| **5** | `0.000209` | **`45,000 / 45K`** (`13,028`) | `3.906 → 2.962` | `48 / 52 / 0` | **100.0%** (`177.4 – 2.5`) | **`98.0%`** (`98% / 98%`) | `156.7 – 29.2` (`+127.5`) | **🛡️ `100.0%`** (`173.4 – 14.4`) | `4.69 / 1.90` |
| **6** | `0.000171` | **`45,000 / 45K`** (`13,028`) | `3.720 → 2.981` | `47 / 53 / 0` | **100.0%** (`178.1 – 2.4`) | **`98.0%`** (`100% / 96%`) | `158.7 – 25.9` (`+132.8`) | **🛡️ `100.0%`** (`173.0 – 14.8`) | `4.47 / 1.90` |
| **7** | `0.000135` | `13,002 / 45K` (`13,002`) | `3.822 → 3.016` | `57 / 43 / 0` | **100.0%** (`177.7 – 2.5`) | **🏆 `99.0%`** (`98% / 100%`) | `157.3 – 26.0` (`+131.3`) | **🛡️ `100.0%`** (`174.8 – 14.8`) | `4.22 / 1.90` |
| **8** | `0.000106` | `25,974 / 45K` (`12,972`) | `3.699 → 3.010` | `53 / 47 / 0` | **100.0%** (`177.5 – 2.6`) | **`97.0%`** (`98% / 96%`) | `155.5 – 28.6` (`+126.9`) | **🛡️ `100.0%`** (`173.1 – 16.0`) | `4.41 / 1.89` |
| **9** | `0.000087` | `38,862 / 45K` (`12,888`) | `3.679 → 3.073` | `45 / 53 / 2` | **100.0%** (`178.4 – 2.4`) | **`97.0%`** (`98% / 96%`) | **`159.8 – 24.3` (`+135.5`)** | **🛡️ `100.0%`** (`173.5 – 14.9`) | `4.66 / 1.89` |
| **10** | `0.000080` | **`45,000 / 45K`** (`12,859`) | `3.754 → 3.106` | `59 / 41 / 0` | **100.0%** (`178.0 – 2.4`) | **🏆 `99.0%`** (`100% / 98%`) | `154.6 – 28.2` (`+126.4`) | **🛡️ `100.0%`** (`173.0 – 15.7`) | `4.18 / 1.88` |

![TF-B15 Dashboard — Telemetry for the 100% Champion Run](./assets/triantica-dashboard-tf-b15.png)

**Key Takeaways from `TF-B15` (Official Phase 3 Toroidal Champion):**
1. **Toroidal Circular Padding (`ToroidalPad2D`) Eliminated Border Blind Spots (`95.9%` Avg Raw Policy vs `80.1%` in `TF-B14`):**
   - Even **without** the inference safety filter (`safetyFilter = false`), replacing zero-padding with circular toroidal padding lifted the raw 1-ply ConvNet's 10-cycle average against `HeuristicPlayer` by **+15.8 percentage points** (`80.1% → 95.9%`, `959W / 40L / 1D`), peaking at **`99.0%` (`99W / 1L / 0D`)** in Cycles 7 and 10 and averaging **`97.7%` across Cycles 4–10**.
   - Mean raw cluster score margin more than doubled from `+56.4` in `TF-B14` to **`+135.5` (`159.8` vs `24.3`)** in Cycle 9, proving that a large fraction of `TF-B14`'s residual blunders occurred near the `x, y ∈ {0, 9}` borders where zero-padding previously severed wrap-around enclosures.
2. **Flawless Mastery with the Inference-Time Safety Filter (`1,000W / 0L / 0D` = `100.0%` Across All 10 Cycles):**
   - With `safetyFilter = true` (`topK = 12`), `TF-B15` went **`100.0%` (`100W / 0L / 0D`, `100% as P1`, `100% as P2`) in every single cycle from Cycle 1 to Cycle 10** (`1,000` consecutive wins vs `HeuristicPlayer`, mean score **`174.2` vs `14.7`**) and **`1,000W / 0L / 0D` (`100.0%`, `177.4` vs `2.5`)** vs `RandomPlayer`.
3. **Production Deployment & Multi-Tier Human Play (`trainedAI_network.json`):**
   - `TF-B15_best_network.json` (`Cycle 7` checkpoint: `99.0%` Raw / `100.0%` +Safety vs Heuristic) has been promoted to `trainedAI_network.json`, powering three selectable difficulty tiers in browser/mobile play: **Easy** (high-temperature sampling), **Medium** (Raw 1-ply ConvNet `99%`), and **Hard** (Master ConvNet + Tactical Safety Filter `100%`), alongside **`HumanGameLogger`** telemetry to record canonical `(state, move, claimed, outcome)` trajectories from real human matches.

---

### Phase 6: Self-Play Tournament + Iterative Retraining

**Priority:** Lower (polish after core improvements)

**Problem:** Network may overfit to its own play style.

**Solution:** Maintain pool of historical versions. Train against diverse opponents.

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Tournament Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Self-play: Current net vs current net                 │
│      → Generate games, add to replay buffer                  │
│                                                             │
│   2. Evaluation: Current net vs past versions                │
│      → If win rate < 55%, demote current                     │
│      → If win rate > 70%, promote candidate                  │
│                                                             │
│   3. Diversity injection: Add random/heuristic games        │
│      → Prevent style collapse                               │
│                                                             │
│   4. Retrain: Sample from buffer, train batch               │
│      → Repeat N times                                       │
│                                                             │
│   Benefit: Robust AI that handles varied opponents          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 6.1 | Implement `ModelPool` class to manage network versions | 1 hour |
| 6.2 | Implement tournament evaluation loop | 45 min |
| 6.3 | Add candidate promotion logic | 30 min |
| 6.4 | Add diversity injection (5-10% random games) | 20 min |
| 6.5 | Run experiment TF-B10 (tournament, 20 cycles) | 2 hours |
| 6.6 | Analyze: win rate vs past versions | 30 min |
| 6.7 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Current network beats all past versions (55%+)
- [ ] vs Random 80%+ (from Phase 5's 75%)
- [ ] vs Heuristic 15%+ (from Phase 5's 10%)
- [ ] No catastrophic forgetting (past skills retained)

**System Component:** Automated Peak Checkpoint Promotion & Tournament Harness (`*_best_network.json` auto-promotion in `src/server/train_tfjs.cjs` and cross-checkpoint tournament mode in `src/server/evaluate.cjs`)

---

### Progress Tracking

Use this table to track implementation progress. Update after each experiment.

| Phase | Experiment | Status | vs Random | vs Heuristic | Notes |
|-------|------------|--------|-----------|--------------|-------|
| Baseline | TF-B1_pure | ✅ Done | 49% | 0% | Pure self-play |
| Baseline | TF-B2_mixed | ✅ Done | 44% | 0% | Mixed opponents |
| Baseline | TF-B3_cluster | ✅ Done | 52% | 0% | TD(λ) rewards |
| 1: Replay | TF-B4 | ❌ No gain | 45% | 0% | Replay alone doesn't help; architecture limit |
| 2: Conv+Slot+Val | TF-B5 | ✅ Done | 50% | 0% | Canonical 18ch + Conv + Value head |
| 3: Actor-Critic | TF-B6 | ✅ Done | 50% | 0% | Optimized hyperparams (LR=0.02, value weight=1.0) |
| 4: BatchNorm+Conv | TF-B7 | ✅ Merged | — | — | BatchNorm integrated directly into Phase 2 ConvNet (`ConvNetwork.cjs`) |
| 5: MCTS-50 | TF-B8 | ❌ Failed | 30% | 0% | Inverted value perspective bug: network learned to lose (see Bug 5) |
| 5: MCTS-50 | TF-B9 | ✅ Done | 79% (84% peak) | 0% | Single-perspective MCTS + 400× policy loss fix; +41% jump vs Random |
| 5: MCTS-50 Mixed | TF-B10 | ✅ Done | 87% | 0% | 50% Network / 50% Heuristic; record 87% vs Random |
| 5.5: Quiescence MCTS | TF-B12 | ✅ Done | **100%** | **63%** | **Heuristic Wall Broken!** 100% vs Random (all 10 cycles), 63% vs Heuristic (66% P1, 60% P2; random 2-ply eval C8–C10) |
| 5.6: Dual-Side Quiescence | TF-B13 | ✅ Done | **100%** | **73% (89% peak)** | **New Record vs Heuristic!** 89% peak (C8: 90% P1, 88% P2), 73% final (C10: 80% P1, 66% P2), 71.6% 10-cycle avg; 115.9K dual-side MCTS samples |
| 5.7: Sliding-Window Pure SP | TF-B14 | ✅ Done | **100%** | **78% (90% peak, 80.1% avg)** | **Official 2D Release Champion (`v1.0.0-2d`)!** 90% peak (C9: 88% P1, 92% P2, score `123.8–54.2`), 80.1% 10-cycle avg (`76%–90%` band) under 100% pure network self-play + 45K sliding buffer + Cosine LR. See [`docs/EXECUTION_ROADMAP.md`](./EXECUTION_ROADMAP.md). |
| 6.0: Phase 3 Full Torus ($T^2$) | TF-B15 | ✅ Done | **100%** | **99% Raw (95.9% avg) / 100% +Safety** | **All-Time Record!** Full $T^2$ toroidal placement + `ToroidalPad2D` circular convolutions + Tactical Safety Filter (`topK=12`). `1,000W/0L/0D` (`100.0%`) with safety filter; `99.0%` peak (`95.9%` avg) raw policy. Promoted to `trainedAI_network.json`. |

**Status Legend:**
- ✅ Done — Experiment complete, documented
- 🔄 Running — Currently executing
- ⏳ Pending — Not started
- ❌ Failed — Did not meet success criteria (requires revision)

---

### Summary: Expected vs. Achieved Impact Across All Phases

| Phase | Architectural Upgrade | Expected vs Random | Expected vs Heuristic | **Actual Achieved (`TF-B4` → `TF-B15`)** |
|-------|-----------------------|--------------------|-----------------------|------------------------------------------|
| 1 | Experience Replay Buffer (`TF-B4`) | 45% | 0% | **45% vs Random, 0% vs Heuristic** (Feedforward bottleneck) |
| 2–3 | 18-Channel Canonical Slot Encoding + Conv2D + Value Head (`TF-B5`/`B6`) | 50% | 0% | **50% vs Random, 0% vs Heuristic** (Spatial features learned, but no lookahead) |
| 5 | Single-Perspective MCTS-50 + Bug 6–11 Fixes (`TF-B9`/`B10`) | 75% | 5–15% | **87% vs Random, 0% vs Heuristic** (Vanilla MCTS blind to deterministic cascades) |
| 5.5 | Quiescence MCTS + 50/50 Mixed Opponents (`TF-B12`) | 85% | 15–20% | **100% vs Random, 63% vs Heuristic** (🏆 Heuristic Wall Broken!) |
| 5.6 | Dual-Side Quiescence MCTS + 75/25 Mix (`TF-B13`) | 100% | 70%+ | **100% vs Random, 89% Peak (71.6% Avg) vs Heuristic** |
| 5.7 | 100% Pure Self-Play + 45K Sliding Replay + Cosine LR (`TF-B14`) | 100% | 80%+ | **100% vs Random, 90% Peak (80.1% Avg) vs Heuristic** (`v1.0.0-2d` Champion) |
| 6.0 | Full $T^2$ Torus + Circular Conv (`ToroidalPad2D`) + Safety Filter (`TF-B15`) | 100% | 95%+ | **100% vs Random, 99.0% Peak (95.9% Avg) Raw / 100.0% (`1,000W/0L`) +Safety** |

---

---

> [← 11D: Monte Carlo Tree Search (MCTS)](./11d-mcts-theory-and-implementation.md) | **[Ch 11 Index](./11-deep-dive-convolutions-mcts.md)** | **[📚 Training Book Index](./README.md)** | [Chapter 12: Toroidal 3D Visualization & Human Telemetry →](./12-toroidal-3d-and-telemetry.md)
