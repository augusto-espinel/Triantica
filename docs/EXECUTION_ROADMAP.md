# Triantica Execution Roadmap: Official 2D Variant Release & 3D Manifold Transition

**Last Updated:** 2026-10-29  
**Completed Experiment:** `TF-B15` (`99.0%` Peak / `95.9%` 10-Cycle Avg Raw Policy, **`100.0%` Across All 10 Cycles with Safety Filter** vs Heuristic, `100.0%` vs Random)  
**Milestone 1 Status:** ✅ Tagged & Published (`v1.0.0-2d` on `2d-variant` branch)  
**Milestone 2 Status:** ✅ Phase 3 Completed on `main` (`TF-B15` Champion Deployed + Mobile UI & Difficulty Tiers Ready)  
**Milestone 3 Status:** 💡 Phase 3.5 / Phase 4 Concept & Visual Architecture Exploration (`docs/TOROIDAL_3D_VISUALIZATION_CONCEPT.md`)

---

## 1. Strategic Overview & Phasing

Following the breakthrough of `QuiescenceMCTS-50` in `TF-B12` (`63%` vs Heuristic) and `TF-B13` (**`89.0%` peak / `73.0%` final / `71.6%` 10-cycle average vs Heuristic**, `100.0%` vs Random), our forensic audit revealed two foundational truths:

1. **The Official 2D Hybrid Topology:**  
   The current 2D engine (`src/core/GridManager.js` and `game.js`) naturally implements an **Asymmetrical Hybrid Edge Topology**:
   - **Corner Placement Legality (`isValidPlacement` $\to$ `hasCorner(x, y)`): Bounded / Planar.** Vertex coordinates `cx, cy ∈ [0..10]` do not wrap around the outer border. Players cannot branch new placements across the border via corner contact alone.
   - **Enclosure Detection (`isTriangleEnclosed`) & Cluster Scoring (`getAdjacentTriangles`): Toroidal (Wrapping).** Orthogonal edge neighbors wrap modulo `(cols, rows)`. An enclosure claim wrapping across the border can establish a seed triangle on the opposite edge from which subsequent planar corner placements can branch.
   - Because placement has a hard planar boundary at `x ∈ {0, 9}` and `y ∈ {0, 9}`, standard `Conv2D` zero-padding (`padding: 'same'`) acts as a natural spatial anchor for the 2D board.

2. **The #1 Training Bottleneck (Replay Buffer Staleness):**  
   `TF-B13` peaked at `89.0%` in Cycle 8 before dipping to `78.0%` (C9) and `73.0%` (C10) because the `500,000`-capacity replay buffer never evicted early samples (`115,892` accumulated), causing ~70% of late-cycle training batches to sample stale trajectories from Cycles 1–6 at a constant learning rate (`0.0005`).

To maintain rigorous scientific discipline, execution is divided into **four sequential phases**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     TRIANTICA EXECUTION ROADMAP                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Bottleneck 1 Fix & Pure Self-Play Scientific Run (TF-B14)  [DONE]  │
│    • 45,000-sample sliding-window ReplayBuffer (~3.2 cycles window)          │
│    • Cosine LR decay (0.0004 → 0.0001) with in-place Adam moment retention   │
│    • Automatic peak checkpoint retention (*_best_network.json)               │
│    • 100% Pure Network Self-Play: 90.0% Peak / 80.1% Avg vs Heuristic        │
│                                      │                                       │
│                                      ▼                                       │
│  PHASE 2: Official 2D Variant Release, Dedicated Branch & Tag        [DONE]  │
│    • Completed TF-B14 (10 cycles) & published full metrics to docs/dashboard │
│    • Promoted TF-B14_best_network.json (90% vs Heur) to trainedAI_network    │
│    • Committed, tagged (v1.0.0-2d), and published `2d-variant` branch        │
│    • Ready to continue 3D-compatible development on `main`                   │
│                                      │                                       │
│                                      ▼                                       │
│  PHASE 3: 3D Capability, Full Edge Symmetry & Bottlenecks 2 & 3      [DONE]  │
│    • Unify toroidal edge symmetry: wrap hasCorner(x, y) modulo (cols, rows)  │
│    • Bottleneck 2: Toroidal Circular Padding in TFConvNetwork / ConvNetwork  │
│    • Bottleneck 3: Inference-Time Tactical Safety Filter (predictMove)       │
│    • TF-B15: 99.0% Peak / 95.9% Avg Raw Policy, 100.0% (+Safety) vs Heur!    │
│    • Pre-3D Mobile UI Revamp + Easy/Medium/Hard AI Tiers + P1/P2 Side Select │
│                                      │                                       │
│                                      ▼                                       │
│  PHASE 3.5: Playable Toroidal Visualization & Dual-View Concept   [EXPLORE]  │
│    • Isometric Infinite-Plane / Ghost-Border Tile View (Primary Play Surface)│
│    • Synchronized 3D Closed Surface / Hinged Folding Companion View          │
│    • Concept & UX Blueprint: docs/TOROIDAL_3D_VISUALIZATION_CONCEPT.md       │
│                                      │                                       │
│                                      ▼                                       │
│  PHASE 4: Manifold Symmetry Exploitation (Bottleneck 4)          [BRAINSTORM]│
│    • Explore dihedral (D_4) / toroidal / mesh gauge symmetry augmentation    │
│      once visual dynamics & 3D manifold gameplay feel are finalized          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase-by-Phase Technical Specification

### Phase 1: Fix Bottleneck 1 & Test Pure Network Self-Play Ceiling (`TF-B14`) — ✅ COMPLETED (`90.0%` Peak / `80.1%` Avg vs Heuristic)

**Scientific Objective:**  
Isolate the impact of fixing **Bottleneck 1 (Replay Buffer Staleness + LR Decay)** and determine how far **100% pure network self-play** (`Dual-Side QuiescenceMCTS-50`, 0% Heuristic teacher) can push the current 2D ConvNet architecture under the asymmetrical 2D ruleset.

**Final Results (`TF-B14`, 10 Cycles, `397.2 min`):**
- **New Record Peak vs Heuristic:** **`90.0%` (`90W / 9L / 1D`, `88.0%` as P1, `92.0%` as P2, mean cluster `123.8` vs `54.2`, margin `+69.7`)** in Cycle 9 (`LR = 0.000109`), automatically captured in `dashboard/data/TF-B14_best_network.json` (`MD5: f2eb302a23dc4a38397a7ed34bd27b2a`) and promoted to `trainedAI_network.json`.
- **10-Cycle Stability:** **`80.1%` 10-cycle average (`801W / 196L / 3D`, range `76%–90%`)** vs `71.6%` (`52%–89%` range) in `TF-B13` (+8.5 percentage points).
- **Eval vs Random:** **`100.0%` (`1,000W / 0L / 0D`)** across all 10 cycles (`175.3` vs `2.6` in Cycle 10).

**Codebase Changes Implemented (`src/server/ReplayBuffer.cjs`, `src/server/train_tfjs.cjs`):**
1. **Sliding-Window Replay Buffer (`--replay-buffer-size 45000`):**
   - Capped default `ReplayBuffer` capacity at `45,000` samples (down from `500,000`).
   - With Dual-Side MCTS generating ~13,500–14,500 canonical samples per 100-game pure self-play cycle, the buffer reaches 100% capacity in Cycle 4 and overwrites the oldest samples via circular pointer `this.position = (this.position + 1) % this.maxSize`, bounding replay memory to the most recent **~3.2 cycles**.
2. **Cosine Learning Rate Schedule (`--lr-schedule cosine`, `--min-learning-rate 0.0001`):**
   - Decays learning rate smoothly across cycles $c \in \{1, \dots, C\}$:
     $$\eta_c = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{c - 1}{C - 1}\pi\right)\right)$$
   - `TFConvNetwork.trainBatchWithIsolation()` updates `this._adamOptimizer.learningRate` in-place rather than re-instantiating `tf.train.adam()`, preserving Adam's first and second moment accumulators ($m_t, v_t$) across cycles.
3. **Automatic Peak Checkpointing (`*_best_network.json`):**
   - Tracks the highest `Eval vs Heuristic` win rate (using mean AI-minus-Heuristic cluster score margin as tie-breaker) and automatically saves `dashboard/data/TF-B14_best_network.json` alongside `TF-B14_network.json`.

**Active Experiment (`TF-B14`) Command:**
```bash
node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 --iterations 10 \
  --learning-rate 0.0004 --lr-schedule cosine --min-learning-rate 0.0001 \
  --batch-size 256 --replay-buffer-size 45000 --replay-batch-size 256 --replay-ratio 2.0 \
  --self-play-mode network --opponent-mix network \
  --architecture conv --mcts-sims 50 --mcts-batch-size 8 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.25 \
  --start-from dashboard/data/TF-B13_network.json \
  --output dashboard/data/TF-B14.json
```

---

### Phase 2: Official 2D Variant Release, Dedicated Branch (`2d-variant`) & Tagging (`v1.0.0-2d`) — ✅ COMPLETED

**Objective:**  
Freeze, tag, and preserve the definitive 2D version of Triantica (using the Bounded Placement / Toroidal Enclosure hybrid topology) on its own dedicated GitHub branch (`2d-variant`) once `TF-B14` completes, while continuing 3D-compatible development on `main`. This allows GitHub Pages to be switched between `2d-variant` and `main` at any time for direct gameplay and performance comparison between the 2D and 3D variants.

**Release Checklist:**
- [x] **1. Audit `TF-B14` 10-Cycle Results:** Compare `TF-B14` peak (`90.0%` C9) and 10-cycle average (`80.1%`) against `TF-B13` (`89.0%` peak / `71.6%` avg vs Heuristic).
- [x] **2. Promote Champion 2D Weights:** Copy `dashboard/data/TF-B14_best_network.json` (`90.0%` vs Heuristic, `MD5: f2eb302a23dc4a38397a7ed34bd27b2a`) to `trainedAI_network.json` in the repository root for browser play.
- [x] **3. Finalize 2D Documentation & Dashboard:**
  - Update `dashboard/experiments.json` with complete 10-cycle `TF-B14` metrics.
  - Document `TF-B14` findings in `docs/TRAINING_BOOK.md` and `docs/QUIESCENCE_MCTS.md`.
- [x] **4. Commit, Tag & Publish Dedicated `2d-variant` Branch:**
  - Commit all 2D release artifacts to `main` and push to `origin/main`.
  - Create and push a dedicated branch **`2d-variant`** (`git checkout -b 2d-variant && git push -u origin 2d-variant`) alongside annotated Git tag **`v1.0.0-2d`**.
  - Switch back to `main` (`git checkout main`) for all subsequent 3D manifold and full-symmetry development, keeping `2d-variant` permanently ready as a GitHub Pages deployment target whenever 2D vs. 3D comparison is needed.

---

### Phase 3: 3D Capability, Full Edge Symmetry & Bottlenecks 2 & 3 — ✅ COMPLETED

**Objective:**  
Transition Triantica from a bounded 2D planar-placement board to a **seam-free 3D rotatable manifold** where edges have full topological symmetry, while resolving **Bottleneck 2** (Toroidal Circular Padding) and **Bottleneck 3** (Inference-Time Safety Filter).

#### Step 3.1: Full Edge Symmetry on the Torus ($T^2$) — ✅ DONE
- Upgraded `hasCorner(x, y)` in `src/core/GridManager.js` and `game.js` to default to `toroidalPlacement = true` on `main` (while preserving `toroidalPlacement: false` compatibility for the `2d-variant` hybrid topology):
  - Vertex coordinates wrap modulo `(this.cols, this.rows)` (`100` unique mesh vertices on a `10×10` torus instead of `121`, each incident to `12` triangle face-slots).
  - Both **placement legality** (corner sharing) and **enclosure / cluster scoring** (edge sharing) are now 100% translation-invariant across $T^2$.

#### Step 3.2: Bottleneck 2 — Toroidal Circular Padding in `TFConvNetwork` & `ConvNetwork.cjs` — ✅ DONE
- Implemented custom TensorFlow.js layers `ToroidalPad2D` and `ToroidalPad3D` in `src/server/train_tfjs.cjs` (using differentiable `tf.slice` and `tf.concat` inside `tf.tidy()` without deprecated `util.isNullOrUndefined` calls, and registered via `tf.serialization.registerClass`), plus matching circular convolution in `src/ai/ConvNetwork.cjs` and `AIPlayer.js` (`BrowserConvNetwork`).
- Verified exact numerical equivalence between `TFConvNetwork` and pure-JS `ConvNetwork` (`max abs policy diff = 4.28e-7`, `value diff = 0.0`) and exact 2D periodic translation equivariance (`max abs diff = 5.96e-8`).

#### Step 3.3: Bottleneck 3 — Inference-Time Tactical Safety Filter (`ConvNetwork.predictMove`) — ✅ DONE
- Integrated `ConvNetwork.countWouldBeEnclosed` and `ConvNetwork.evaluateOpponentReplyRisk` (checking the local $5\times 5$ toroidal neighborhood for 1-ply opponent enclosure traps and rewarding immediate AI enclosures) into `src/ai/ConvNetwork.cjs`, `src/players/AIPlayer.js`, and browser `AIPlayer.js`.
- **Empirical Verification (`500` random-opening games vs `HeuristicPlayer`):**
  - **Raw Policy (`safetyFilter = false`):** `87.0%` (`87W / 13L / 0D` per 100 games)
  - **With Tactical Safety Filter (`safetyFilter = true`):** **`99.0%` (`495W / 5L / 0D` across 500 games; `100.0%` on 100 deterministic openings)**, eliminating virtually all 1-ply tactical blunders at ~0.35s/game without search overhead.

#### Step 3.4: 3D Manifold Engine & Mesh Representation ($\mathcal{M}$) — ✅ DONE
- Generalized `GridManager` (`src/core/GridManager.js`) with `buildToroidalManifoldGraph(rows, cols, toroidalPlacement)` and `buildGeneralMeshManifoldGraph(meshFaces, numVertices)` to represent any closed 3D triangulated manifold $\mathcal{M}$ via a **Dual Face-Adjacency Graph** $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$:
  - **Nodes $v \in V_T$:** Triangle face-slots on the 3D mesh (`400` on $T^2$, or $|F|$ on any general 3D triangle mesh such as an octahedron or subdivided icosphere).
  - **Edge-Adjacency $\mathcal{N}_{\text{edge}}(v)$ (3 boundary groups per triangle):** Governs `isManifoldFaceEnclosed` and `getManifoldAdjacentFaces`.
  - **Vertex-Adjacency $\mathcal{V}(v)$ (3 mesh vertices per triangle):** Governs `isManifoldFaceValidPlacement` via `vertexToFaces` incidence lists.

---

#### Step 3.5: Experiment `TF-B15` & Pre-3D Mobile UI Revamp — ✅ DONE
- **Completed `TF-B15` (`10` cycles, `585.2 min`, warm-started from `TF-B14_best_network.json`):**
  - **vs Random (Raw Policy):** **`100.0%`** (`1,000W / 0L / 0D` across all 10 cycles, mean score `178.0` vs `2.4`).
  - **vs Heuristic (Raw 1-Ply Policy):** **`99.0%` Peak** (Cycles 7 & 10) and **`95.9%` 10-Cycle Average** (`959W / 40L / 1D`, up `+15.8%` from `TF-B14`'s `80.1%` average).
  - **vs Heuristic (+Safety Filter, `topK = 12`):** **`100.0%` across all 10 cycles (`1,000W / 0L / 0D`)**, mean cluster score `~174.2` vs `14.7`.
  - Promoted `TF-B15_best_network.json` to `trainedAI_network.json`.
- **Pre-3D Mobile UI & Multi-Tier Difficulty Revamp (`game.js`, `index.html`, `AIPlayer.js`):**
  - Responsive `Phaser.Scale.FIT` + `CENTER_BOTH` viewport scaling for mobile & desktop playtesting.
  - 2-step **Human vs AI** menu flow: **Difficulty Selection** (`Easy` casual temperature sampling, `Medium` raw ConvNet `99%`, `Hard` Master ConvNet + Safety Filter `100%`) $\to$ **Side Selection** (`Play as P1 Red` or `Play as P2 Blue`).
  - Gated AI benchmark modes (`AvR`, `AvH`, `AvA`) behind `const DEBUG_MODE = false`.

---

### Phase 3.5 & Phase 4: Visual Concept Exploration & Symmetry Brainstorming — 💡 EXPLORATION PHASE

Before committing to low-level symmetry augmentation (`D_4` / gauge equivariance) or locking in a final 3D mesh ruleset, the immediate priority is **experiencing the toroidal game dynamics with friends** and developing a clear **visual concept** that makes wrap-around topology intuitive to human players.

See **[`docs/TOROIDAL_3D_VISUALIZATION_CONCEPT.md`](./TOROIDAL_3D_VISUALIZATION_CONCEPT.md)** for the full design exploration covering:
1. **Isometric Infinite Repeating Plane (Primary Play Surface):** Playing on the flattened surface with an angled/isometric camera and translucent repeating "ghost tiles" (or seamless panning) around the $10\times 10$ fundamental domain so cross-border enclosures and corner contacts are immediately visible.
2. **Synchronized 3D Closed-Surface Companion Viewport:** A secondary 3D preview showing how the flat grid folds along triangle/cell hinges (`u`, `v` folding parameters) into a closed 3D surface (torus or polyhedral manifold), updating live with every move.
3. **Phase 4 Brainstorming Notes:** How visual feedback and playtesting insights will guide future 3D manifold variants and symmetry exploitation.

---

## 3. Bottleneck Summary Reference Table

| Bottleneck ID | Technical Issue | Target Phase | Planned Resolution | Status |
|:---:|:---|:---:|:---|:---:|
| **Bottleneck 1** | Replay Buffer Staleness (`500K` capacity kept Cycle 1–6 noise) & Constant LR | **Phase 1** (`TF-B14`) | `45,000` sliding-window `ReplayBuffer` + Cosine LR (`0.0004 → 0.0001`) + in-place Adam moment preservation + `*_best_network.json` checkpointing | ✅ **Completed (`TF-B14`: `90.0%` Peak / `80.1%` Avg vs Heuristic)** |
| **Asymmetry (2D vs 3D)** | Planar `hasCorner` placement vs. Toroidal `isTriangleEnclosed` / `getAdjacentTriangles` | **Phase 2** (`2d-variant`) $\to$ **Phase 3** (`main`) | Official feature of `v1.0.0-2d` (`2d-variant` branch); unified with modulo vertex wrapping (`toroidalPlacement = true`) & manifold graph $G=(V_T,E_{\text{edge}},E_{\text{vertex}})$ on `main` | ✅ **Completed (`2d-variant` & `main`)** |
| **Bottleneck 2** | `Conv2D` Zero-Padding (`padding: 'same'`) splits wrap-around features at borders | **Phase 3** (3D Transition) | Replaced with `ToroidalPad2D` / `ToroidalPad3D` (`10×10 → 12×12` wrap before `3×3` `valid` conv) + pure-JS circular conv in `ConvNetwork.cjs` | ✅ **Completed (Exact Equivariance `5.96e-8`)** |
| **Bottleneck 3** | 1-Ply Raw Policy Inference lacks tactical hole-risk check at test time | **Phase 3** (3D Transition) | Added $O(1)$ `countWouldBeEnclosed` + $5\times 5$ `evaluateOpponentReplyRisk` filter to top-$K$ candidates in `ConvNetwork.predictMove()` | ✅ **Completed (`99.0%` vs Heuristic over 500 games)** |
| **Bottleneck 4** | Symmetry Exploitation ($D_4$ rotations/reflections & manifold symmetries) | **Phase 4** (Post-3D Polish) | 8× dihedral / toroidal / gauge symmetry augmentation after 3D training improvements are complete | ⏳ **Next (Phase 4)** |
