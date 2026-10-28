# Triantica Execution Roadmap: Official 2D Variant Release & 3D Manifold Transition

**Last Updated:** 2026-10-28  
**Completed Experiment:** `TF-B14` (`90.0%` Peak / `80.1%` 10-Cycle Avg vs Heuristic, `100.0%` vs Random)  
**Milestone 1 Status:** ✅ Ready to Tag & Publish (`v1.0.0-2d` on `2d-variant` branch)  
**Target Milestone 2:** 3D Toroidal & Arbitrary Manifold Generalization (`v2.0.0-3d` on `main`)

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
│  PHASE 3: 3D Capability, Full Edge Symmetry & Bottlenecks 2 & 3      [NEXT]  │
│    • Unify toroidal edge symmetry: wrap hasCorner(x, y) modulo (cols, rows)  │
│    • Bottleneck 2: Toroidal Circular Padding in TFConvNetwork / ConvNetwork  │
│    • Bottleneck 3: Inference-Time Tactical Safety Filter (predictMove)       │
│    • Generalize GridManager to Dual Face-Adjacency Mesh Graph G=(V_T,E_e,E_v)│
│                                      │                                       │
│                                      ▼                                       │
│  PHASE 4: Manifold Symmetry Exploitation (Bottleneck 4)              [FINAL] │
│    • Tackle dihedral (D_4) / toroidal / mesh gauge symmetry augmentation     │
│      after all 3D-compatible training & architectural upgrades are proven    │
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

### Phase 2: Official 2D Variant Release, Dedicated Branch (`2d-variant`) & Tagging (`v1.0.0-2d`) — ⏳ NEXT

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

### Phase 3: 3D Capability, Full Edge Symmetry & Bottlenecks 2 & 3 — ⏳ PLANNED (POST-2D RELEASE)

**Objective:**  
Transition Triantica from a bounded 2D planar-placement board to a **seam-free 3D rotatable manifold** where edges have full topological symmetry, while resolving **Bottleneck 2** (Toroidal Circular Padding) and **Bottleneck 3** (Inference-Time Safety Filter).

#### Step 3.1: Full Edge Symmetry on the Torus ($T^2$)
- Upgrade `hasCorner(x, y)` in `src/core/GridManager.js` (and `game.js`) to support `toroidalPlacement = true` by wrapping vertex coordinates modulo `(this.cols, this.rows)`:
  - Vertex `(cols, y) ≡ (0, y)` and `(x, rows) ≡ (x, 0)` (`100` unique vertices on a `10×10` torus instead of `121`).
  - Both **placement legality** (corner sharing) and **enclosure / cluster scoring** (edge sharing) become 100% translation-invariant across the torus, eliminating artificial board borders so the player can freely rotate the 3D torus to any angle.

#### Step 3.2: Bottleneck 2 — Toroidal Circular Padding in `TFConvNetwork` & `ConvNetwork.cjs`
- Replace `padding: 'same'` (zero-padding) in the `3×3` `Conv2D` layers with **Circular Toroidal Padding** (`pad = 1` wrapping row `0 ↔ rows-1` and col `0 ↔ cols-1` before `padding: 'valid'` `3×3` convolution) in both:
  - `src/server/train_tfjs.cjs` (`TFConvNetwork`)
  - `src/ai/ConvNetwork.cjs` (pure-JS inference engine used in browser & fast server eval)
- Guarantees exact translation equivariance: shifting the board on the 3D torus shifts the feature maps identically without boundary distortion.

#### Step 3.3: Bottleneck 3 — Inference-Time Tactical Safety Filter (`ConvNetwork.predictMove`)
- Integrate `QuiescenceMCTS`'s fast $O(1)$ local neighborhood risk check (`countWouldBeEnclosed`) into 1-ply inference (`src/ai/ConvNetwork.cjs` / `AIPlayer.js`).
- Evaluate the top-$K$ policy moves (`K = 5`): if a candidate move has `claimed === 0` (passes the turn) and creates an immediate opponent enclosure (`oppEnclosed > 0`), apply the risk penalty before final move selection, eliminating residual 1-ply tactical blunders at zero search cost.

#### Step 3.4: 3D Manifold Engine & Mesh Representation ($\mathcal{M}$)
- Generalize `GridManager` to support arbitrary closed 3D triangulated manifolds $\mathcal{M}$ via a **Dual Face-Adjacency Graph** $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$:
  - **Nodes $v \in V_T$:** Triangle slots on the 3D mesh.
  - **Edge-Adjacency $\mathcal{N}_{\text{edge}}(v)$ (3 neighbors per triangle):** Governs `isTriangleEnclosed` (all 3 edge-neighbors occupied) and `getLargestContiguous` (connected components over $E_{\text{edge}}$).
  - **Vertex-Adjacency $\mathcal{V}(v)$ (3 mesh vertices per triangle):** Governs `isValidPlacement` (at least one vertex in $\mathcal{V}(v)$ is touched by an existing triangle).
  - **`QuiescenceMCTS` Compatibility:** `QuiescenceMCTS.cjs` is already 100% manifold-agnostic and operates unchanged on any 3D mesh.

---

### Phase 4: Symmetry Exploitation (Bottleneck 4) — ⏳ DEFERRED UNTIL POST-3D

**Objective:**  
Once the 3D-compatible architecture, full edge symmetry, circular/mesh convolutions, and inference-time safety filter have been thoroughly explored and benchmarked:
- Implement **Dihedral ($D_4$) / Toroidal Translation / Mesh Gauge Symmetry Augmentation** during replay buffer ingestion or canonical state encoding.
- Provides an **8× sample efficiency multiplier** on the square torus (and local gauge equivariance on general 3D triangle meshes) as the final polish stage.

---

## 3. Bottleneck Summary Reference Table

| Bottleneck ID | Technical Issue | Target Phase | Planned Resolution | Status |
|:---:|:---|:---:|:---|:---:|
| **Bottleneck 1** | Replay Buffer Staleness (`500K` capacity kept Cycle 1–6 noise) & Constant LR | **Phase 1** (`TF-B14`) | `45,000` sliding-window `ReplayBuffer` + Cosine LR (`0.0004 → 0.0001`) + in-place Adam moment preservation + `*_best_network.json` checkpointing | ✅ **Completed (`TF-B14`: `90.0%` Peak / `80.1%` Avg vs Heuristic)** |
| **Asymmetry (2D vs 3D)** | Planar `hasCorner` placement vs. Toroidal `isTriangleEnclosed` / `getAdjacentTriangles` | **Phase 2** (`2d-variant`) $\to$ **Phase 3** (`main`) | Official feature of `v1.0.0-2d` (`2d-variant` branch); unify with modulo vertex wrapping (`toroidalPlacement = true`) in `v2.0.0-3d` (`main`) | ✅ **2D Released (`2d-variant` / `v1.0.0-2d`); 3D Next** |
| **Bottleneck 2** | `Conv2D` Zero-Padding (`padding: 'same'`) splits wrap-around features at borders | **Phase 3** (3D Transition) | Replace with Toroidal Circular Padding (`10×10 → 12×12` wrap before `3×3` conv) / Mesh Graph Conv | ⏳ **Scheduled for Phase 3** |
| **Bottleneck 3** | 1-Ply Raw Policy Inference lacks tactical hole-risk check at test time | **Phase 3** (3D Transition) | Add $O(1)$ `countWouldBeEnclosed` safety filter to top-$K$ candidates in `ConvNetwork.predictMove()` | ⏳ **Scheduled for Phase 3** |
| **Bottleneck 4** | Symmetry Exploitation ($D_4$ rotations/reflections & manifold symmetries) | **Phase 4** (Post-3D Polish) | 8× dihedral / toroidal / gauge symmetry augmentation after 3D training improvements are complete | ⏳ **Deferred to Phase 4** |
