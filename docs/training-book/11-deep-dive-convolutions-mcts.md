> **[📚 Training Book Index](./README.md)** | [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | [11A: Replay Buffer & Canonical ConvNet →](./11a-replay-and-conv-encoding.md)

---

## 11. Deep Dive: Convolutions, Dual Heads, MCTS & Pipeline Consolidation {#11-optimization-roadmap}

This chapter walks through the step-by-step architectural roadmap that took Triantica from the feedforward baseline (`37–52%` vs Random, `0%` vs Heuristic) to **100.0% Full Torus ($T^2$) mastery (`TF-B15`)**. Because this chapter covers six major engineering phases with full mathematical derivations, tensor traces, and code walkthroughs, it is broken into five modular sub-files:

### Chapter 11 Sub-Files

| Part | File | Lines | Topics Covered |
|------|------|------:|----------------|
| **11A** | **[Phase 1 & Phase 2: Replay Buffer & Canonical ConvNet Encoding](./11a-replay-and-conv-encoding.md)** | ~270 | Experience Replay Buffer (`ReplayBuffer.cjs`), Canonical `10×10×18` 1-to-1 Slot Encoding, 3-Layer `Conv2D` receptive field growth, and Dual-Head Policy/Value wiring. |
| **11B** | **[Phase 2 Results & Milestone Runs (`TF-B5` → `TF-B13`)](./11b-milestone-runs-tf-b5-to-b13.md)** | ~375 | Step-by-step experimental results (`TF-B5`, `TF-B6`, `TF-B8`, `TF-B9`, `TF-B10`, `TF-B12`, `TF-B13`), the `400x` Policy Cross-Entropy gradient fix, and why Initial Error "sawtooths" across self-play cycles. |
| **11C** | **[Phase 3 & Phase 4: Value Head (Actor-Critic) & Batch Normalization](./11c-actor-critic-and-batchnorm.md)** | ~125 | Actor-Critic shared trunk design, Value MSE loss dynamics, and stabilizing deep convolutional training with `tf.layers.batchNormalization()`. |
| **11D** | **[Phase 5: Monte Carlo Tree Search (MCTS) Theory & Implementation](./11d-mcts-theory-and-implementation.md)** | ~500 | Complete PUCT MCTS derivation, 4-step search loop, M2 Ultra batched leaf evaluation, virtual loss, root Dirichlet exploration noise, and annotated `MCTS.cjs` walkthrough. |
| **11E** | **[Phase 5.5 & Phase 6: Quiescence MCTS, Full Torus ($T^2$) Mastery & Summary](./11e-quiescence-mcts-and-torus-mastery.md)** | ~405 | Horizon effect & deterministic enclosure cascades, `QuiescenceMCTS.cjs` mathematical formulation, `TF-B14` (45K FIFO sliding replay + pure self-play), `TF-B15` (`ToroidalPad2D` circular convolutions + Top-12 Tactical Safety Filter), and Phase 1–6 summary table. |

---

### Initial Pre-ConvNet Baseline vs. Final Achieved State

| Metric | Initial Dense Baseline (`TF-B1`–`B3`) | Original Roadmap Target | **Final Achieved (`TF-B15` Champion)** |
|--------|--------------------------------------|-------------------------|----------------------------------------|
| vs Random | 37–52% | 80%+ | **100.0% (`1,000W / 0L / 0D` across all 10 cycles)** |
| vs Heuristic | 0% | 10%+ | **99.0% Peak (`95.9%` Avg) Raw 1-Ply / 100.0% (`1,000W / 0L`) +Safety** |
| Training time (10 cycles) | ~40 min (Dense) | Fast M2 Ultra batching | **~140 ms / 50-sim MCTS move (Dual-State GPU Batching)** |
| Replay Buffer Strategy | None (1-cycle memory) | Circular Replay Buffer | **45,000-sample FIFO Sliding Window (`~3.5` cycles of Dual-Side MCTS)** |

---

> [← Chapter 10: Key Takeaways](./10-key-takeaways.md) | **[📚 Training Book Index](./README.md)** | [11A: Replay Buffer & Canonical ConvNet →](./11a-replay-and-conv-encoding.md)
