> **[📚 Training Book Index](./README.md)** | [← Chapter 9: Experiment Reference](./09-experiment-reference.md) | [Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS →](./11-deep-dive-convolutions-mcts.md)

---

## 10. Key Takeaways for RL Practitioners {#10-takeaways}

### Five Core Engineering Lessons

1. **Architecture & Representation Dominate Hyperparameters:** No amount of learning-rate tuning, momentum tweaking, or reward shaping could make a 1D feedforward network play strategically (`0%` vs Heuristic across `3,000+` games). True progress came from matching the network topology (`10×10×18` canonical spatial tensor + $3\times 3$ Conv2D) and search algorithm to the structure of the game.
2. **Domain-Aware Search Breaks the Strategic Barrier:** Neither a 2-Head ConvNet alone (`TF-B5`) nor vanilla AlphaZero MCTS (`TF-B10`) could beat `HeuristicPlayer`, because evaluating non-quiescent leaf states blinded the Value Head to multi-turn enclosure cascades. Extending MCTS with **greedy leaf quiescence rollouts** (`src/server/QuiescenceMCTS.cjs`) immediately unlocked `63% → 90% → 99%` mastery.
3. **Systemic Bugs Hide Silently Across Subsystem Boundaries:** Across our journey we uncovered **14 critical bugs** spanning perspective alignment, diagonal overlap logic, canonical slot collisions, replay value zeroing, $400\times$ policy loss dilution, `tanh` saturation, browser heuristic overrides, and evaluation seat-swaps. Auditing tensor magnitudes, role-split win rates (`as P1` vs. `as P2`), and cycle loss curves is mandatory before trusting any benchmark.
4. **Convolution Padding Must Respect Manifold Topology:** Standard zero-padding (`padding='same'`) treats the borders of a toroidal grid as artificial walls, causing a ~20% win-rate ceiling at the seam. Replacing zero-padding with **Toroidal Circular Padding (`ToroidalPad2D`)** across all convolutional layers lifted raw 1-ply policy performance from `80.1%` to **`95.9%` average (`99.0%` peak)**.
5. **Hybrid Neuro-Symbolic Safety Eliminates Residual Blunders:** Pairing the `0.8 ms` raw 1-ply neural policy with an $O(1)$ **`topK = 12` Inference-Time Tactical Safety Filter** (`< 0.4 ms` total budget) closed the final 1% horizon gap, achieving **`100.0%` (`1,000W / 0L / 0D` across all 10 cycles)** against `HeuristicPlayer`.

<details>
<summary>📐 Diagram: Key Lessons (click to expand ASCII)</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                    Key Lessons                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Architecture > Hyperparameters                          │
│     ──────────────────────────────                          │
│     No amount of learning rate tuning will make a            │
│     feedforward network play strategically.                  │
│                                                             │
│  2. Domain-Aware Search Breaks the Strategic Barrier        │
│     ────────────────────────────────────────────────        │
│     Conv + Value Head alone hits 0% vs Heuristic.           │
│     Vanilla MCTS also fails on deterministic cascades.      │
│     Quiescence MCTS + Dual-Side Teacher unlocks 90%–100%.   │
│                                                             │
│  3. Systemic Bugs Hide Across Interfaces                    │
│     ────────────────────────────────────                    │
│     We uncovered 11 critical bugs across geometry, MCTS,    │
│     canonical encoding, replay targets, and 1/400 loss      │
│     scaling. Always audit end-to-end tensor magnitudes.     │
│                                                             │
│  4. Topology Must Match the Convolution Padding             │
│     ───────────────────────────────────────────             │
│     Replacing zero-padding with circular toroidal padding   │
│     (`ToroidalPad2D`) eliminated border blind spots and     │
│     lifted raw 1-ply policy from 80.1% to 95.9% (99% peak). │
│                                                             │
│  5. Hybrid Neuro-Symbolic Safety Eliminates Horizon Blunders│
│     ────────────────────────────────────────────────────────│
│     Pairing the 1-ply ConvNet with a lightweight top-K      │
│     tactical safety filter achieved 1,000W / 0L / 0D (100%) │
│     across all 10 cycles against HeuristicPlayer.           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Practical Recommendations for Systems & RL Engineers

- **Prototype Fast, Instrument Deeply:** Start with the simplest viable model (`brain.js`) to validate the game loop and build diagnostic telemetry first (role-split win rates, cluster score distributions, opening/board Shannon entropy, and start-vs-end cycle loss).
- **Separate Representation, Critic, and Search:** When an agent stalls against a tactical baseline, diagnose whether the bottleneck is **spatial perception** (1D vector vs. 2D/toroidal tensor), **credit assignment** (outcome smearing vs. learned Value Critic $V(s)$), or **lookahead horizon** (1-ply vs. MCTS with quiescence rollouts).
- **Design for Hardware Symmetry:** On unified memory architectures like the Mac Studio M2 Ultra, aligning tensor channel depths (`32, 64`), MCTS leaf batches (`B = 8`), and replay mini-batches (`B = 256`) keeps the entire `594 KB` weight set in L2 cache and saturates AMX/Metal matrix tiles without PCIe copy bottlenecks.

### Further Reading

**Reinforcement Learning:**
- [Sutton & Barto: Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

**Game AI:**
- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

**TensorFlow.js:**
- [TF.js Documentation](https://www.tensorflow.org/js)
- [tfjs-node GitHub](https://github.com/tensorflow/tfjs-node)

---

> [← Chapter 9: Experiment Reference](./09-experiment-reference.md) | **[📚 Training Book Index](./README.md)** | [Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS →](./11-deep-dive-convolutions-mcts.md)
