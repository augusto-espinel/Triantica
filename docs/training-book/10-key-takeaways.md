> **[📚 Training Book Index](./README.md)** | [← Chapter 9: Experiment Reference](./09-experiment-reference.md) | [Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS →](./11-deep-dive-convolutions-mcts.md)

---

## 10. Key Takeaways for RL Practitioners {#10-takeaways}

### What We Learned

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

### Recommendations for Similar Projects

**Starting Out:**
1. Use simple frameworks first (brain.js) for rapid iteration
2. Implement comprehensive logging before tuning
3. Verify baseline metrics match expected distributions

**During Development:**
1. Keep a training journal (like this one)
2. Compare against previous experiments, not just absolute metrics
3. Use visualization to catch bugs early (win rate histograms, score distributions)

**Architecture Decisions:**
1. Start with feedforward to establish baseline
2. If you hit a wall, evaluate if search is needed
3. Actor-critic helps with variance but may not break strategic walls

**Hardware:**
1. Unified memory (Apple Silicon) simplifies batch size tuning
2. Node.js version compatibility matters for native ML libraries
3. Profile GPU utilization to find batch size sweet spot

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

---

> [← Chapter 9: Experiment Reference](./09-experiment-reference.md) | **[📚 Training Book Index](./README.md)** | [Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS →](./11-deep-dive-convolutions-mcts.md)
