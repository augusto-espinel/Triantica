# Triantica: A Practical Guide to Reinforcement Learning

**From Zero to RL Practitioner — Learning Through a Real Game AI Project**

---

## About This Guide

This document is both a training journal and a comprehensive introduction to reinforcement learning (RL). If you're not an ML engineer but want to understand RL through hands-on experience, this guide walks through every concept, mistake, and breakthrough from the Triantica project.

**What is Triantica?** A grid-based strategy game where players place triangles to enclose territory. The AI learns through self-play reinforcement learning — playing millions of games against itself and improving over time.

**What you'll learn:**
- Core RL concepts (policy networks, actor-critic evaluation, MCTS search, and exploration vs exploitation)
- How 2D convolutional filters scan canonical board tensors to detect spatial geometry (diagonals, corners, and enclosures)
- How the Policy Head ("where to play") and Value Head ("who is winning") collaborate inside MCTS and during backpropagation
- Exact matrix and tensor dimensions at every layer—and how they map to Apple M2 Ultra Unified Memory and Metal GPU cores
- How to diagnose and fix subtle engine, search, encoding, and loss-scaling bugs when an AI appears to learn but fails to win

**Prerequisites:** Basic programming knowledge. No ML background required.

> **Modular Chapter Layout:** To make updates fast and prevent LLM context-window saturation, the Training Book is organized into standalone chapter files under `docs/training-book/`.

---

## Table of Contents

| Chapter | File | Scope & Key Topics |
|---------|------|--------------------|
| **1** | **[The Game: Why Triantica is Interesting for RL](./01-the-game.md)** | Game rules, $T^2$ toroidal topology, enclosure cascades, and the 3-stage state representation evolution (`401` flat → `10×10×18` canonical → $T^2$ circular manifold). |
| **2** | **[RL Fundamentals Through This Project](./02-rl-fundamentals.md)** | The RL loop, policies, Policy Gradient vs Actor-Critic vs AlphaZero MCTS, credit assignment, exploration vs exploitation, and telemetry metrics. |
| **3** | **[The Architecture Journey: brain.js → TensorFlow.js](./03-architecture-journey.md)** | Phase 1 (`brain.js` CPU baseline) to Phase 2 (`@tensorflow/tfjs-node`), migration challenges, and forensic comparison of `exp20` vs `TF-B1`. |
| **4** | **[Hardware: Mac Studio M2 Ultra & Matrix Sizing Deep Dive](./04-hardware-m2-ultra.md)** | Unified memory architecture, exact tensor dimensions at every layer, Apple AMX / Accelerate vs Metal GPU dispatch, and wall-clock cycle breakdown. |
| **5** | **[Parameters Explored](./05-parameters-explored.md)** | Hyperparameter overview across Phases 1–6, learning rate schedules (constant vs cosine annealing), and loss weighting. |
| **6** | **[Training Strategies](./06-training-strategies.md)** | Pure self-play, mixed opponents ("Learn from the Master"), $TD(\lambda)$ cluster shaping vs true Value Critics, and sliding-window FIFO replay. |
| **7** | **[Critical Bugs, Forensic Audits, and Why They Matter](./07-critical-bugs-and-audits.md)** | Detailed post-mortems of all 14 critical bugs across training, perspective encoding, MCTS value backups, and batched quiescence search. |
| **8** | **[The Heuristic Wall: Architectural Limits & How We Broke It](./08-the-heuristic-wall.md)** | Why feedforward networks plateaued at `0%` vs Heuristic, and the 5-stage architectural climb (`0% → 63% → 90% → 100%`). |
| **9** | **[Experiment Reference](./09-experiment-reference.md)** | Complete cross-stage experiment results table (`exp12` through `TF-B15`), telemetry signatures, and parameter glossary. |
| **10** | **[Key Takeaways for RL Practitioners](./10-key-takeaways.md)** | Core engineering lessons, diagnostic checklists, recommendations for board-game RL projects, and further reading. |
| **11** | **[Deep Dive: Convolutions, Dual Heads, MCTS & Pipeline Consolidation](./11-deep-dive-convolutions-mcts.md)** | Phase-by-phase technical blueprint (Phases 1–6): Replay Buffer, `10×10×18` ConvNet, Dual-Head Actor-Critic, BatchNorm, PUCT MCTS, Quiescence MCTS math, and $T^2$ Toroidal Padding. |
| **12** | **[Toroidal 3D Visualization, Dual-View Projection & Human Telemetry](./12-toroidal-3d-and-telemetry.md)** | Synchronized 2D viewport + interactive 3D ring torus (`Torus3DView.js`), parametric surface math, and continuous human match telemetry (`HumanGameLogger.js`). |
| **Appendices** | **[Appendices A, B & C: Commands, Project Structure & Telemetry Schema](./appendices.md)** | Reproducible CLI commands for all 6 phases, repository directory tree, and complete CLI flag & dashboard JSON schema reference. |

---

*Last updated: 2026-10-31*  
*Version: 4.1 (Modular Chapter Edition — Phase 1 to Phase 6 Full Torus $T^2$ Mastery)*  
*Framework: TensorFlow.js (`@tensorflow/tfjs-node`) on Apple Silicon (AMX / Accelerate)*  
*Hardware: Mac Studio M2 Ultra (24-core CPU, 60-core GPU, 64GB Unified Memory)*
