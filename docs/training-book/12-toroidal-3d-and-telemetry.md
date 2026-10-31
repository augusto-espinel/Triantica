> **[📚 Training Book Index](./README.md)** | [← Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS](./11-deep-dive-convolutions-mcts.md) | [Appendices A, B & C →](./appendices.md)

---

## 12. Toroidal 3D Visualization, Dual-View Projection & Human Telemetry {#12-toroidal-visualization}

To make the $T^2$ topology intuitive for human players while preserving the mathematical integrity of the core engine and AI tensor pipeline, Triantica enforces a strict **Engine-View Separation Architecture**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 STRICT ENGINE / VIEW-PROJECTION SEPARATION                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CORE ENGINE & AI LAYER (`GridManager.js`, `ConvNetwork.cjs`, `MCTS.cjs`)   │
│   ────────────────────────────────────────────────────────────────────────   │
│   • Canonical discrete `[10 × 10]` cell coordinates (`x, y ∈ {0..9}`)        │
│   • Phase 2 (`v1.0.0-2d`): Planar corner placement + Toroidal flood-fill     │
│   • Phase 3 (`TF-B15`): Full $T^2$ modulo vertex & edge adjacency (100       │
│     unique vertices, each shared by 12 triangles) + `ToroidalPad2D`          │
│   • Zero dependency on camera angles, 3D meshes, or Phaser rendering         │
│                                                                              │
│                                      │                                       │
│                         Pure State Projection $(u, v)$                       │
│                                      ▼                                       │
│   VIEW-PROJECTION LAYER (`ToroidalViewport.js`, `Torus3DView.js`)            │
│   ───────────────────────────────────────────────────────────────            │
│   • 2D Toroidal Panning (`Shift+Drag` / Right-Drag / Arrow Keys):            │
│     Shifts visual origin `(offsetX, offsetY)` modulo 10 so any wrap-around   │
│     cluster can be centered seamlessly on the flat screen.                   │
│   • Synchronized 3D Torus Surface (`Torus3DView.js`):                        │
│     Maps normalized grid coordinates $(u, v) = (x/10, y/10) \in [0, 1)^2$    │
│     onto the parametric 3D ring torus ($R = 2.5, r = 1.0$):                  │
│       $X(\theta, \phi) = (R + r \cos\phi)\cos\theta$                         │
│       $Y(\theta, \phi) = (R + r \cos\phi)\sin\theta$                         │
│       $Z(\theta, \phi) = r \sin\phi$                                         │
│     where $\theta = 2\pi u$ (major ring) and $\phi = 2\pi v$ (minor tube).   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Continuous Learning from Human Play (Human Match Telemetry Logger, `HumanGameLogger.js`)
With `TF-B15` deployed to production (`trainedAI_network.json`) across **Easy**, **Medium** (`99.0%` Raw 1-ply ConvNet), and **Hard** (`100.0%` ConvNet + Tactical Safety Filter), the browser and server runtimes include a **human match telemetry logger** (`src/data/HumanGameLogger.js`, persisted via `/api/human-games`). Every completed human match records canonical 18-channel states, chosen actions, enclosure claims, and final outcomes—enabling future human-in-the-loop replay mixing without altering the core self-play pipeline.

---

---

> [← Chapter 11: Deep Dive — Convolutions, Dual Heads & MCTS](./11-deep-dive-convolutions-mcts.md) | **[📚 Training Book Index](./README.md)** | [Appendices A, B & C →](./appendices.md)
