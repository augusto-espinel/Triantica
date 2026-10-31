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

---

## Table of Contents

1. [The Game: Why Triantica is Interesting for RL](#1-the-game)
2. [RL Fundamentals Through This Project](#2-rl-fundamentals)
3. [The Architecture Journey: brain.js → TensorFlow.js](#3-architecture-journey)
4. [Hardware: Mac Studio M2 Ultra & Matrix Sizing Deep Dive](#4-hardware)
5. [Parameters Explored](#5-parameters)
6. [Training Strategies](#6-training-strategies)
7. [Critical Bugs, Forensic Audits, and Why They Matter](#7-critical-bugs)
8. [The Heuristic Wall: Understanding Architectural Limits & How We Broke It](#8-heuristic-wall)
9. [Experiment Reference](#9-experiment-reference)
10. [Key Takeaways for RL Practitioners](#10-takeaways)
11. [Deep Dive: Convolutions, Dual Heads, MCTS & Pipeline Consolidation](#11-optimization-roadmap)
12. [Toroidal 3D Visualization, Dual-View Projection & Human Telemetry](#12-toroidal-visualization)

---

## 1. The Game: Why Triantica is Interesting for RL {#1-the-game}

### Game Rules in One Minute

```
┌──────────────────────────────────────────────────────────┐
│  • 10×10 toroidal grid (edges wrap around seamlessly)    │
│  • Players alternate placing right-isosceles triangles   │
│  • Up to 2 triangles per cell along a shared diagonal:   │
│      Group 0 (╱): Rot 0 (Top-Left) & Rot 2 (Bottom-Right)│
│      Group 1 (╲): Rot 1 (Bottom-Left) & Rot 3 (Top-Right)│
│  • Must touch an existing corner to be a legal placement │
│  • Enclose empty triangle slots to auto-claim them       │
│  • Capturing enclosures grants extra turns:              │
│      extraTurns += floor((1 + claimed) / 2)              │
│  • Largest contiguous cluster of triangles wins!         │
└──────────────────────────────────────────────────────────┘
```

#### Cell Geometry & The 4 Triangle Rotations

Every square cell `(x, y)` on the $10 \times 10$ board can hold up to **2 right-isosceles triangles**. Because two triangles in the same square cannot overlap, they must share the same diagonal cut (`╱` or `╲`):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                TRIANTICA CELL GEOMETRY & DIAGONAL GROUPS                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│      GROUP 0: Slash Diagonal (╱)            GROUP 1: Backslash Diagonal (╲)  │
│                                                                              │
│     Rot 0 (0° Top-Left)   Rot 2 (180° Bot-Right)   Rot 1 (90° Bot-Left)   Rot 3 (270° Top-Right) │
│        ┌─────────┐           ┌─────────┐              ┌─────────┐            ┌─────────┐         │
│        │ ▲ ▲ ▲ ╱ │           │       ╱ │              │ ╲       │            │ ╲ ▲ ▲ ▲ │         │
│        │ ▲ ▲ ╱   │     +     │     ╱ ▲ │              │   ╲     │      +     │   ╲ ▲ ▲ │         │
│        │ ▲ ╱     │           │   ╱ ▲ ▲ │              │ ▲   ╲   │            │     ╲ ▲ │         │
│        │ ╱       │           │ ╱ ▲ ▲ ▲ │              │ ▲ ▲   ╲ │            │       ╲ │         │
│        └─────────┘           └─────────┘              └─────────┘            └─────────┘         │
│                                                                              │
│   • Rule 1 (No Overlap): If a cell already holds Rot 0, only Rot 2 fits.     │
│                          If a cell already holds Rot 1, only Rot 3 fits.     │
│   • Rule 2 (Corner Touch): Every move after Ply 1 must touch at least one    │
│                            vertex of an already-placed triangle (8-way       │
│                            neighborhood on the T² torus).                    │
│   • Rule 3 (Cluster Connectivity): Triangles belong to the same contiguous   │
│                            cluster ONLY if they belong to the same player    │
│                            AND share a full edge (horizontal, vertical, or   │
│                            internal diagonal—corner touch alone does not     │
│                            connect a cluster!).                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Visual example of enclosure:**

```
Before move:          After move (enclosure):
    ┌───┬───┬───┐         ┌───┬───┬───┐
    │ ▲ │   │ ▼ │         │ ▲ │███│ ▼ │
    ├───┼───┼───┤         ├───┼───┼───┤
    │ ◀ │ ? │ ▶ │    →    │ ◀ │▲▼▶│ ▶ │
    ├───┼───┼───┤         ├───┼───┼───┤
    │ ▼ │   │ ▲ │         │ ▼ │███│ ▲ │
    └───┴───┴───┘         └───┴───┴───┘
                        Center enclosed!
```

#### Why the Extra-Turn Snowball Mechanic Changes Everything

When a move encloses $k \ge 1$ empty triangle slots via flood-fill (`claimEnclosedTriangles`), two things happen simultaneously:
1. All $k$ enclosed slots are immediately filled with the active player's triangles (expanding their contiguous cluster).
2. The player earns **bonus turns** according to:
   $$\text{extraTurns} = \left\lfloor \frac{1 + k}{2} \right\rfloor$$

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 THE ENCLOSURE CASCADE ("SNOWBALL") DYNAMICS                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Blunder (Leaves 1 Hole) ──► Opponent Caps 1 Slot (k=1)                     │
│                                     │                                        │
│                                     ▼  extraTurns = floor((1+1)/2) = +1 Turn │
│                               Opponent Places Free Piece                     │
│                                     │                                        │
│                                     ▼  Creates New Enclosure (k=3 Slots!)    │
│                               extraTurns = floor((1+3)/2) = +2 Turns!        │
│                                     │                                        │
│                                     ▼                                        │
│                         Cascading Board Sweep (170 vs 4)                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Strategic & RL Consequence (Second-Mover Counter-Attack Edge):**
- Even enclosing a **single triangle ($k = 1$)** yields $\lfloor 2/2 \rfloor = \mathbf{1\text{ extra turn}}$, making single-slot enclosures tempo-free ("free moves" that keep the initiative!).
- Enclosing **$k = 3$ triangles** yields **2 extra turns**, allowing a player to chain multi-step enclosure cascades before the opponent ever gets to move again.
- **P2 Counter-Attack Dynamics:** In early deterministic openings without random 2-ply seeding, Player 2 reacts directly to Player 1's opening placement, using the first defensive wall to trigger the initial enclosure cascade. Because winning is determined by the **largest single contiguous edge-sharing cluster** (out of 200 total triangles when the board fills), a single mid-game cascade can sever the opponent's cluster in two while merging the attacker's pieces into a 140+ triangle super-cluster.

**Key insight:** The game has **400 possible actions** per turn (100 cells × 4 rotations), but only ~50 are typically valid. The AI must learn:
1. Which moves are legal (corner-touch + diagonal compatibility)
2. Which moves create immediate enclosures and tempo-preserving bonus turns
3. Which moves build toward future enclosures while **never leaving an unguarded 1-move cavity** for the opponent to chain-cascade!

### Why This Game Exposes RL Challenges

```
┌─────────────────────────────────────────────────────────────┐
│                    Game Complexity Comparison                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Game        │ Legal Moves │ Evaluation      │ Difficulty  │
│  ────────────┼─────────────┼─────────────────┼────────────│
│  Chess       │ ~35         │ Piece values     │ Medium      │
│  Go          │ ~250        │ Territory count │ Hard        │
│  Triantica   │ ~50         │ Delayed signal  │ Hard        │
│                                                             │
│  The challenge: A move that looks good now might be       │
│  terrible in 20 turns because it blocked your own         │
│  future enclosure.                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**This is the core RL problem:** Credit assignment. How do you learn which move, 30 turns ago, caused your victory?

### State Representation

The neural network doesn't see the board visually. It sees a vector of numbers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Vector (401 values)                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each cell (100 cells × 4 values):                      │
│  ─────────────────────────────                              │
│  [0] Triangle 1 player: +1 if AI, -1 if opponent, 0 if empty│
│  [1] Triangle 1 rotation: 0.0, 0.25, 0.5, or 0.75           │
│  [2] Triangle 2 player: +1 if AI, -1 if opponent, 0 if empty│
│  [3] Triangle 2 rotation: 0.0, 0.25, 0.5, or 0.75           │
│                                                             │
│  Final element [400]: Always 1 (bias signal)               │
│                                                             │
│  Example cell with one AI triangle at 90°:                  │
│  [+1, 0.25, 0, -1, ...]  ← T1 is AI at 90°, T2 empty       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this encoding?** This "fixed perspective" encoding means:
- AI always sees its own pieces as +1
- Opponent's pieces as -1
- The same board position looks the same regardless of which player the AI is assigned

This is critical for generalization (see Bug #1 in Section 7).

### How State Representation Evolved Across the Project

While our early feedforward experiments (`brain.js` and `TF-B1` through `TF-B4`) used this flat `401`-element vector, a 1D vector destroys 2D board geometry: cells `(x=4, y=4)` and `(x=4, y=5)` are adjacent on the board, yet sit `40` indices apart in the flattened array.

To unlock spatial pattern recognition and eventually break the Heuristic Wall, our representation evolved through three distinct stages (detailed in [Chapter 11](#11-optimization-roadmap)):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│            EVOLUTION OF TRIANTICA BOARD STATE REPRESENTATION                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Flat 401-Element Vector (brain.js & TF-B1..B4)                     │
│  • Shape: [401] (100 cells × 4 scalar values + 1 bias)                       │
│  • Flaw:  Arbitrary creation-order slot shuffling (Bug 3) + no 2D locality   │
│                                                                              │
│  Stage 2: Canonical 10×10×18 Spatial Tensor (TF-B5..TF-B14)                  │
│  • Shape: [10, 10, 18] (Slot 0 = rots {0,1}; Slot 1 = rots {2,3}; 9-way 1-hot│
│  • Gain:  Strict rotation invariance + 3×3 Conv2D enclosure detection,       │
│           but zero-padded borders blurred toroidal wrap-around edges         │
│                                                                              │
│  Stage 3: Full T² Toroidal Manifold + ToroidalPad2D (TF-B15 Champion)        │
│  • Shape: [10, 10, 18] wrapped to [12, 12, C] before every 3×3 Conv2D layer  │
│  • Gain:  True translation equivariance across all 100 cells (no edge seams) │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. RL Fundamentals Through This Project {#2-rl-fundamentals}

### The RL Loop Explained

Reinforcement learning is different from supervised learning. There's no labeled dataset — the agent learns by interacting with the environment.

```
┌─────────────────────────────────────────────────────────────┐
│                    The RL Training Loop                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                      ┌─────────────┐      │
│   │             │   1. Choose action   │             │      │
│   │    Agent    │ ───────────────────► │    Game     │      │
│   │  (Neural    │      (move)         │ (State)     │      │
│   │   Network)  │                     │             │      │
│   │             │ ◄─────────────────── │             │      │
│   └─────────────┘   2. Observe result  └─────────────┘      │
│         │                                                    │
│         │  3. New state + reward                             │
│         │                                                    │
│         ▼                                                    │
│   ┌─────────────┐                                            │
│   │   Update    │   4. Learn from experience                │
│   │   policy    │      (adjust neural network)               │
│   └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**In Triantica:**
- **Agent**: Neural network that outputs move probabilities
- **Environment**: The game board + rules
- **State**: 401-number vector representing the board
- **Action**: Which triangle to place where (400 options)
- **Reward**: Points scored + bonus for winning

### What is a Policy?

A **policy** (denoted π) is the agent's strategy — it maps states to actions.

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy Network                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Layer         Hidden Layer         Output Layer      │
│  (401 neurons)       (200 neurons)        (400 neurons)     │
│                                                             │
│     ○ ○ ○ ○              ○ ○ ○               ○ ○ ○ ○        │
│     ○ ○ ○ ○    ╲╱╲╱      ○ ○ ○     ╲╱╲╱     ○ ○ ○ ○        │
│     ○ ○ ○ ○    ╱╲╱╲      ○ ○ ○     ╱╲╱╲     ○ ○ ○ ○        │
│     ○ ○ ○ ○              ○ ○ ○               ○ ○ ○ ○        │
│       ...                ...                 ...            │
│                                                             │
│  Board state    Learned patterns      Move probabilities   │
│  (401 values)   (intermediate)        (higher = better)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Mathematical form: π(a|s) = P(taking action a | in state s)
```

**Key insight:** The policy network doesn't output a single "best move". It outputs a probability distribution over all possible moves. This allows:
1. **Exploration**: Sometimes picking lower-probability moves to discover new strategies
2. **Training**: We can reinforce good moves (increase their probability) and discourage bad ones

### Policy Gradient: How We Train

The simplest RL algorithm is **policy gradient**. Here's how it works:

```
┌─────────────────────────────────────────────────────────────┐
│                Policy Gradient Training                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: Play games                                         │
│  ──────────────────                                         │
│  - Agent plays against itself (self-play)                  │
│  - Record: state, action chosen, reward received           │
│                                                             │
│  Step 2: Label moves                                        │
│  ──────────────────                                         │
│  - If game won: label all moves as "good" (target = +1)    │
│  - If game lost: label all moves as "bad" (target = -1)   │
│  - Adjust by temporal position (early moves less credit)   │
│                                                             │
│  Step 3: Train network                                       │
│  ──────────────────                                         │
│  - Input: state                                             │
│  - Target output: probability of chosen move = reward      │
│  - Minimize difference between predicted and target        │
│                                                             │
│  Effect: Network learns to increase probability of         │
│         moves that led to winning games                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Mathematical intuition:**

```
Loss = -log(π(action | state)) × reward

If reward is positive (won): minimize -log(π) → increase π
If reward is negative (lost): minimize -log(π) → decrease π
```

### Three Approaches to RL

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  THE 3 RL PARADIGMS BUILT IN TRIANTICA                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Pure Policy Gradient (Phase 1 & 2: brain.js exp1–22, TF-B1–B4)           │
│     ──────────────────────────────────────────────────────────────           │
│     • Play games, label winning moves as "good", losing moves as "bad"       │
│     • Simple, fast, no lookahead; high variance (noisy gradients)            │
│     • Result: 91% vs Random (exp20), 0% vs Heuristic ("The Heuristic Wall") │
│                                                                              │
│  2. 2-Head Convolutional Actor-Critic (Phase 3: TF-B5–TF-B6)                 │
│     ────────────────────────────────────────────────────────                 │
│     • Shared 3×3 Conv2D backbone + Two Heads: Actor (π) & Critic (V)         │
│     • Critic V(s) ∈ [-1, +1]: "Who is winning from this position?"          │
│     • Actor π(a|s): "Which legal move has the highest probability?"          │
│     • Result: Spatial feature detection unlocked; ready for tree search      │
│                                                                              │
│  3. AlphaZero + Quiescence MCTS on T² Torus (Phase 4 & 5: TF-B8–TF-B15)      │
│     ────────────────────────────────────────────────────────────────────     │
│     • Neural network guides Monte Carlo Tree Search (50 sims/move)           │
│     • Quiescence Extension resolves forced enclosure cascades at leaves      │
│     • ToroidalPad2D wraps the 10×10 board seamlessly across all 4 borders    │
│     • Result: 100% vs Random, 99.0% Raw / 100.0% +Safety vs Heuristic!       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

Our initial feedforward network (`brain.js` and `TF-B1`–`TF-B4`) used pure **policy gradient** — the simplest approach. It worked for beating random players (`91%`), but scored `0%` against strategic opponents (`HeuristicPlayer`), hitting what we called the **"Heuristic Wall"** (see [Chapter 8](#8-heuristic-wall)). Over the course of the project, we systematically built all three paradigms: transitioning to a **2-Head Convolutional Actor-Critic** (`TF-B5`–`TF-B6`), integrating **AlphaZero-style MCTS** (`TF-B8`–`TF-B10`), and ultimately inventing **Quiescence MCTS + Toroidal Manifold Convolutions** (`TF-B12`–`TF-B15`) to reach **`99%` raw / `100%` +Safety** against the Heuristic master.

### Why Credit Assignment Fails in Triantica Without a Value Critic & Search

Why did pure Policy Gradient hit an impenetrable `0%` ceiling against a tactical opponent, even after 45 training cycles? Because of **delayed catastrophic punishment**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│            THE CREDIT ASSIGNMENT PROBLEM IN SNOWBALL GAMES                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Ply 1..38: AI plays 19 solid territorial moves (building a 30-piece cluster)│
│  Ply 40:    AI makes ONE careless move that leaves a 1-move cavity           │
│  Ply 41:    Heuristic opponent caps the cavity → triggers a 12-turn cascade! │
│  Ply 170:   Game ends. Score: AI = 14, Heuristic = 166. Outcome z = -1.      │
│                                                                              │
│  What Pure Policy Gradient (Stage 1) Does:                                   │
│  ─────────────────────────────────────────                                   │
│  • Assigns a negative target (z = -1) to ALL 20 moves the AI played!         │
│  • The 19 good moves at Plies 1..38 are punished alongside the 1 blunder     │
│    at Ply 40. The network unlearns its good opening and midgame patterns!    │
│                                                                              │
│  What Actor-Critic + Quiescence MCTS (Stages 2 & 3) Does:                    │
│  ────────────────────────────────────────────────────────                    │
│  • At Ply 40, MCTS simulates the candidate move, triggers the Quiescence     │
│    cascade rollout immediately at the leaf, and the Value Head sees V = -0.9 │
│  • MCTS shifts visit counts AWAY from the blunder before the move is played  │
│  • Every state s_t is localized: good positions keep high value V(s_t) > 0,  │
│    while only the specific blunder branch is penalized!                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Exploration vs Exploitation

```
┌─────────────────────────────────────────────────────────────┐
│               Exploration vs Exploitation                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Exploration (ε = 0.3):                                     │
│  ────────────────────                                       │
│  • Try random moves 30% of the time                         │
│  • Discover new strategies                                  │
│  • Risk: Wastes time on bad moves                           │
│                                                             │
│  Exploitation (ε = 0.0):                                    │
│  ─────────────────────                                       │
│  • Always pick highest probability move                     │
│  • Use current knowledge                                     │
│  • Risk: Misses better strategies                           │
│                                                             │
│  Balance:                                                   │
│  ─────────                                                   │
│  • High exploration early (ε = 0.5)                         │
│  • Reduce over time (ε → 0.1)                               │
│  • Zero for evaluation                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

In our project, how we balanced exploration and exploitation evolved alongside the architecture:
1. **Stage 1 (Feedforward Policy Gradient):** Used **$\varepsilon$-greedy exploration** (`explorationRate = 0.1` to `0.3`) plus a `50%` random opening override on the first 3 plies to prevent deterministic self-play loops.
2. **Stage 2 & 3 (AlphaZero / Quiescence MCTS):** Replaced naive random moves with **Dirichlet root noise** ($\text{Dir}(\alpha=0.3)$, $\varepsilon=0.25$) blended into the root prior $P(s,a)$, combined with **PUCT uncertainty bonuses** ($c_{\text{puct}} \cdot P(s,a) \frac{\sqrt{N(s)}}{1 + N(s,a)}$) and temperature-controlled visit sampling ($\tau = 1.0$ for moves $1\dots 15$, $\tau = 0.1$ thereafter). This ensures the agent explores *plausible tactical alternatives* rather than throwing away games with random blunders.

### Telemetry & Metrics: How We Read the AI's Mind During Training

When training a reinforcement learning agent over hours of self-play, watching a single "Win Rate" number is like flying an airplane with only an altimeter. When the win rate stalls at `0%` or suddenly crashes from `54%` to `15%`, you need diagnostic instruments that separate **exploration health**, **optimization health**, **role symmetry**, and **true strategic dominance**.

Throughout this guide, every experiment is tracked through six core telemetry signals on the Triantica Training Dashboard:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│             THE 6 CORE TELEMETRY SIGNALS & WHAT THEY DIAGNOSE                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Win Rate Split by Role (P1 First-Mover vs P2 Second-Mover)               │
│     • Why we track it: Triantica has a massive P2 counter-attack advantage   │
│       in deterministic openings. Overall win rate hides role-blindness!      │
│     • Healthy signature: Balanced competence in both seats (e.g., TF-B15     │
│       achieves 100% as P1 and 98% raw / 100% +Safety as P2).                 │
│     • Pathological signature: 50% overall win rate caused by 100% P2 wins    │
│       and 0% P1 wins—or a flat 50.0% caused by the Role-Swap Bug (Bug #11).  │
│                                                                              │
│  2. Score Distribution (Largest Contiguous Cluster: AI vs Opponent)          │
│     • Why we track it: Win rate is binary (0 or 1); cluster scores (out of   │
│       200 total triangles) measure territorial margin BEFORE the first win.  │
│     • Healthy signature: Against Heuristic, losing 4 to 170 means total      │
│       cascade wipeout; climbing to 65 vs 95 signals the AI is surviving      │
│       mid-game cascades even if win rate is still 0%; 138 vs 32 (TF-B15)     │
│       proves total territorial dominance.                                    │
│                                                                              │
│  3. Opening Diversity (Shannon Entropy of First Moves, in bits)              │
│     • Formula: H_open = - Σ p(m_1) log_2 p(m_1)  (Max = log_2(100) ≈ 6.64 b) │
│     • Why we track it: Detects policy mode collapse early in the game.       │
│     • Healthy signature: 5.0 – 6.2 bits during MCTS self-play (exploring     │
│       40–75 distinct openings per 100-game cycle).                           │
│     • Pathological signature: Dropping below 2.5 bits means the network      │
│       is memorizing a single deterministic opening line.                     │
│                                                                              │
│  4. Board Diversity (Shannon Entropy of Final End-Game Boards, in bits)      │
│     • Why we track it: Even if openings vary, do games converge into the     │
│       same repetitive end-game structures?                                   │
│     • Healthy signature: ~1.95 – 1.98 bits across 100 games, confirming      │
│       rich structural variety in completed 200-triangle boards.              │
│                                                                              │
│  5. Cycle Error Curve: Initial Error (e_start) vs Final Error (e_end)        │
│     • Why we track it: Measures how surprised the network is by new          │
│       self-play data (e_start) and how well gradient descent fits it (e_end).│
│     • Healthy signature: A clear "sawtooth" drop every cycle                 │
│       (e.g., e_start = 1.91 ──► e_end = 1.38 in ConvNet Actor-Critic),       │
│       with both baseline and final loss trending downward across cycles.     │
│     • Pathological signatures:                                               │
│       - Crash-to-zero (e_end → 0.000000): Catastrophic overfitting to a tiny │
│         current-cycle buffer without replay memory.                          │
│       - Micro-drop plateau (1.312 ──► 1.299): Gradient cancellation (dual-   │
│         perspective bug) or 400× MSE policy loss dilution (Bug #10).         │
│                                                                              │
│  6. Raw 1-Ply Policy Eval vs +Safety Filter Eval                             │
│     • Why we track it: Separates pure neural intuition (single forward pass  │
│       argmax(π) in ~0.8 ms) from tactical blunder-checking.                  │
│     • Healthy signature: Raw network reaches 99.0% vs Heuristic on intuition │
│       alone; the 1-ply safety filter closes the final 1% edge case to 100%.  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Architecture Journey: brain.js → TensorFlow.js {#3-architecture-journey}

### Phase 1: brain.js (Experiments 1-22)

**Why start simple?** brain.js is a minimal neural network library (~1000 lines) that runs in browsers. Perfect for rapid prototyping and iteration.

```javascript
// brain.js: Simple and intuitive
const net = new brain.NeuralNetwork({ hiddenLayers: [200] });

net.train(trainingData, {
  iterations: 200,
  learningRate: 0.01,
  errorThresh: 0.001
});

// Run inference
const output = net.run(inputState);
```

**What worked:**
- Quick iteration: 8 minutes per 20-cycle training run
- Browser deployment trivial (just load brain.js)
- **91% win rate vs Random** achieved (exp20)
- Good for debugging (simple architecture)

**What didn't work:**
- **0% vs Heuristic** — we hit a wall
- No GPU support (CPU only)
- No custom loss functions
- No convolutional layers for spatial patterns
- Limited to feedforward networks

### Phase 2: TensorFlow.js Migration

```
┌─────────────────────────────────────────────────────────────┐
│                Why TensorFlow.js?                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  brain.js              │     TensorFlow.js                   │
│  ──────────────────────┼────────────────────────────────    │
│  MSE loss only         │  Custom loss functions ✅          │
│  Feedforward only      │  Conv layers, ResNet ✅            │
│  CPU only              │  Metal GPU acceleration ✅          │
│  No value head         │  Multi-head outputs ✅             │
│  ~8 min/train          │  ~35 min/train (100x faster step) │
│  No gradient control   │  Manual gradient computation ✅    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Migration Challenges

The TF.js implementation initially got **49% vs Random** (vs 91% with brain.js). Investigation revealed three critical bugs:

```
┌─────────────────────────────────────────────────────────────┐
│                    Migration Bugs                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 1: Reward Signal Dilution                              │
│  ─────────────────────────────                              │
│                                                             │
│  brain.js (correct):                                        │
│    target = output[chosenMoveIndex]                         │
│    // Only the chosen move gets the reward signal           │
│                                                             │
│  TF.js (buggy):                                             │
│    target = mean(output)  // Average of ALL 400 values!    │
│    // Legal moves at ~0.5, illegal at ~0.01                │
│    // Win signal (0.99) diluted to ~0.28                   │
│                                                             │
│  Result: Network barely learned from wins                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 2: Normalization Mismatch                              │
│  ────────────────────────────                               │
│                                                             │
│  brain.js: Dynamic range based on actual values             │
│  TF.js: Hardcoded min/max that didn't match game            │
│                                                             │
│  Result: Input values in wrong range                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Bug 3: Training Iterations                                  │
│  ─────────────────────                                       │
│                                                             │
│  brain.js: 200 iterations (default)                          │
│  TF.js: 10 iterations (copy-paste error)                     │
│                                                             │
│  Result: Undertrained network                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**How we found them:** Compared TF.js training logs with brain.js logs side-by-side. The win signal magnitude was visibly different.

### Forensic Audit: Why `exp20` (`brain.js`, `91%`) Beat Early `TF-B1` (`TF.js`, `49%`)

Even after fixing the initial migration bugs above, a deeper mystery remained: why did `exp20` in `brain.js` reach **`91%` vs Random**, whereas our initial clean `TF-B1` feedforward run in TensorFlow.js plateaued around **`49–53%` vs Random**?

We built a dedicated numerical audit suite (`reward_forensic_audit.cjs` and `reward_signal_audit.cjs`) to trace every gradient update and sample count across both engines:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│        FORENSIC AUDIT: exp20 (brain.js, 91%) vs TF-B1 (TF.js, 49%)           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Metric / Mechanism        │ exp20 (brain.js)      │ TF-B1 (Early TF.js)     │
│  ──────────────────────────┼───────────────────────┼─────────────────────────│
│  Games per Cycle           │ 100 games (~8,500 pos)│ 20 games (~850 pos)     │
│  Experience Replay Buffer  │ YES (+50% = 12,750)   │ NO (current cycle only) │
│  Weight Update Mode        │ Online SGD (batch = 1)│ Mini-batch MSE          │
│  Weight Updates / Cycle    │ 382,500 updates       │ 5,300 batched updates   │
│  Gradient Isolation        │ false (full 400 MSE)  │ true (masked 1-of-400)  │
│  vs Random Win Rate        │ 91.0%                 │ 49.0%                   │
│  vs Heuristic Win Rate     │  0.0%                 │  0.0%                   │
│                                                                              │
│  THE GRADIENT DILUTION & ISOLATION PARADOX:                                  │
│  ──────────────────────────────────────────                                  │
│  • Without gradient isolation (full 400-output MSE), ~350 illegal moves are  │
│    pushed toward 0.01 (gradient ≈ 0.12 × 350 = 42.0), while the ONE chosen   │
│    move receives gradient ≈ 0.04 (only 1.8% of the total gradient!).         │
│  • In brain.js, 382,500 online SGD steps + a 50% Replay Buffer brute-forced  │
│    the network to suppress all 350 illegal moves first, which alone was      │
│    enough to beat a Random player 91% of the time!                           │
│  • Conversely, when TF-B1 enabled Gradient Isolation on a tiny 20-game       │
│    buffer with NO replay memory, the network overfit the 850 chosen moves    │
│    to zero error (e_end → 0.000000) while forgetting illegal move boundaries │
│    from prior cycles (catastrophic forgetting).                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Three Enduring Lessons from the Migration Audit:**
1. **Experience Replay is Non-Negotiable:** Training only on the current cycle's games causes the network to oscillate and forget earlier board configurations. Adding the **500,000-state sliding-window Replay Buffer** (`--replay-buffer-size 500000 --replay-ratio 2.0`) stabilized every subsequent TF.js run.
2. **Suppressing Illegal Moves Beat `Random`, Not `Heuristic`:** `exp20`'s `91%` win rate against `RandomPlayer` was largely achieved by learning legal corner-touch boundaries and basic 1-ply captures—not deep territorial strategy. Against `HeuristicPlayer`, both `exp20` and `TF-B1` scored **`0.0%`**.
3. **Core Engineering Principle (Masked Cross-Entropy over Active Decisions):** A policy loss function must neither average MSE across 399 unchosen outputs (which dilutes the chosen move gradient by $400\times$, see Bug #10 in Chapter 7) nor leave unchosen legal moves unconstrained. This realization directly led to our **Legal-Move Masked Softmax Cross-Entropy** ($-\sum_{a \in \text{legal}} \pi_{\text{MCTS}}(a) \log p(a)$) in `TFConvNetwork`.

---

## 4. Hardware: Mac Studio M2 Ultra & Matrix Sizing Deep Dive {#4-hardware}

### The Unified Memory Advantage

```
┌─────────────────────────────────────────────────────────────┐
│              Traditional x86 + Discrete GPU                 │
│                                                             │
│   CPU ────── PCIe Bus (32–64 GB/s) ────── GPU               │
│   System RAM                              Discrete VRAM     │
│   (JS Replay Buffer) ──[Serialize/Copy]─► (Tensor Memory)   │
│                           │                                 │
│                    (I/O Bottleneck!)                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│              Apple M2 Ultra Unified Memory                  │
│                                                             │
│   24-Core CPU ◄──── Unified Memory Fabric ────► 60/76-Core  │
│                     800 GB/s Bandwidth          Metal GPU   │
│                     (Up to 192GB Shared Pool)               │
│                                                             │
│   Zero-copy tensor allocation. 500,000-state Replay Buffer  │
│   and batched MCTS leaf tensors reside in the same pool.    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why Unified Memory is a Game-Changer for AlphaZero-Style MCTS:**
In standard supervised deep learning, large batches sit on the GPU while the CPU stays idle. In **MCTS Self-Play RL**, execution alternates thousands of times per game between:
1. **CPU-bound Tree Traversal & Board Simulation** (`GridManager.clone()`, `executeMove()`, `claimEnclosedTriangles()`).
2. **GPU-bound Neural Network Inference** (`TFConvNetwork.batchPredict()` on $B=8$ MCTS leaves).
3. **GPU-bound Mini-Batch Backpropagation** (`trainBatchWithIsolation()` on $B=256$ replay samples).

On discrete PCIe GPUs, copying small MCTS leaf batches ($B=8$) back and forth across PCIe introduces latency that dwarfs the matrix multiplication itself. On the **Mac Studio M2 Ultra** (800 GB/s memory bandwidth), CPU arrays (`Float32Array`) and TensorFlow C++/Metal tensors share physical memory pages, eliminating bus stalls.

---

### How Matrix & Tensor Sizes Come About at Every Step

Every layer in `TFConvNetwork` transforms a 4D spatial tensor or 2D flattened matrix. Understanding **where each dimension comes from** and **how `im2col` (Image-to-Column) turns 2D convolutions into dense Matrix Multiplications (`GEMM`)** explains both the network's representational power and its hardware efficiency on the M2 Ultra.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           END-TO-END TENSOR & MATRIX PIPELINE (Batch Size B = 256)           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [STEP 0: Canonical Board Encoding]                                          │
│   10×10 Grid × 2 Slots/Cell × 9 One-Hot States/Slot = 18 Channels            │
│   Tensor Shape: [256, 10, 10, 18]  (460,800 floats = 1.84 MB per batch)      │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 32 filters) + BN + ReLU     │
│  [STEP 1: Conv Block 1 — Local Edge & Diagonal Detector]                     │
│   Kernel Tensor: [3, 3, 18, 32]  ──► Matrix Multiply: [25,600 × 162]·[162×32]│
│   Output Shape:  [256, 10, 10, 32] (5,184 weights + 32 biases + 128 BN)      │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 64 filters) + BN + ReLU     │
│  [STEP 2: Conv Block 2 — Multi-Cell Corner & Wedge Detector]                 │
│   Kernel Tensor: [3, 3, 32, 64]  ──► Matrix Multiply: [25,600 × 288]·[288×64]│
│   Output Shape:  [256, 10, 10, 64] (18,432 weights + 64 biases + 256 BN)     │
│         │                                                                    │
│         ▼  3×3 Conv2D (padding='same', stride=1, 64 filters) + BN + ReLU     │
│  [STEP 3: Conv Block 3 — 7×7 Receptive Field Enclosure Detector]             │
│   Kernel Tensor: [3, 3, 64, 64]  ──► Matrix Multiply: [25,600 × 576]·[576×64]│
│   Output Shape:  [256, 10, 10, 64] (36,864 weights + 64 biases + 256 BN)     │
│         │                                                                    │
│         ├───────────────────────────────────────┐                            │
│         ▼                                       ▼                            │
│  [STEP 4A: POLICY HEAD ("Where to Play")]  [STEP 4B: VALUE HEAD ("Who Wins")]│
│   1×1 Conv (2 filters): [256, 10, 10, 2]    1×1 Conv (1 filter): [256,10,10,1]│
│   Flatten:              [256, 200]          Flatten:             [256, 100]  │
│   Dense (200 → 400):    [256, 400]          Dense 1 (100 → 64):  [256, 64]   │
│   Activation:           Sigmoid ∈ (0, 1)    Dense 2 (64 → 1):    [256, 1]    │
│   (400 actions = 10×10 cells × 4 rots)      Activation:          tanh ∈ [-1,1]│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step Derivation of Matrix Dimensions

1. **Input Tensor `[B, 10, 10, 18]`:**
   - **Why `10 × 10`?** Matches the physical $10 \times 10$ Triantica board so spatial adjacency (up, down, left, right, diagonal) is preserved in memory coordinates.
   - **Why `18` channels?** Each cell holds up to **2 triangles** (Slot 0 for rotations $\{0,1\}$, Slot 1 for rotations $\{2,3\}$). Each slot has **9 mutually exclusive states**:
     - `1` channel for *Empty* (`ch 0` for Slot 0, `ch 9` for Slot 1)
     - `4` channels for *My Piece* at rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ$
     - `4` channels for *Opponent Piece* at rotations $0^\circ, 90^\circ, 180^\circ, 270^\circ$
   - **Why `padding='same'` vs. Circular Toroidal Padding (`ToroidalPad2D`)?** In standard planar ConvNets (`TF-B5`–`TF-B14`), `padding='same'` adds a 1-cell border of zeros around the $10 \times 10$ grid to create a $12 \times 12$ input window ($H_{out} = \lfloor \frac{10 + 2(1) - 3}{1} \rfloor + 1 = 10$). However, because Triantica is played on a continuous $10 \times 10$ torus ($T^2$), zero-padding creates an artificial "wall of zeros" at the borders! In `TF-B15` (Phase 3), we replaced zero-padding with **`ToroidalPad2D`**—wrapping Row `0` $\leftrightarrow$ Row `9` and Col `0` $\leftrightarrow$ Col `9` into a $[B, 12, 12, C]$ tensor before every `Conv2D(padding='valid')` layer so boundary enclosures are detected identically to center-board enclosures.

2. **How Convolutions Become Matrix Multiplications (`im2col` GEMM):**
   On hardware accelerators (Apple AMX coprocessor & Metal GPU), a 2D convolution is not executed as slow nested `for` loops. Instead, it is unrolled into a single massive **General Matrix Multiplication (`GEMM`)** $Y = X_{\text{col}} \cdot W$:
   - Across a batch of $B = 256$ boards, there are $256 \times 10 \times 10 = \mathbf{25,600}$ spatial patch locations.
   - **Conv1 (`18 → 32`):** Each $3 \times 3$ patch across $18$ input channels is flattened into a row vector of length $3 \times 3 \times 18 = \mathbf{162}$. The 32 filters form a weight matrix of shape $\mathbf{[162, 32]}$ ($5,184$ weights). The layer executes:
     $$\underbrace{[25,600 \times 162]}_{\text{Unrolled Patches } X_{\text{col}}} \times \underbrace{[162 \times 32]}_{\text{Filter Matrix } W_1} = \underbrace{[25,600 \times 32]}_{\text{Reshaped to } [256, 10, 10, 32]}$$
   - **Conv2 (`32 → 64`):** Each $3 \times 3$ patch across $32$ channels flattens to $3 \times 3 \times 32 = \mathbf{288}$ elements. Multiplied by $W_2$ of shape $\mathbf{[288, 64]}$ ($18,432$ weights):
     $$[25,600 \times 288] \times [288 \times 64] = [25,600 \times 64] \longrightarrow [256, 10, 10, 64]$$
   - **Conv3 (`64 → 64`):** Each $3 \times 3$ patch across $64$ channels flattens to $3 \times 3 \times 64 = \mathbf{576}$ elements. Multiplied by $W_3$ of shape $\mathbf{[576, 64]}$ ($36,864$ weights):
     $$[25,600 \times 576] \times [576 \times 64] = [25,600 \times 64] \longrightarrow [256, 10, 10, 64]$$

3. **Why $1 \times 1$ Bottleneck Convolutions Before the Dense Heads?**
   - After Conv3, each board position has a `[10, 10, 64]` feature map ($6,400$ floats). Connecting $6,400$ features directly to $400$ policy outputs would require $6,400 \times 400 = \mathbf{2,560,000}$ parameters—causing severe overfitting and slowing down MCTS inference by $15\times$!
   - Instead, following AlphaGo Zero, we apply a **$1 \times 1$ bottleneck convolution**:
     - **Policy Bottleneck (`pConv`):** Projects `[10, 10, 64]` $\to$ `[10, 10, 2]` via a $[64 \times 2]$ matrix ($128$ weights). Flattening `[10, 10, 2]` yields a compact **200-element vector** (2 learned summary features per board cell), which then connects to the **400-unit Policy Dense layer** via a $[200 \times 400]$ matrix ($80,000$ weights).
     - **Value Bottleneck (`vConv`):** Projects `[10, 10, 64]` $\to$ `[10, 10, 1]` via a $[64 \times 1]$ matrix ($64$ weights). Flattening yields a **100-element vector** (1 scalar territory score per cell), which feeds `[100 × 64]` ($6,400$ weights) $\to$ `[64 × 1]` ($64$ weights) $\to$ `tanh`.

#### Complete Parameter & Compute Breakdown (`TFConvNetwork`)

| Layer | Input Shape | Matrix Operation (`GEMM`) | Output Shape | Parameters | FLOPs / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `[B, 10, 10, 18]` | — | `[B, 10, 10, 18]` | `0` | — |
| **Conv1 + BN** | `[B, 10, 10, 18]` | `[100B, 162] × [162, 32]` | `[B, 10, 10, 32]` | `5,344` | `1.04M` |
| **Conv2 + BN** | `[B, 10, 10, 32]` | `[100B, 288] × [288, 64]` | `[B, 10, 10, 64]` | `18,752` | `3.69M` |
| **Conv3 + BN** | `[B, 10, 10, 64]` | `[100B, 576] × [576, 64]` | `[B, 10, 10, 64]` | `37,184` | `7.37M` |
| **Policy `1×1` Conv**| `[B, 10, 10, 64]` | `[100B, 64] × [64, 2]` | `[B, 10, 10, 2]` | `130` | `25.6K` |
| **Policy Dense** | `[B, 200]` | `[B, 200] × [200, 400]` | `[B, 400]` | `80,400` | `160.0K` |
| **Value `1×1` Conv** | `[B, 10, 10, 64]` | `[100B, 64] × [64, 1]` | `[B, 10, 10, 1]` | `65` | `12.8K` |
| **Value Dense 1** | `[B, 100]` | `[B, 100] × [100, 64]` | `[B, 64]` | `6,464` | `12.8K` |
| **Value Dense 2** | `[B, 64]` | `[B, 64] × [64, 1]` | `[B, 1]` | `65` | `128` |
| **TOTAL** | | | **400 Policy + 1 Value**| **148,404** (~594 KB)| **~12.3M FLOPs** |

### Why These Exact Matrix Sizes Fly on the M2 Ultra

1. **Entire Network Weights Fit in L2 Cache (`594 KB`):**
   - At `148,404` `float32` parameters (`~594 KB`), the *entire neural network weight set* fits inside the M2 Ultra's **on-chip cache hierarchy** (32MB–64MB L2 + System Level Cache). During MCTS search (where the network is queried 50 times per move across ~150 moves = 7,500 forward passes per game), weights never have to be fetched from main DRAM.
2. **SIMD Tile Alignment (`32`, `64`, `256`):**
   - Apple Silicon matrix coprocessors (AMX) and Metal SIMD groups operate on **32-wide and 64-wide register tiles**. Choosing filter depths of **`32 → 64 → 64`**, a value hidden layer of **`64`**, a MCTS leaf batch size of **`8`**, and a training mini-batch size of **`256`** ensures zero partial-tile padding waste during `GEMM` execution.
3. **Zero-Swap 500K Replay Buffer (`~6.8 GB` in 192GB Unified RAM):**
   - Storing `500,000` canonical experiences—each containing a `1,800`-float state (`7.2 KB`), a `400`-float MCTS visit distribution (`1.6 KB`), `chosenIndex`, and `valueTarget`—consumes less than **4% of the M2 Ultra's 192GB unified memory**, allowing instant random batch sampling (`batchSize = 256`, `replayRatio = 2x`) with zero disk I/O.

### Node.js Version & Native Backend Notes

```
⚠️ CRITICAL: TensorFlow.js native bindings require Node v20 LTS

Node v25 → C++ V8 ABI mismatch → @tensorflow/tfjs-node fails to load
Node v20 → Native C++ / Accelerate / Metal bindings load cleanly

Use: fnm use v20.20.2  (or nvm use 20.20.2)
```

### Batch Size Optimization on M2 Ultra

```
Batch Size │  Hardware Utilization │  Throughput & Gradient Behavior
───────────┼───────────────────────┼──────────────────────────────────────────
    1      │  ~5%  (overhead bound)│  High kernel-launch overhead, noisy grads
    8      │  ~45% (MCTS leaf opt) │  Sweet spot for parallel MCTS leaf evals
   32      │  ~65%                 │  Good for pure feedforward policy updates
  256      │  ~98% (AMX/GPU full)  │  Optimal for ConvNet + BatchNorm + Replay
```

**Recommendation:** Use `mctsBatchSize: 8` during tree search (balancing virtual-loss tree diversity against vectorization) and `batchSize: 256` with `tf.train.adam(0.001)` during replay-buffer gradient updates.

### Wall-Clock Processing Time Derivation (Why Self-Play is 96% of Cycle Time)

A common intuition from supervised deep learning is that **backpropagation** (gradient descent after the games finish) is the primary computational bottleneck, while playing the games is fast. In **AlphaZero / QuiescenceMCTS reinforcement learning**, the exact opposite is true: **~96% of wall-clock time is spent generating the 100 self-play games**, while **backpropagation takes only ~85 seconds (~3% of cycle time)** and **3-way evaluation takes ~25 seconds (~1%)**.

Here is the first-principles derivation measured on the Mac Studio M2 Ultra (`TF-B14` / `TF-B15`, `QuiescenceMCTS-50`, `100` games/cycle):

#### 1. Phase A — Dual-Side `QuiescenceMCTS-50` Self-Play (`~2,580s` = `~43.0 min`, `96.1%`)

In pure Dual-Side MCTS self-play, **both** Player 1 and Player 2 execute a 50-simulation Monte Carlo Tree Search before every move on the board:
- **Moves per cycle:**
  $$N_{\text{moves}} = 100\text{ games/cycle} \times \approx 130\text{ moves/game} = \mathbf{13,000\text{ moves/cycle}}$$
- **MCTS tree simulations per cycle:**
  $$N_{\text{sims}} = 13,000\text{ moves} \times 50\text{ simulations/move} = \mathbf{650,000\text{ MCTS simulations/cycle}}$$
- **Work performed at each of the `650,000` leaf expansions:**
  1. **Board Cloning & Tree Traversal (CPU / V8 JS):** Clone the $10 \times 10$ toroidal `GridManager` along the search path, enumerate all legal moves (`isValidMove` across up to 400 slots), and apply `placeTriangle` + `claimEnclosedTriangles`.
  2. **Quiescence Cascade Rollout (`quiescenceMode = 'greedy'`, depth $\le 5$):** Whenever a leaf move is non-enclosing (`claimed === 0`), `QuiescenceMCTS` scans all opponent replies using $O(1)$ `countWouldBeEnclosed` checks and greedily plays out multi-step opponent enclosure chains (up to 5 plies) so the value head evaluates the *post-cascade quiescent board* rather than a mid-tactical horizon trap.
  3. **Batched Neural Network Forward Passes (`mctsBatchSize = 8`):** Every 8 virtual-loss leaf nodes are grouped into a `[8, 10, 10, 18]` tensor and evaluated by `TFConvNetwork.runBatch()`:
     $$N_{\text{forward\_batches}} = \frac{650,000\text{ simulations}}{8\text{ leaves/batch}} = \mathbf{81,250\text{ ConvNet forward calls/cycle}}$$
     At `~12.3M FLOPs` per sample (`~98.4M FLOPs` per batch of 8), the neural network executes **`~8.0 TeraFLOPs`** of forward inference per cycle across `81,250` separate JS $\leftrightarrow$ C++ TF.js bridge calls, plus `650,000+` JavaScript board state clones and quiescence checks.
  - **Measured Throughput:** `~25.8 seconds/game` (`~198 ms/move` for 50 simulations $\approx \mathbf{3.96\text{ ms}}$ per QuiescenceMCTS simulation) $\implies$ **`~43.0 minutes` per 100-game cycle**.

#### 2. Phase B — Backpropagation & Experience Replay (`~85s` = `~1.4 min`, `3.1%`)

Once the 100 self-play games finish, the pipeline has collected $\approx 13,000$ canonical `(state, mctsPolicy, chosenMoveIdx, valueTarget)` training tuples:
- **Why Backpropagation is $30\times$ Faster than Self-Play:**
  - Instead of small `batchSize = 8` forward passes interleaved with sequential JavaScript game logic, backpropagation packs states into large **`batchSize = 256`** tensors (`[256, 10, 10, 18]`), saturating the M2 Ultra's AMX / Metal SIMD tiles at ~98% hardware utilization.
  - **Current-cycle gradient updates (`10` epochs):**
    $$\frac{13,000\text{ samples}}{256\text{ samples/batch}} \approx 51\text{ batches/epoch} \times 10\text{ epochs} = \mathbf{510\text{ gradient steps}}$$
  - **Sliding-Window Replay Buffer updates (`replayRatio = 2.0`):**
    $$2.0 \times 51\text{ batches} \approx \mathbf{102\text{ replay gradient steps}}$$
  - **Total Backprop Compute:** **`612` large-batch (`B = 256`) forward+backward steps** (`156,672` sample evaluations total, compared to `650,000` leaf evaluations during self-play, and executed in `612` bulk C++ kernel calls instead of `81,250` small calls!).
  - **Measured Runtime (`TF-B14` Cycles 1–10):** **`83.2s – 86.3s` (`~1.4 minutes`) per cycle**.

#### 3. Phase C — 3-Way Evaluation (`300` Games Total: `~25s` = `~0.4 min`, `0.8%`)

After weight updates, the model plays **300 full evaluation games** (`100` vs Random Raw, `100` vs Heuristic Raw Policy, `100` vs Heuristic +Safety Filter):
- Because evaluation tests the **1-ply policy network** (`predictMove()` with `exploration = 0`, 1 forward pass per AI move instead of 50 MCTS simulations), 300 games require only:
  $$300\text{ games} \times \approx 65\text{ AI moves/game} = \mathbf{19,500\text{ single-state forward passes}}$$
- Even including the $O(1)$ `countWouldBeEnclosed` + $5 \times 5$ `evaluateOpponentReplyRisk` checks on the top-$K=12$ moves in the `+Safety Filter` split (`< 0.4 ms/move`), all 300 evaluation games complete in **`~22–28 seconds`**.

#### Summary Table: Per-Cycle Time Budget (`100` Games/Cycle, `QuiescenceMCTS-50`)

| Cycle Phase | Operations per Cycle | Tensor Batch Size | Wall-Clock Time | Share of Cycle | Scaling Law |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. QuiescenceMCTS Self-Play** | `100` games, `13,000` moves, **`650,000` tree sims + cascades**, `81,250` net calls | `B = 8` | **`~43.0 min`** (`2,580s`) | **`96.1%`** | Linear in `games × mctsSims` |
| **2. Backpropagation + Replay** | `13,000` new samples (`10` iters) + `26,000` replay samples (`612` steps) | `B = 256` | **`~1.4 min`** (`85s`) | **`3.1%`** | Linear in `games × (iterations + replayRatio)` |
| **3. 3-Way Evaluation** | `300` games (`100` Rand, `100` Heur Raw, `100` Heur +Safety), `~19,500` net calls | `B = 1` | **`~0.4 min`** (`25s`) | **`0.8%`** | Linear in `evalGames` |
| **Total per Cycle** | **400 total games played** (`100` MCTS + `300` 1-ply eval) | — | **`~44.8 min`** (`2,690s`) | **`100.0%`** | **`~7.4 hours` for 10 cycles** |

> **Practical Takeaway for Experiment Design:** Because self-play dominates 96% of runtime, total training duration scales almost purely with `cycles × games × mctsSims`. Increasing training epochs (`--iterations 10 → 20`), increasing `--replay-ratio`, or expanding evaluation from 200 to 300 games (as in our 3-way evaluation split) adds less than **1–2% overhead** to total cycle time!

---

## 5. Parameters Explored {#5-parameters}

### Parameter Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Parameter Categories                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Hyperparameters                                   │
│  ──────────────────────────                                 │
│  • learningRate (α) — Step size for gradient descent        │
│  • iterations — Training epochs per cycle                   │
│  • batchSize — Samples per gradient update                 │
│  • momentum — Optimizer dampening                           │
│                                                             │
│  Exploration Parameters                                     │
│  ──────────────────────────                                 │
│  • exploration (ε) — Random move probability               │
│  • explorationDecay — Reduction over time                  │
│                                                             │
│  Self-Play Configuration                                    │
│  ──────────────────────────                                 │
│  • games — Games per training cycle                         │
│  • cycles — Number of training iterations                   │
│  • opponentMix — Distribution of opponent types             │
│  • roleAlternation — Whether AI plays both sides           │
│                                                             │
│  Reward Engineering                                        │
│  ─────────────────────                                       │
│  • clusterRewardFactor — Bonus for territory clustering     │
│  • clusterLambda — TD(λ) decay rate                         │
│  • clusterGamma — Discount factor for future rewards        │
│                                                             │
│  Technical Options                                          │
│  ─────────────────────                                       │
│  • gradientIsolation — Isolate gradient to chosen move    │
│  • enableClusterReward — Use TD(λ) rewards                  │
│  • startFrom — Initialize from existing network            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Learning Rate (α)

**What it does:** Controls how much the network weights change with each gradient update.

```
                    High Learning Rate (0.1+)
                            │
                            ▼
         ┌─────────────────────────────────────┐
         │  UNSTABLE - oscillates, diverges   │
         │  Weights bounce around, never settle│
         └─────────────────────────────────────┘
         
         ┌─────────────────────────────────────┐
         │  SWEET SPOT: 0.01 - 0.05           │
         │  - Smooth convergence              │
         │  - Good exploration                 │
         │  - Stable learning                  │
         └─────────────────────────────────────┘
         
         ┌─────────────────────────────────────┐
         │  TOO SLOW: 0.001 - 0.005           │
         │  - Takes forever to converge       │
         │  - Gets stuck in local optima       │
         │  - Never reaches peak performance   │
         └─────────────────────────────────────┘
                            ▲
                    Low Learning Rate (0.001-)
```

| Value | Observed Behavior | Best For |
|-------|-------------------|----------|
| 0.001 | Very slow convergence, risk of local minima | Fine-tuning trained models |
| **0.01** | Smooth learning, stable convergence | **Default choice** |
| 0.05 | Faster learning, slight oscillation | Fresh training |
| 0.1+ | Diverges, unstable, weights explode | Avoid |
**Key insight:** Learning rate affects *convergence speed*, not final performance. A good architecture at lr=0.01 eventually reaches the same peak as at lr=0.05.

**Momentum:** We use momentum=0.9 to smooth gradient updates.

```
Momentum accumulates past gradients:

velocity = momentum × velocity + gradient
weight = weight - learningRate × velocity

Effect: Dampens oscillations, accelerates convergence
```

---

## 6. Training Strategies {#6-training-strategies}

### Strategy 1: Pure Self-Play

```
┌─────────────────────────────────────────────────────────────┐
│                    Pure Self-Play                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Network plays against itself                             │
│   ─────────────────────────────                             │
│   P1: Network (exploration=0.3)                             │
│   P2: Network (exploration=0.3)                             │
│                                                             │
│   Pros: No external dependency, improves together            │
│   Cons: Can converge to weird equilibria, no strategic      │
│         pressure from competent opponents                   │
│                                                             │
│   Result: 37-49% vs Random, 0% vs Heuristic                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiments TF-B1_v3, TF-B1_pure:** Pure self-play baselines
- Final vs Random: 37-49%
- Final vs Heuristic: 0%

The network learned basic legality but no strategic depth.

### Strategy 2: Mixed Opponents ("Learn from the Master" vs Curriculum Learning)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           WHY OPPONENT MIX FAILED IN STAGE 1 BUT WORKED IN STAGE 4           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Attempt A: "Learn from the Master" on Feedforward Net (exp21 & TF-B2)       │
│  ─────────────────────────────────────────────────────────────────────       │
│  • Config: 40%–70% Heuristic, 20%–30% Network, 10%–30% Random                │
│  • Hypothesis: Playing against the Heuristic master will expose the AI       │
│                to strong territorial play and force it to adapt.             │
│  • What Actually Happened:                                                   │
│      - AI lost 100% of games vs Heuristic (avg score: AI = 4, Heur = 170).   │
│      - In Policy Gradient, a 100% loss rate means EVERY move made against    │
│        Heuristic receives a negative gradient (-1)!                          │
│      - Worse: exp21's vs-Random win rate COLLAPSED from 91% down to 49%!     │
│        Constant negative punishment erased the network's existing intuition. │
│                                                                              │
│  Attempt B: 50/50 Mixed Opponents + Quiescence MCTS (TF-B10 & TF-B12)        │
│  ────────────────────────────────────────────────────────────────────        │
│  • Config: 50% QuiescenceMCTS-50 Self-Play + 50% Heuristic Opponent          │
│  • Why It Succeeded Here (0% ──► 63% vs Heuristic in TF-B12!):               │
│      1. Quiescence MCTS lookahead allowed the AI to start winning ~20–60%    │
│         of training games against Heuristic, providing POSITIVE target       │
│         trajectories alongside losses.                                       │
│      2. Even when the AI lost a game against Heuristic, MCTS policy targets  │
│         π_MCTS trained the Policy Head toward the locally best defensive     │
│         replies discovered by the search tree, while only the Value Head V   │
│         absorbed the -1 outcome!                                             │
│                                                                              │
│  Attempt C: Pure Dual-Side Quiescence MCTS Self-Play (TF-B14 & TF-B15)       │
│  ─────────────────────────────────────────────────────────────────────       │
│  • Config: 100% Dual-Side QuiescenceMCTS-50 (Zero Heuristic in training!)    │
│  • Result: 90% (TF-B14) ──► 99% Raw / 100% +Safety (TF-B15) vs Heuristic!    │
│  • Insight: Once the search tree itself models cascade dynamics via          │
│    Quiescence rollouts, pure self-play surpasses mixed-opponent training     │
│    because BOTH sides of every training game generate high-entropy,          │
│    50-simulation tactical search targets!                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Experiments `exp21` (`brain.js`) & `TF-B2` (`TF.js`):** Mixed opponent feedforward training
- `exp21` ("Learn from the Master"): Degraded from `91%` to **`49%` vs Random**, **`0%` vs Heuristic**
- `TF-B2_v3`, `TF-B2_mixed`: **`43–51%` vs Random**, **`0%` vs Heuristic**

**Key Systems Lesson:** In pure Policy Gradient RL, **an opponent that beats you 100% of the time teaches you nothing**—it only injects uniform negative gradients that destroy existing policy structure. Curriculum opponents (`Heuristic`) only become useful once your search architecture (`QuiescenceMCTS`) is strong enough to survive their tactical traps and separate move quality ($\pi_{\text{MCTS}}$) from game outcome ($z$).

### Strategy 3: Cluster Reward Shaping ($TD(\lambda)$) vs True Value Critics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 CLUSTER REWARD SHAPING (TD-λ) vs VALUE HEAD                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Traditional Policy Gradient: Reward only at game end (Ply 170)             │
│   TD(λ) Shaped Reward: Blend immediate cluster growth with final outcome     │
│                                                                              │
│   Shaped Reward Formula (exp17, exp22, TF-B3):                               │
│   ────────────────────────────────────────────                               │
│   • r_immediate = 0.3 × enclosed_count + 0.1 × max_cluster_delta             │
│   • TD(λ) trace: λ = 0.9, γ = 0.99                                           │
│                                                                              │
│   Results Across Engines:                                                    │
│   ───────────────────────                                                    │
│   • exp17b / exp22 (brain.js): +5% boost vs Random (70% → 75%), 0% vs Heur   │
│   • TF-B3 (TF.js feedforward): 43–52% vs Random, 0% vs Heuristic             │
│                                                                              │
│   Why Hand-Crafted Reward Shaping Hit a Ceiling:                             │
│   ──────────────────────────────────────────────                             │
│   • Rewarding immediate cluster growth (+0.1 per triangle) encourages        │
│     greedy outward expansion ("building long thin walls").                   │
│   • Against HeuristicPlayer, building long unclosed walls is suicidal:       │
│     Heuristic waits until the wall is 90% complete, steps in to close the    │
│     last gap, and steals the entire cavity + bonus turns!                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Finding:** Hand-crafted intermediate rewards ($TD(\lambda)$ cluster bonuses) help beat naive `Random` players by encouraging piece connectivity, but **backfire against tactical opponents** because they reward *greedy local expansion* over *defensive cavity prevention*. Replacing hand-crafted reward shaping with a learned **Value Head ($V(s) \in [-1, +1]$) trained on pure terminal game outcomes ($z \in \{-1, 0, +1\}$) + Quiescence MCTS** eliminated this bias completely.

### Strategy 4: Optimized Hyperparameters

```
┌─────────────────────────────────────────────────────────────┐
│                 Optimized Training                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Key changes from baseline:                               │
│   ──────────────────────────                               │
│   • learningRate: 0.05 → 0.01 (more stable)               │
│   • iterations: 10 → 100 (more thorough)                   │
│   • batchSize: 256 → 32 (better gradients)                 │
│   • gradientIsolation: true (cleaner updates)             │
│                                                             │
│   Training time: ~35 min for 10 cycles                     │
│                                                             │
│   Result: 47% vs Random (best TF.js result)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Experiment TF-B1_optimized:** Optimized hyperparameters
- Final vs Random: 47%
- Final vs Heuristic: 0%

**Key lesson:** Hyperparameter tuning improves performance but doesn't change the fundamental capability ceiling.

---

## 7. Critical Bugs, Forensic Audits, and Why They Matter {#7-critical-bugs}

### Bug 1: Self-Play Perspective Mismatch (exp19)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Self-play: P1 wins 100% of games                          │
│   vs Random: Only 50-55% win rate                           │
│                                                             │
│   Statistical impossibility in a fair game!                 │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** P2 was using P1's policy to select moves.

```javascript
// WRONG: P2 uses P1's network output
// Both players see the same state, same output
move = getAIMove(network, grid, exploration);  // Always from P1's view

// CORRECT: Each player uses their own perspective
const state = getStateFixed(grid, currentPlayer);  // Fixed perspective
move = getAIMove(network, grid, exploration, aiPlayerNumber);
```

**How we found it:** Self-play win rate histogram showed P1 winning every single game. In a fair game, each side should win ~50%.

**Generalizable Principle (Canonical Ego-Centric State):** In symmetric multi-agent RL, the neural network must always view the state from the active agent's perspective (`+1` = self, `-1` = opponent), never in global player coordinates (`Player 1` vs `Player 2`).

### Bug 2: Reward Signal Dilution (TF.js Migration)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   brain.js: 91% vs Random                                   │
│   TF.js:    49% vs Random                                   │
│                                                             │
│   Same architecture, same data — why the drop?             │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Averaged reward instead of chosen move reward.

```javascript
// WRONG (TF.js bug)
const rawImmediateReward = vector.reduce((sum, v) => sum + v, 0) / 400;
// Average of all 400 values, including ~350 illegal moves at 0.01
// Win signal (0.99) diluted to ~0.28

// CORRECT (original brain.js)
const chosenMoveReward = vector[chosenMoveIndex];
// Only the chosen move gets the reward signal
```

**How we found it:** Compared training logs side-by-side. The win signal magnitude was visibly different: 0.99 vs 0.28.

**Generalizable Principle (Action-Specific Credit Assignment):** Only the action actually executed (or the target MCTS distribution) should drive the policy gradient; averaging over unselected outputs destroys signal-to-noise ratio.

### Bug 3: Heuristic Player Number (TF Baselines)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Heuristic scores 173/173 (max) every game                 │
│   AI scores 3-4 points every game                           │
│                                                             │
│   Why does Heuristic always get perfect score?             │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** HeuristicPlayer hardcoded to `playerNumber=2`.

```javascript
// WRONG
const heuristicP2 = new HeuristicPlayer(2, {...});
// When AI plays as P2, opponent should be P1!
// Both players think they're P2 → chaos

// CORRECT: Create heuristic with correct opponent number
const opponentNumber = (aiPlayerNumber === 1) ? 2 : 1;
const heuristicOpponent = new HeuristicPlayer(opponentNumber, {...});
```

**How we found it:** Audited game scores. Heuristic scoring 173/173 every game was statistically suspicious.

**Generalizable Principle (Role-Swap Invariance):** Whenever an evaluation harness alternates first-mover and second-mover seats for fairness, opponent identity must be dynamically bound to `(3 - aiPlayerNumber)`.

### Bug 4: Win Rate Measurement (exp12)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Training reports ~70% win rate                           │
│   Evaluation reports ~50%                                 │
│                                                             │
│   Which is right?                                          │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Player swap for fairness, but win counting by slot number.

```javascript
// WRONG
const aiPlayer = (g % 2 === 0) ? 1 : 2;  // AI alternates roles
...
if (winner === 1) wins[1]++;  // Counts slot, not identity

// CORRECT: Track wins by AI identity, not slot number
results.wins = { ai: 0, random: 0, draws: 0 };
if (winner === aiPlayer) results.wins.ai++;
```

**How we found it:** Noticed training and evaluation giving different results for the same opponent.

**Generalizable Principle (Decouple Seat Index from Agent Identity):** Metrics must always be keyed by agent identity (`ai` vs `opponent`), while separately tracking role-split performance (`as P1` vs `as P2`) to detect first-mover bias.

### Bug 5: MCTS Value Perspective Mismatch (TF-B8)

```
┌─────────────────────────────────────────────────────────────┐
│                   What We Saw                                │
├─────────────────────────────────────────────────────────────┤
│   Training converges smoothly (loss: 0.99 → 1.00)          │
│   But win rate degrades: 40% → 30% → 26%                   │
│                                                             │
│   Network IS learning — just learning to LOSE!              │
└─────────────────────────────────────────────────────────────┘
```

**Root cause:** Neural network returns values from the perspective of the player whose turn it is at the leaf node, but MCTS backpropagation expected values from the root player's perspective.

```javascript
// WRONG: Pass NN value directly to backpropagation
const { value: nnValue } = await this.network.run(leafState);
leaf.expand(leafPolicy);
leafValue = nnValue;  // From leaf.player's perspective!
this._backpropagate(path, leafValue, rootPlayer);  // Expects rootPlayer's perspective!

// CORRECT: Flip value if from opponent's perspective
const valueFromRootPerspective = leaf.player === rootPlayer ? leafValue : -leafValue;
this._backpropagate(path, valueFromRootPerspective, rootPlayer);
```

**Why This Caused 30% Win Rate:**
- When P2's position was bad (NN returns -0.7), search thought it was good (+0.7)
- "Bad" positions got high Q-values → high visit counts → trained as good moves
- Network learned: "Seek positions where NN outputs negative values"
- Result: systematic losing behavior

**How we found it:** TF-B8 showed systematic degradation pattern:
- Started: 40% vs Random
- Peak: 42% (cycle 2)
- Worst: 26% (cycle 6)
- Final: 30% (cycle 10)

The smooth loss curve + consistent degradation = inverted learning signal.

**Key Insight:** "Consistently getting worse is more encouraging than consistently remaining random" — it means the learning mechanism works, just the signal was flipped.

---

### Part II: The Forensic Cross-Codebase Audit (Bugs 6–11)

Following the completion of `TF-B8-V2` (which still oscillated between 15% and 54% vs Random and scored 0% vs Heuristic despite fixing Bug 5), we conducted a complete line-by-line audit across the **Browser Game (`game.js`, `AIPlayer.js`)**, **Core Engine (`src/core/GridManager.js`)**, **MCTS & Training (`src/server/MCTS.cjs`, `src/server/train_tfjs.cjs`, `src/server/ReplayBuffer.cjs`)**, and **Evaluation (`eval_tfjs.cjs`, `scripts/evaluate.cjs`)**.

We uncovered **six severe systemic bugs** spanning game geometry, tree search physics, tensor encoding, and gradient calculus:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 THE 6 SYSTEMIC BUGS UNCOVERED IN PHASE 5                     │
├──────────┬───────────────────────────────────┬───────────────────────────────┤
│ Bug ID   │ Component                         │ Systemic Impact               │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 6    │ Diagonal Overlap Check Inverted   │ Server & MCTS played a mutated│
│          │ (`src/core/GridManager.js`)       │ game where triangles crossed! │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 7    │ MCTS Blind to Enclosures & Bonus  │ MCTS tree never claimed cells │
│          │ Turns (`MCTSNode.expand()`)       │ or granted extra turns!       │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 8    │ Canonical Encoding Dropped 2nd    │ Full cells hid 50% of their   │
│          │ Triangle (`getStateCanonical`)    │ triangles from the ConvNet!   │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 9    │ ReplayBuffer Zeroed Value Targets │ 67% of batches trained the    │
│          │ (`ReplayBuffer.cjs`)              │ Value Head to predict 0 (draw)│
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 10   │ Dual-Perspective Gradient Clash   │ Opponent moves in self-play   │
│          │ (`train_tfjs.cjs`)                │ destabilized role updates     │
├──────────┼───────────────────────────────────┼───────────────────────────────┤
│ Bug 11   │ 1/400 Policy Loss Dilution &      │ Policy gradients 400× too weak│
│          │ Value `tanh` Saturation           │ (`0.45–0.54`); Value locked!  │
└──────────┴───────────────────────────────────┴───────────────────────────────┘
```

#### Bug 6: Inverted Diagonal Overlap Rule (Engine Split-Brain)

**Geometry of a Triantica Cell:**
Each cell `[x, y]` is split by one of two diagonals:
- **Group 0 (`╱` slash diagonal):** Rotations `0` (Top-Left `◤`) and `2` (Bottom-Right `◢`)
- **Group 1 (`╲` backslash diagonal):** Rotations `1` (Bottom-Left `◣`) and `3` (Top-Right `◥`)

```
       GROUP 0 (╱ Diagonal)                  GROUP 1 (╲ Diagonal)
   ┌───────────┐   ┌───────────┐         ┌───────────┐   ┌───────────┐
   │ ◤       ╱ │   │         ╱ │         │ ╲         │   │ ╲       ◥ │
   │   0   ╱   │ + │       ╱   │ = FULL  │   ╲       │ + │   ╲   3   │ = FULL
   │     ╱     │   │     ╱   2 │   CELL  │ 1   ╲     │   │     ╲     │   CELL
   │   ╱       │   │   ╱       │         │ ◣     ╲   │   │       ╲   │
   └───────────┘   └───────────┘         └───────────┘   └───────────┘
   Valid 2nd triangle: SAME diagonal group (newGroup === existingGroup)
                       OPPOSITE rotation   (rot !== existingRot)
```

**What Was Wrong:**
When `GridManager` was extracted to `src/core/GridManager.js` for server training, the boolean negation `!` was accidentally dropped from the overlap check:
```javascript
// WRONG (in src/core/GridManager.js & game.js:isValidMove):
if (newGroup === existingGroup || cell.triangles[0].rotation === rotation) {
  return false; // ❌ Forbids sharing the diagonal; forces 50% physical overlap!
}

// CORRECT (Consolidated in GridManager.isValidMove & placeTriangle):
const existingRot = cell.triangles[0].rotation;
const existingGroup = this.getDiagonalGroup(existingRot);
const newGroup = this.getDiagonalGroup(rotation);
if (newGroup !== existingGroup || existingRot === rotation) {
  return false; // ✅ Must share the diagonal and have opposite rotation
}
```
**Why It Mattered:** Every server training run (`TF-B1` through `TF-B8-V2`) and MCTS search was playing a **mutated game** where two triangles in the same cell had to cross diagonals (overlapping by 50%), while browser `game.js:placeTriangle` enforced the real rule! Furthermore, we consolidated `getAdjacentTriangles()` so two triangles in the same cell only form a contiguous cluster if they share the diagonal edge (`newGroup === existingGroup && t.rotation !== rotation`).

#### Bug 7: MCTS Was Completely Blind to Enclosures & Extra Turns

**What Was Wrong:**
In `src/server/MCTS.cjs`, `MCTSNode.expand()` advanced the board state using raw `placeTriangle()` and unconditionally flipped `currentPlayer`:
```javascript
// WRONG (MCTSNode.expand before fix):
const newGrid = this.grid.clone();
newGrid.placeTriangle(move.x, move.y, move.rotation, this.player);
newGrid.currentPlayer = this.player === 1 ? 2 : 1;
// ❌ Never called claimEnclosedTriangles(this.player)!
// ❌ Never granted extraTurns += Math.floor((1 + claimed) / 2)!
```
**Why It Mattered:** Inside MCTS lookahead, **enclosing territory and earning bonus turns literally did not exist!** MCTS could not see that completing a boundary captures enclosed triangles and grants an immediate extra turn, nor could it see when it left an enclosure combo open for the `HeuristicPlayer`.
- **The Fix (`grid.executeMove`):** We consolidated move placement, `claimEnclosedTriangles(pl)`, and `extraTurns += Math.floor((1 + claimed) / 2)` into a single method `GridManager.prototype.executeMove(x, y, rotation, player)` and called it everywhere (`MCTSNode.expand`, `train_tfjs.cjs`, `GameSession.js`, and evaluation scripts). Immediately after fixing this, AI cluster sizes jumped by **+60% to +85%** (from ~19 to ~35–40)!

#### Bug 8: Canonical 18-Channel Encoding Dropped the 2nd Triangle in Every Full Cell

**What Was Wrong:**
`getStateCanonical` originally iterated `for (let group = 0; group < 2; group++)` and called `cell.triangles.find(t => grid.getDiagonalGroup(t.rotation) === group)`. Once Bug 6 was fixed, both triangles in a full cell belong to the **same** diagonal group (e.g., rotations `0` and `2` are both Group 0). Thus `.find()` only ever returned the first triangle and left the second 9-channel slot empty—**blinding the ConvNet to 50% of all triangles in completed cells!**
- **The Fix (`grid.getCanonicalState`):** We indexed the two 9-channel slots by **rotation half** (`Slot 0: rotation < 2` i.e. $\{0, 1\}$, `Slot 1: rotation >= 2` i.e. $\{2, 3\}$). Since a valid cell can hold at most one triangle from $\{0, 1\}$ and at most one from $\{2, 3\}$, this guarantees **zero collisions, 100% visibility of both triangles, and strict move-order invariance**.

#### Bug 9: Replay Buffer Zeroed Out All Historical Value Targets

**What Was Wrong:**
In `train_tfjs.cjs`, `ReplayBuffer` stored `(input, output, chosenIndex)` but omitted `valueTarget`. During replay training:
```javascript
// WRONG (train_tfjs.cjs before fix):
const replayValueTargets = new Array(replayBatchSize).fill(0); // ❌ Forces Value Head to 0!
```
With `replayRatio = 2.0`, **67% of all gradient updates trained the Value Head to predict `0.0` (draw) on every board position**, destroying its ability to evaluate winning vs losing states!
- **The Fix:** Updated `ReplayBuffer.cjs` to store and sample `valueTarget` (`+1`, `0`, `-1`) alongside state and policy.

#### Bug 10: Dual-Perspective Gradient Conflict in MCTS Self-Play

**What Was Wrong:**
While the non-MCTS training loop filtered training samples to the designated AI player (`if (step.player !== aiPlayerNumber) continue;`), the MCTS training loop pushed **both** players' moves from the same self-play game into the training batch.
- **The Fix:** Enforced `if (step.player !== aiPlayerNumber) continue;` (with `aiPlayerNumber` alternating between P1 and P2 every game) and added explicit **P1 (First Mover) vs P2 (Second Mover) win-share tracking** across all training logs and dashboard charts.

#### Bug 11: 1/400 Policy Loss Dilution & Value Head `tanh` Saturation (The `TF-B8` → `TF-B9` Breakthrough)

**What We Saw After Re-Running `TF-B8` With Bugs 6–10 Fixed:**
Win rate vs Random stabilized at **43.0% avg (47.0% in Cycle 10)** with zero collapses and large clusters (~37.1 mean), yet the AI still didn't dominate Random (expected 70%+). Inspecting the trained `TF-B8_network.json` weights revealed why:
- Policy Head outputs across all 400 moves were stuck between **`0.450` and `0.545`** (barely moved from `0.500` random initialization!).
- Value Head output was saturated at **`-1.0000`** on every board state!

**Root Cause (Mathematical Proof):**
1. **1/400 Policy Gradient Dilution:**
   In `TFConvNetwork.trainBatchWithIsolation()`, policy loss was computed on the `[256, 400]` tensor using `.mean()`:
   $$\mathcal{L}_{\text{policy}}^{\text{bug}} = \frac{1}{256 \times \mathbf{400}} \sum_{b=1}^{256} \sum_{a=1}^{400} \text{CE}(\pi_{b,a}, \hat{p}_{b,a})$$
   Because the MCTS target distribution $\sum_{a=1}^{400} \pi_{b,a} = 1.0$ already sums to $1$ across the 400 actions, dividing by `400` again **diluted the policy gradient by $400\times$**!
2. **Value Head Saturation (`tanh` + `LR=0.05, momentum=0.9`):**
   Meanwhile, the Value Head `[256, 1]` MSE loss was only divided by `256`—receiving a **$400\times$ stronger gradient** than the Policy Head! Combined with an aggressive `momentum(0.05, 0.9)` optimizer, the first gradient step pushed the Value Head pre-activation $z$ to $|z| > 8$, where $\tanh(z) = \pm 1.0$ and the derivative $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z) \approx 0$. Once saturated at `-1.0`, the Value Head's gradient vanished permanently!

```javascript
// WRONG (Diluted Policy Loss + Saturated Value Head):
const bce = policyTargetTensor.mul(pClip.log())
  .add(tf.scalar(1).sub(policyTargetTensor).mul(tf.scalar(1).sub(pClip).log()))
  .neg().mean(); // ❌ Divides by (BatchSize * 400) -> 400x weaker gradient!
const optimizer = tf.train.momentum(0.05, 0.9); // ❌ Overshoots tanh into saturation!

// CORRECT (Fixed in TF-B9):
const logits = pClip.div(tf.scalar(1).sub(pClip)).log();
const logProbs = tf.logSoftmax(logits, -1);
// ✅ Sum over the 400 actions (-1), then average over the batch (256):
const policyLoss = policyTargetTensor.mul(logProbs).neg().sum(-1).mean();
// ✅ Persistent Adam optimizer (LR=0.001) prevents tanh saturation:
if (!this._optimizer || this._optimizerLR !== adamLR) {
  this._optimizer = tf.train.adam(adamLR);
  this._optimizerLR = adamLR;
}
```

**Live Verification Impact:**
In a 20-game live verification test immediately after applying Bug 11's fix:
- Policy Head outputs cleanly separated legal high-visit MCTS moves (**`0.985`**) from poor/illegal moves (**`0.000`**).
- Value Head remained unsaturated and calibrated (`+0.2538` on mid-game boards).
- Win rate vs Random jumped from `35–42%` to **`56.0%`** (`52% as P1`, `60% as P2`) in Cycle 1 alone!

#### Bug 12: Browser Train/Test Distribution Mismatch (`game.js` 90% Heuristic Override)

**What We Saw (2026-10-22 Night Audit):**
When watching "AI vs AI" in the browser (`game.js`), both AIs appeared to play balanced, highly tactical games with large enclosure cascades—yet when we tested the exact same trained `brain.js` network in the server evaluation harness (`evaluate.cjs`), it lost `0–100` against `HeuristicPlayer`!

```
┌──────────────────────────────────────────────────────────────────────────────┐
│          THE BROWSER "ILLUSION OF COMPETENCE" (TRAIN/TEST MISMATCH)          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Move Selection Source     │ Browser AI (game.js)  │ Server Training/Eval    │
│  ──────────────────────────┼───────────────────────┼─────────────────────────│
│  Neural Network Prediction │  10% of moves! ❌      │  80% (Train) / 100% Eval│
│  Hardcoded Greedy Heuristic│  90% of moves! ❌      │  10% (Train) /   0% Eval│
│  Random Exploration        │   0% of moves         │  10% (Train) /   0% Eval│
│                                                                              │
│  Impact: Watching "AI vs AI" in the browser was actually watching            │
│          Heuristic vs Heuristic 90% of the time—masking the network's        │
│          true behavior and hiding the P2 snowball asymmetry!                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

- **The Fix (`76cece8` & Phase 6 Consolidation):** Aligned browser `AIPlayer` move selection directly with the pure neural network policy (`100%` network prediction during evaluation/play, with optional 1-ply safety filter toggle). Once the 90% heuristic crutch was removed, the browser immediately revealed the true state of the network—and later allowed us to verify `TF-B15`'s genuine `99%` raw / `100%` +Safety mastery in real time.

#### Bug 13: The `eval_tfjs.cjs` Role-Swap Bug (Why Every Model Looked Locked at `50.0%`)

**What We Saw (2026-10-28 Audit of `TF-B11` / `TF-B12` Standalone Evals):**
During training inside `train_tfjs.cjs`, `TF-B12` achieved **`100.0%` vs Random** and **`63.0%` vs Heuristic**. Yet when we ran the standalone CLI evaluator `eval_tfjs.cjs` on the saved checkpoint, it reported **exactly `50.0%` (`50W / 50L`)** across every single model—even when evaluating against `RandomPlayer`!

**Root Cause:**
In `eval_tfjs.cjs`, the evaluation loop alternated `aiPlayer = (i % 2 === 0) ? 1 : 2` for seat fairness, but instantiated the game session with fixed player objects `(aiPlayerObj, opponentObj)` without swapping their seats when `aiPlayer === 2`:
```javascript
// WRONG (in standalone eval_tfjs.cjs before fix):
const aiPlayer = (i % 2 === 0) ? 1 : 2;
// ❌ Passed (ai, opponent) in fixed order regardless of aiPlayer!
// In odd games (aiPlayer === 2), the AI played as P1 while the harness
// credited P2's wins to the AI!
```
Because P1/P2 wins were inverted on every odd-numbered game (`50` games normal + `50` games inverted), any deterministic win rate $w$ was averaged with $(1 - w)$:
$$\text{Reported Win Rate} = \frac{1}{2} w + \frac{1}{2}(1 - w) = \mathbf{50.0\%}$$
- **The Fix:** Consolidated all evaluation logic (`train_tfjs.cjs`, `eval_tfjs.cjs`, and `scripts/evaluate.cjs`) to bind player seats strictly by `aiPlayerNumber`, enforce **2-ply random opening seeding** (`1` random legal move per player so 100-game evals test 100 distinct board openings rather than 1 repeated deterministic trajectory), and report explicit **`as P1` vs `as P2` win splits**.

## 8. The Heuristic Wall: Understanding Architectural Limits & How We Broke It {#8-heuristic-wall}

### The Experiment That Changed Everything

After 22 brain.js experiments and 10+ TF.js experiments, a clear pattern emerged:

```
┌─────────────────────────────────────────────────────────────┐
│                   Performance Summary                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Opponent      │  Best Win Rate  │  All Experiments         │
│   ──────────────┼─────────────────┼────────────────────────  │
│   Random        │     91%        │      27-91%              │
│   Heuristic     │      0%        │       0%                 │
│                                                             │
│   Conclusion: The architecture cannot beat strategic play   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Why Can't a Feedforward Network Beat Heuristic?

```
┌─────────────────────────────────────────────────────────────┐
│                    Heuristic Strategy                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   For each legal move:                                      │
│   1. Simulate placement on a cloned grid                    │
│   2. Count immediate enclosures (+100 pts per triangle!)    │
│   3. Penalize opponent counter-enclosures (-80 pts each!)   │
│   4. Score cluster connections (+10 pts per triangle)       │
│   5. Add center bias + small tie-breaking jitter            │
│                                                             │
│   Result: Deterministic 2-ply tactical lookahead that       │
│   NEVER misses an enclosure and NEVER leaves a 1-move hole. │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Feedforward Network                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Flatten 10×10×4 board into 401 unstructured numbers    │
│   2. Multiply by a single dense matrix [401 × 200]          │
│   3. Output probability for each move [200 × 400]           │
│                                                             │
│   No spatial locality. No position critic. No lookahead.    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Two Architectural Blind Spots: Why Even Vanilla AlphaZero MCTS Scored `0%`

When we first diagnosed the Heuristic Wall after `exp22` and `TF-B4`, our initial hypothesis was simple: *"Add a Convolutional Actor-Critic (`TF-B5`) and standard AlphaZero MCTS (`TF-B8`–`TF-B10`), and the wall will fall."*

**We were wrong.** Even after building the `10×10×18` ConvNet and fixing every gradient bug in standard MCTS-50 (`TF-B9` and `TF-B10`), our win rate against `Random` climbed to **`87.0%`**, yet our win rate against `HeuristicPlayer` remained **locked at `0.0%`!**

Why did both a `2-Head ConvNet` (`TF-B5`) AND `Vanilla AlphaZero MCTS-50` (`TF-B10`) score `0%` against `Heuristic`? Because of two distinct domain mechanics in Triantica:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     BLIND SPOT #1: SPATIAL TRANSLATION BLINDNESS (FLAT vs CONV vs TORUS)     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Consider a 3-sided "almost enclosed" cavity around cell (x, y):             │
│                                                                              │
│  • In a Flat 401-Vector (Stage 1: brain.js / TF-B1–B4):                      │
│    Shifting that cavity 1 cell to the right changes the active input indices │
│    from [140, 141, 180] to [144, 145, 184]. Because Dense layers have no     │
│    weight sharing across space, the network must re-learn the exact same     │
│    cavity geometry 100 separate times across all 100 cells!                  │
│                                                                              │
│  • In a Zero-Padded 3×3 Conv2D (Stage 2–4: TF-B5–TF-B14):                    │
│    A 3×3 convolutional kernel shares weights across the interior 8×8 cells,  │
│    recognizing cavities anywhere in the center. HOWEVER, `padding='same'`    │
│    fills the 1-cell border around the 10×10 grid with ZEROS. To the ConvNet, │
│    the board edge looks like an impenetrable brick wall, blinding it to      │
│    enclosures that wrap across the torus seam (x=9 ↔ x=0, y=9 ↔ y=0)!       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│     BLIND SPOT #2: THE MCTS HORIZON EFFECT ON FORCED ENCLOSURE CASCADES      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Suppose the AI considers Move A at the root of an MCTS-50 search:           │
│  Move A is a non-enclosing move that accidentally leaves a 1-move cavity.    │
│                                                                              │
│  What happens in Vanilla MCTS-50 (TF-B8, TF-B9, TF-B10)?                     │
│  ───────────────────────────────────────────────────────                     │
│  1. MCTS expands Move A as a new leaf node s_leaf and queries Value Head V.  │
│  2. At s_leaf, the cascade HAS NOT HAPPENED YET! The board still looks       │
│     balanced in piece count, so the Value Head returns V(s_leaf) ≈ +0.10.    │
│  3. Even on subsequent simulations below Move A, the priors P(s,a) come      │
│     from an untrained/early policy network that explores 50 legal replies    │
│     semi-uniformly—missing the 5-step forced cascade sequence!               │
│  4. Then in the actual game, HeuristicPlayer steps up, spots the cavity      │
│     with 100% determinism, chains 4 extra turns in a row, and wipes out      │
│     the board before the AI ever gets another turn!                          │
│                                                                              │
│  How Quiescence MCTS (TF-B12–TF-B15) Solves the Horizon Effect:              │
│  ──────────────────────────────────────────────────────────────              │
│  • Whenever an MCTS leaf expansion lands on a non-enclosing move that        │
│    passes the turn while leaving opponent enclosures on the board,           │
│    QuiescenceMCTS immediately plays out the opponent's greedy enclosure      │
│    cascade (up to depth 5) BEFORE calling the Value Head V(s_quiet), AND     │
│    subtracts an explicit cascade risk penalty (-0.20 × cascaded claims).     │
│  • Result: Blunder leaves immediately evaluate to V ≈ -0.95 on Visit #1!     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### The 5-Stage Architectural Climb That Broke the Wall (`0% → 63% → 90% → 100%`)

To turn this diagnosis into a complete engineering solution, we progressed through five distinct architectural stages—each solving the exact bottleneck exposed by the stage before it (see [Chapter 11](#11-optimization-roadmap) for full mathematical derivations and tensor traces):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     STAGE-BY-STAGE BREAKTHROUGH PROGRESSION (vs Random & vs Heuristic)       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Feedforward Policy Gradient (exp1–exp22, TF-B1–TF-B4)              │
│  • Architecture: 401 → [200] → 400 Dense MLP + TD(λ) shaped rewards          │
│  • Telemetry:    91% vs Random (exp20), 0.0% vs Heuristic (Score: 4 vs 170)  │
│  • Bottleneck:   No spatial weight sharing, no value critic, no lookahead.   │
│                                                                              │
│  Stage 2: 2-Head Spatial ConvNet Actor-Critic (TF-B5 & TF-B6)                │
│  • Architecture: 10×10×18 Canonical Tensor → 3× Conv2D(64) → Policy + Value  │
│  • Telemetry:    50% vs Random, 0.0% vs Heuristic (Loss sawtooth: 1.91→1.38) │
│  • Bottleneck:   1-ply pattern recognition cannot foresee multi-turn traps.  │
│                                                                              │
│  Stage 3: Consolidated AlphaZero MCTS-50 (TF-B8, TF-B9, TF-B10)              │
│  • Architecture: Batched PUCT MCTS-50 + Masked Softmax CE Sum + Adam (1e-3)  │
│  • Telemetry:    87.0% vs Random (P1: 84%, P2: 90%), 0.0% vs Heuristic       │
│  • Bottleneck:   MCTS Horizon Effect—leaves are evaluated before forced      │
│                  enclosure cascades resolve, blinding V(s) to 1-move holes.  │
│                                                                              │
│  Stage 4: Quiescence MCTS + Sliding-Window Pure Self-Play (TF-B12–TF-B14)    │
│  • Architecture: Greedy cascade rollouts at MCTS leaves + Root Prior Shaping │
│                  + Dual-Side Teacher + 45K Sliding Replay + Cosine LR Decay  │
│  • Telemetry:    100.0% vs Random | vs Heuristic: 63% (B12) → 89% (B13)      │
│                  → 90.0% Peak / 80.1% Avg (B14) (Score: 106 vs 64!)          │
│  • Bottleneck:   Zero-padded Conv2D (`padding='same'`) is blind to cavities  │
│                  wrapping across the 10×10 torus borders (the ~20% gap).     │
│                                                                              │
│  Stage 5: Full T² Toroidal Manifold + Tactical Safety Filter (TF-B15)        │
│  • Architecture: Circular `ToroidalPad2D` on all 4 Conv layers + Full T²     │
│                  modulo corner symmetry + O(1) Top-K 1-Ply Safety Filter     │
│  • Telemetry:    100.0% vs Random | vs Heuristic: 99.0% Peak / 95.9% Avg Raw │
│                  1-Ply Policy, and 100.0% (1,000W / 0L / 0D) with +Safety!   │
│                  (Cluster Score: AI = 138.4 vs Heuristic = 31.8!)            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

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

To see *why* our telemetry metrics (introduced in [Chapter 2](#2-rl-fundamentals)) were so essential, compare the diagnostic signatures of key milestone runs across the five architectural stages:

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

## 11. Deep Dive: Convolutions, Dual Heads, MCTS & Pipeline Consolidation {#11-optimization-roadmap}

This chapter outlines a prioritized plan to improve AI performance. Each phase builds on the previous, with clear success metrics and implementation notes specific to TensorFlow.js on Mac Studio M2 Ultra.

### Current Baseline

| Metric | Current | Target |
|--------|---------|--------|
| vs Random | 37-52% | 80%+ |
| vs Heuristic | 0% | 10%+ |
| Training time (20 cycles) | 35-40 min | <20 min |
| Samples per cycle | 100 games | 1000 games |

**Why these targets?** 10% vs Heuristic demonstrates the architecture can learn strategic concepts. 80% vs Random proves stability. Faster training enables rapid iteration.

---

### Phase 1: Experience Replay Buffer

**Priority:** Highest (immediate gain, low complexity)

**Problem:** Current training uses each game once. The network forgets early games by the time it sees later ones.

**Solution:** Store game histories in a circular buffer. Sample random batches for training.

```
┌─────────────────────────────────────────────────────────────┐
│                  Experience Replay Buffer                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Traditional: Play 100 games → Train on all → Discard      │
│                                                             │
│   Replay Buffer: Play 100 games → Add to buffer → Sample   │
│                                                             │
│                 ┌───┬───┬───┬───┬───┐                       │
│   Buffer:       │ G1│ G2│ G3│...│G10K│  (100K samples)      │
│                 └───┴───┴───┴───┴───┘                       │
│                      ↓ Random sample                        │
│                 ┌───┬───┬───┐                              │
│   Batch:         │ G5│G42│ G7│  (diverse examples)          │
│                 └───┴───┴───┘                              │
│                                                             │
│   Benefit: See each position multiple times, from different│
│            game contexts. Stable learning.                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 1.1 | Create `ReplayBuffer` class with circular storage | 30 min |
| 1.2 | Implement `add(samples)` method | 15 min |
| 1.3 | Implement `sample(batchSize)` with priority sampling | 30 min |
| 1.4 | Modify training loop to use buffer instead of immediate training | 45 min |
| 1.5 | Add buffer size parameter to CLI | 10 min |
| 1.6 | Test: Train 5 cycles, verify buffer fills correctly | 20 min |
| 1.7 | Run experiment TF-B4 (replay buffer, 20 cycles) | 40 min |
| 1.8 | Document results in TRAINING_BOOK | 15 min |

**Success Criteria:**
- [ ] Buffer fills to capacity (100K samples) within first 10 cycles
- [ ] Training loss decreases smoothly (no spikes)
- [ ] vs Random improves to 60%+ (from current 50%)
- [ ] No memory leaks (verify with Activity Monitor)

**M2 Ultra Optimization:**
- Set buffer size to 500K-1M samples (64GB unified memory)
- Use `tf.data.Dataset` for efficient streaming
- Store tensors in GPU memory when possible

**Code Location:** `src/server/ReplayBuffer.js`

---

### Phase 2: Canonical Slot Encoding + Convolutional Layers

**Priority:** High (architectural fix for input representation AND spatial reasoning)

**Two problems, one solution:**
1. **Encoding is ambiguous** — rotation is a continuous value (0, 0.25, 0.5, 0.75), implying ordinal relationships that don't exist (90° isn't "between" 0° and 180°)
2. **Architecture flattens the board** — losing spatial relationships that are essential for detecting enclosures

**Current:** `401 inputs → [200 hidden] → 400 outputs` (dense, continuous rotation)

**Proposed:** `10×10×18 one-hot input → [Conv layers] → Policy head + Value head`

#### Step 2A: Canonical 1-to-1 Slot Encoding

**Key principle:** Triantica is Markovian — only the current board state matters, not the order moves were played. Each possible placement is a discrete binary fact: present or absent. Empty slots deserve first-class representation, not implicit absence.

```
┌─────────────────────────────────────────────────────────────┐
│              Current vs Canonical Encoding                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CURRENT (4 values per cell):                               │
│    [player_t1, rotation_t1, player_t2, rotation_t2]         │
│    rotation = 0, 0.25, 0.5, 0.75, or -1                    │
│    ❌ Implies 90° is "between" 0° and 180°                 │
│    ❌ Player and geometry mixed in same channel             │
│    ❌ Empty = -1 rotation (special case, not a real angle)  │
│    ❌ Conv layer can't cleanly detect patterns              │
│                                                             │
│  CANONICAL (18 one-hot channels per cell):                  │
│    Per cell, per half (top/bottom):                         │
│      Channel 0:  "empty"                                   │
│      Channel 1-4: "my piece, rotation 0°/90°/180°/270°"    │
│      Channel 5-8: "opponent, rotation 0°/90°/180°/270°"    │
│                                                             │
│    ✅ Every position has exactly ONE active channel/half    │
│    ✅ Empty is first-class, not implicit absence            │
│    ✅ No spurious ordering between rotations                 │
│    ✅ Conv filters get clean binary inputs                   │
│    ✅ Pure board state — no move order information           │
│                                                             │
│  Total: 10×10×18 = 1,800 input values                       │
│  (vs 401 current — still tiny for 64GB M2 Ultra)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why empty slots need their own channel:**
A conv filter detecting "empty cell next to my enclosure" needs to recognize absence. With the 16-channel encoding (all zeros = empty), the filter must apply negative weights across all 16 channels — a complex pattern. With the 18-channel encoding, the filter simply needs a positive weight on channel 0 (empty) — trivial to learn.

**Why this matters for convolutions:**
A 3×3 conv filter on the current 4-channel input would need to simultaneously decode player identity AND rotation from mixed signals. On the 18-channel input, a filter simply checks "do I see my triangles at these specific rotations in these positions?" — directly detecting enclosure patterns.

#### Step 2B: How Convolutional Filters Scan the Board & Connect to Policy and Value Heads

To understand why a Convolutional Neural Network (CNN) dramatically outperforms a flattened Dense network on spatial games, let's visualize **how a $3 \times 3$ filter slides across the $10 \times 10 \times 18$ board tensor**, how stacked convolutions expand the **receptive field**, and how the shared feature map branches into the **Policy Head** and **Value Head**.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     1. HOW A 3×3×18 CONVOLUTION FILTER DETECTS AN ENCLOSURE AT CELL (x,y)    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│     10×10 Board (Spatial Grid)             3×3 Receptive Window Around (x,y) │
│   ┌───┬───┬───┬───┬───┬───┬───┐                 ┌───────┬───────┬───────┐    │
│   │ . │ . │ . │ . │ . │ . │ . │                 │(x-1,  │ (x,   │(x+1,  │    │
│   ├───┼───╔═══╦═══╦═══╗───┼───┤                 │ y-1)  │ y-1)◢ │ y-1)  │    │
│   │ . │ . ║   ║ ◢ ║   ║ . │ . │                 ├───────┼───────┼───────┤    │
│   ├───┼───╠═══╬═══╬═══╣───┼───┤   Dot-Product   │(x-1,  │ (x,y) │(x+1,  │    │
│   │ . │ . ║ ◤ ║ ? ║ ◥ ║ . │ . │ ──────────────► │  y) ◤ │ EMPTY │  y) ◥ │    │
│   ├───┼───╠═══╬═══╬═══╣───┼───┤   over 3×3×18   │       │Ch0=1.0│       │    │
│   │ . │ . ║   ║ ◣ ║   ║ . │ . │   (162 weights) ├───────┼───────┼───────┤    │
│   ├───┼───╚═══╩═══╩═══╝───┼───┤                 │(x-1,  │ (x,   │(x+1,  │    │
│   │ . │ . │ . │ . │ . │ . │ . │                 │ y+1)  │ y+1)◣ │ y+1)  │    │
│   └───┴───┴───┴───┴───┴───┴───┘                 └───────┴───────┴───────┘    │
│                                                                              │
│   Filter k computes:  z_k(x,y) = ReLU( BatchNorm( ∑_{dx,dy,c} W_k · X + b )) │
│                                                                              │
│   • Because the SAME 162 weights slide over all 100 cells, a filter that     │
│     learns to recognize a "1-move enclosure trap" at (2,3) INSTANTLY         │
│     recognizes that exact geometric trap at all 100 cells (Translation       │
│     Invariance)! A Dense network must re-learn it 100 separate times.        │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Hierarchical Feature Extraction Across the 3 Conv Layers (Receptive Field Growth)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     2. HIERARCHICAL SPATIAL ABSTRACTION (32 → 64 → 64 FILTERS)               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Tensor       Conv1 (32 filters)   Conv2 (64 filters)   Conv3 (64 flt) │
│  [10, 10, 18]  ───► [10, 10, 32]    ───► [10, 10, 64]    ───► [10, 10, 64]   │
│                                                                              │
│  Receptive Field:   3×3 cells (9 cells)  5×5 cells (25 cells) 7×7 (49 cells) │
│  What it sees:      • Shared diagonals   • Multi-cell wedges  • Enclosed     │
│                     • Corner touches     • 2-move traps         pockets      │
│                     • Empty vs occupied  • Local cluster      • Territory    │
│                       slots                bridges              boundaries   │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### How the Shared Conv Trunk Connects to the Policy & Value Heads

Once `Conv3` produces the `[B, 10, 10, 64]` spatial feature representation (where each cell `(x, y)` holds a 64-dimensional vector summarizing its $7 \times 7$ neighborhood), the network splits into two specialized heads that work together during **MCTS Search** and **Backpropagation**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     3. DUAL-HEAD SYNERGY: POLICY (ACTOR) + VALUE (CRITIC) IN MCTS            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                  Shared Conv3 Feature Map: [B, 10, 10, 64]                   │
│                                      │                                       │
│                    ┌─────────────────┴─────────────────┐                     │
│                    ▼                                   ▼                     │
│     ┌─────────────────────────────┐     ┌─────────────────────────────┐      │
│     │   POLICY HEAD ("The Actor") │     │   VALUE HEAD ("The Critic") │      │
│     ├─────────────────────────────┤     ├─────────────────────────────┤      │
│     │ 1×1 Conv (2 filters)        │     │ 1×1 Conv (1 filter)         │      │
│     │   [B, 10, 10, 64]           │     │   [B, 10, 10, 64]           │      │
│     │   ──► [B, 10, 10, 2]        │     │   ──► [B, 10, 10, 1]        │      │
│     │ Flatten: [B, 200]           │     │ Flatten: [B, 100]           │      │
│     │ Dense(400) + Sigmoid        │     │ Dense(64, ReLU) ──► Dense(1)│      │
│     │ Output P(s,a): [B, 400]     │     │ Output V(s) = tanh: [B, 1]  │      │
│     └──────────────┬──────────────┘     └──────────────┬──────────────┘      │
│                    │                                   │                     │
│                    ▼                                   ▼                     │
│     ROLE IN MCTS TREE SEARCH:           ROLE IN MCTS TREE SEARCH:            │
│     • Provides Prior P(s,a) in UCT:     • Evaluates leaf board state s_L     │
│       UCT = Q/N + c·P(s,a)·√N/(1+n)       without playing random rollouts    │
│     • Prunes 400 actions down to the      to the end of the game!            │
│       top ~5–10 tactical candidates     • Backpropagates V(s_L) ∈ [-1,+1]    │
│       so 50 sims aren't wasted on         up the tree to update Q(s,a).      │
│       terrible moves.                                                        │
│                                                                              │
│     ROLE IN TRAINING PERFORMANCE:       ROLE IN TRAINING PERFORMANCE:        │
│     • Trained against MCTS visit        • Trained against actual game        │
│       distribution π_MCTS(a):             outcome z ∈ {-1, 0, +1}:           │
│       L_policy = -∑_a π_a log(p̂_a)        L_value = (V(s) - z)^2             │
│     • Distills 50-step lookahead        • Forces the shared 32→64→64 Conv    │
│       into instant 1-step intuition!      filters to learn features that     │
│                                           predict LONG-TERM WINNING, not     │
│                                           just greedy 1-move captures!       │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why Dual-Head Co-Training Significantly Improves Performance:**
1. **Shared Representation Regularization:** Because both heads backpropagate gradients into the same `Conv1 → Conv2 → Conv3` filters ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}}$), the convolutional filters must extract spatial features that simultaneously explain *where MCTS wants to play next* ($\pi$) and *who will ultimately win the game* ($z$). This prevents the Policy Head from memorizing superficial local patterns that lose in the endgame.
2. **Mutual Amplification Loop (Policy Iteration):**
   - A sharper **Policy Head** focuses MCTS simulations onto the best 5–10 candidate moves, allowing a 50-simulation budget to search 4–6 plies deep instead of 1–2 plies wide.
   - A more accurate **Value Head** gives MCTS reliable leaf evaluations at depth 4–6, producing higher-quality visit counts $N(s,a)$, which in turn become a stronger training target $\pi_{\text{MCTS}}$ for the Policy Head in the next cycle!

**Why convolutions work for Triantica (with proper encoding):**
- Enclosures are local spatial patterns (3-7 cells)
- One-hot channels let filters detect exact rotation combinations
- Territory = contiguous regions → convs naturally detect regions
- No spurious ordinal relationships between rotations

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 2.1 | Implement `getStateCanonical(grid, aiPlayerNumber)` — 18-channel one-hot encoding | 45 min |
| 2.2 | Write roundtrip test: encode → decode → verify board state matches | 30 min |
| 2.3 | Implement `createConvNetwork()` function with 10×10×18 input | 1 hour |
| 2.4 | Add policy head (flatten → dense → softmax) | 30 min |
| 2.5 | Add value head (flatten → dense → tanh) — combining Phase 2 + 3 | 30 min |
| 2.6 | Modify training loop: canonical encoding + conv network | 45 min |
| 2.7 | Test: Single game with conv network, verify output shapes | 20 min |
| 2.8 | Run experiment TF-B5 (canonical + conv, 20 cycles) | 50 min |
| 2.9 | Compare training curves vs dense baseline | 20 min |
| 2.10 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Roundtrip test passes: canonical encoding fully captures board state
- [ ] Network produces valid probability distribution over moves
- [ ] Training time per cycle < 3 minutes
- [ ] vs Random matches dense baseline (50%) by cycle 5
- [ ] vs Random exceeds dense baseline (60%+) by cycle 20
- [ ] vs Heuristic shows first wins (even 1% matters)

**M2 Ultra Optimization:**
- Use `tf.conv2d` with `tf.memory()` monitoring
- Batch size 64-128 (larger batches for conv)
- Enable `tf.webgl.forceHalfFloat()` for faster computation

**Code Location:** `src/server/ConvNetwork.js`

---

### Phase 2 Results (2026-10-25)

**Experiments Completed:** TF-B5_conv, TF-B6_optimized

#### TF-B5: Canonical Conv Network + Value Head

| Metric | Value |
|--------|-------|
| Architecture | 3×Conv2D (32/64/64) + Policy head + Value head |
| Parameters | ~148K |
| Input | 10×10×18 canonical one-hot tensor |
| Training time | 42 min (20 cycles, 100 games/cycle) |
| Best vs Random | 50% (cycle 7) |
| Final vs Random | 36% |
| vs Heuristic | 0% (0 wins in 2000 games) |

#### TF-B6: Hyperparameter Optimization

**Changes from TF-B5:**
- Learning rate: 0.01 → **0.02** (BatchNorm allows higher LR)
- Value loss weight: 0.5 → **1.0** (equal signal for value learning)
- Reward target range: [0.01, 0.99] → **[0.0, 1.0]** (stronger gradients)

| Metric | Value |
|--------|-------|
| Training time | 42.5 min (20 cycles) |
| Best vs Random | **50%** (cycle 17) |
| Final vs Random | 33% |
| vs Heuristic | **0%** (0 wins in 2000 games) |

#### Key Findings

```
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: No Breakthrough                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Architecture upgrade successful:                        │
│     • Canonical encoding removes ambiguity                  │
│     • Conv layers can detect spatial patterns               │
│     • Value head provides dense learning signal             │
│                                                             │
│  2. Performance unchanged:                                  │
│     • vs Random: 50% (same as dense baseline)               │
│     • vs Heuristic: 0% (wall still stands)                   │
│                                                             │
│  3. Hyperparameter tuning had no effect:                    │
│     • Higher LR didn't improve learning                     │
│     • Equal value weight didn't help                        │
│     • Full gradient range didn't matter                     │
│                                                             │
│  4. Root cause confirmed:                                   │
│     • The problem is NOT input representation               │
│     • The problem is NOT network capacity                    │
│     • The problem is NOT learning rate/weighting             │
│     • The problem IS: no lookahead/search                   │
│                                                             │
│  Conclusion: Architecture cannot compensate for lack of    │
│              strategic lookahead. MCTS or beam search is    │
│              required to break the Heuristic Wall.         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Comparison: Dense vs Conv

| Metric | Dense (TF-B1) | Conv (TF-B5) | Conv+Opt (TF-B6) |
|--------|---------------|--------------|------------------|
| Parameters | ~160K | ~148K | ~148K |
| Input dim | 401 | 1800 | 1800 |
| Best vs Random | 52% | 50% | 50% |
| vs Heuristic | 0% | 0% | 0% |
| Training time | 40m | 42m | 42m |

#### TF-B8: MCTS-50 Training (AlphaGo Zero Style)

**Goal:** Test if MCTS with 50 simulations can provide strategic lookahead to break the Heuristic Wall.

**Architecture:** Conv (Phase 2 - TF-B5 proven)

**Parameters:**
- Cycles: 10
- Games/cycle: 100
- MCTS sims: 50 (batched, parallel evaluation)
- MCTS batch: 8
- Learning rate: 0.02
- Batch size: 256
- Exploration: 0.1 (MCTS provides exploration via Dirichlet noise)
- Training target: MCTS visit policy π(a)

**Command:**
```bash
node src/server/train_tfjs.cjs --cycles 10 --games 100 --mcts-sims 50 --architecture conv --output dashboard/data/TF-B8.json
```

**Expected:** vs Random 65-70%, vs Heuristic 3-8%

**Actual:** vs Random **30%**, vs Heuristic **0%**

**Status:** ❌ Failed — Network learned to lose (performance degraded)

**Runtime:** 438.1 minutes (~7.3 hours)

**Key Finding:** The network consistently got worse over training:
- Started: 40% vs Random
- Peak (cycle 2): 42% vs Random
- Final (cycle 10): 30% vs Random

This confirms a **sign flip bug** in the training signal. The network IS learning — just learning the wrong thing. The learning mechanism works; the reward signal is inverted.

**Cycle-by-Cycle Results:**

| Cycle | vs Random | vs Heuristic | Training Loss |
|-------|-----------|--------------|---------------|
| 1 | 40% | 0% | 0.99 → 1.00 |
| 2 | **42%** ⬆️ | 0% | 1.98 → 1.00 |
| 3 | 30% | 0% | 1.88 → 1.00 |
| 4 | 37% | 0% | 1.81 → 1.00 |
| 5 | 31% | 0% | 1.93 → 1.00 |
| 6 | **26%** ⚠️ | 0% | 2.34 → 1.00 |
| 7 | 37% | 0% | 1.79 → 1.00 |
| 8 | 37% | 0% | 2.09 → 1.00 |
| 9 | 32% | 0% | 2.04 → 1.00 |
| 10 | **30%** | 0% | 2.03 → 1.00 |

**Analysis:**
- Win rate fluctuates but trend is **downward**
- Final performance (30%) is **worse than random** (50%)
- Training loss converges well (always reaches ~1.00)
- This is NOT overfitting — loss decreases smoothly
- This is NOT capacity issue — network CAN learn
- **This is inverted signal** — network learned to play badly

**Bug Hypotheses (to debug):**
1. `rewardForPlayer` sign flipped somewhere in self-play
2. Canonical encoding: `myPieces=1, enemy=-1` inverted
3. Value target: game outcome from wrong player's perspective
4. Visit count normalization: high visits = bad move?

**Key Implementation Details:**

1. **Training Pipeline Change:**
   - Policy target: Visit distribution π(a) = N(child)/N(root)
   - Value target: Game outcome (unchanged)
   - Network learns: "What would MCTS do?"

2. **MCTS Parameters:**
   - Exploration constant c = 1.5
   - Dirichlet noise α = 0.03, ε = 0.25
   - Temperature: 1.0 early, 0.0 late game
   - Batch size: 8 (parallel leaf evaluation)

3. **Performance Notes:**
   - ~50x slower than regular self-play
   - 1 game ≈ 50 seconds with MCTS-50
   - Batched evaluation: 2x speedup over sequential

---

#### TF-B8 (Clean Consolidated Re-Run): MCTS-50 with All 5 Engine/Search Bugs Fixed

Before incrementing to `TF-B9`, we re-ran `TF-B8` (`dashboard/data/TF-B8.json`) across all 10 cycles (309.1 minutes) with the consolidated `GridManager` rules (`isValidMove`, `executeMove` with enclosures & extra turns, slot-based `getCanonicalState`, ReplayBuffer value targets, and single-perspective MCTS filtering) plus **P1 vs P2 role win-share tracking**:

| Cycle | vs Random (Overall) | AI as P1 (1st Move) | AI as P2 (2nd Move) | vs Heuristic | Eval AI Mean Cluster | Training Loss (Start → End) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | **42.0%** | 42.0% | 42.0% | 0.0% | 37.0 (med 32.0) | `1.394 → 2.056` |
| **C2** | **41.0%** | 40.0% | 42.0% | 0.0% | 35.7 (med 30.0) | `2.289 → 2.222` |
| **C3** | **33.0%** | 26.0% | 40.0% | 0.0% | 33.6 (med 30.0) | `2.272 → 2.252` |
| **C4** | **49.0%** | 38.0% | **60.0%** | 0.0% | 38.8 (med 33.5) | `2.116 → 2.177` |
| **C5** | **41.0%** | **54.0%** | 28.0% | 0.0% | 35.3 (med 31.5) | `1.764 → 1.976` |
| **C6** | **47.0%** | **52.0%** | 42.0% | 0.0% | 37.7 (med 33.0) | `2.206 → 2.149` |
| **C7** | **42.0%** | 36.0% | 48.0% | 0.0% | 34.6 (med 31.0) | `1.738 → 1.955` |
| **C8** | **43.0%** | 48.0% | 38.0% | 0.0% | **40.4 (med 35.0)** | `2.194 → 2.116` |
| **C9** | **45.0%** | **52.0%** | 38.0% | 0.0% | 36.5 (med 33.0) | `1.923 → 2.017` |
| **C10** | **47.0%** | **48.0%** | **46.0%** | 0.0% | 37.1 (med 34.0) | `2.104 → 2.075` |

**Findings from the Clean `TF-B8` Re-Run:**
- **Cluster sizes increased by +60% to +85%** (`33.6–40.4` vs `18.4–26.5` in `TF-B8-V2`) because MCTS now sees enclosures and extra turns via `grid.executeMove`.
- **Stability restored:** Average win rate rose from `30.0%` (`TF-B8 v1`) and `35.4%` (`TF-B8-V2`) to **`43.0%`**, finishing at **`47.0%`** (`48% P1 / 46% P2`) with zero late-cycle collapses.
- **Discovery of Bug 11 (Loss Dilution & Value Saturation):** Inspecting `TF-B8_network.json` revealed why win rate plateaued near 47%: `.mean()` over `[256, 400]` diluted policy gradients by $400\times$ (leaving policy outputs stuck in `0.450–0.545`), while `momentum(0.05, 0.9)` saturated the `tanh` Value Head at `-1.0`.

---

#### TF-B9: MCTS-50 with Policy Cross-Entropy Sum (`400x` Gradient Fix) & Adam (`0.001`)

**Goal:** Isolate and verify the impact of fixing Bug 11 (`.neg().sum(-1).mean()` policy loss reduction + persistent `tf.train.adam(0.001)` optimizer) on top of the consolidated `TF-B8` engine/MCTS baseline.

**Architecture:** Conv (`10×10×18` → `32→64→64` Conv2D + BatchNorm + Policy & Value Heads)

**Parameters:**
- Cycles: 10
- Games/cycle: 100 (Eval: 100 games vs Random + 100 vs Heuristic, with P1/P2 role tracking)
- MCTS sims: 50 (batched leaf evaluation, `mctsBatchSize = 8`)
- Optimizer & LR: `tf.train.adam(0.001)`
- Batch size: 256 (`replayRatio = 2x`, `500,000` buffer)

**Command:**
```bash
node src/server/train_tfjs.cjs --cycles 10 --games 100 --eval-games 100 --mcts-sims 50 --architecture conv --learning-rate 0.001 --output dashboard/data/TF-B9.json
```

**Live Pre-Run Verification (20 Games, Cycle 1):**
- Policy Head immediately separates top MCTS moves (`0.985`) from bad/illegal moves (`0.000`).
- Value Head stays unsaturated (`+0.2538` on mid-game boards).
- Achieved **56.0% vs Random** (`52.0% as P1`, `60.0% as P2`) in the 20-game live verification test.

**Status:** ✅ **Completed** (`292.6` minutes, 10/10 cycles, **84.0% Peak / 79.0% Final vs Random**, **69.9% 10-Cycle Mean**, published to `dashboard/data/TF-B9.json` and `trainedAI_network.json`).

**Complete 10-Cycle Trajectory (`TF-B9`):**

| Cycle | Initial Error (`Epoch 1`) | Final Error (`Epoch 10`) | vs Random (Overall) | AI as P1 (1st Move) | AI as P2 (2nd Move) | Eval AI Cluster (`Mean / Med / Max`) | vs Heuristic (Cluster Mean) | Self-Play (`P1/P2/D`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | `3.672` | `2.577` | **52.0%** | 58.0% | 46.0% | `41.5 / 35 / 96` | `0.0%` (`4.0`) | `55 / 44 / 1` |
| **C2** | `3.324` ⬇️ | `2.387` | **55.0% ⬆️** | 58.0% | 52.0% ⬆️ | `45.0 / 36 / 111` | `0.0%` (`6.5`) | `49 / 48 / 3` |
| **C3** | `3.205` ⬇️ | `2.375` | **63.0% ⬆️** | 58.0% | 68.0% ⬆️ | `52.7 / 45 / 132` | `0.0%` (`3.5`) | `55 / 45 / 0` |
| **C4** | `3.154` ⬇️ | `2.371` | **64.0% ⬆️** | 60.0% ⬆️ | 68.0% | `55.4 / 47 / 148` | `0.0%` (`2.5`) | `50 / 49 / 1` |
| **C5** | **`3.260` ⬆️** | `2.392` | **74.0% 🚀** | **80.0% 🚀** | 68.0% | `58.4 / 56 / 124` | `0.0%` (**`10.5`**) | `53 / 44 / 3` |
| **C6** | `3.193` ⬇️ | `2.373` | **72.0%** | **80.0%** | 64.0% | `65.7 / 63 / 135` | `0.0%` (`8.5`) | `59 / 41 / 0` |
| **C7** | **`3.249` ⬆️** | `2.375` | **84.0% 🚀** | **88.0% 🚀** | **80.0% 🚀** | `66.2 / 63 / 132` | `0.0%` (`6.5`) | `59 / 41 / 0` |
| **C8** | `3.151` ⬇️ | **`2.357`** | **79.0%** | 84.0% | 74.0% | `66.2 / 64 / 135` | `0.0%` (`8.0`) | `60 / 39 / 1` |
| **C9** | **`2.997` ⬇️** | `2.366` | **77.0%** | 82.0% | 72.0% | **`66.5 / 61 / 138`** | `0.0%` (`9.0`) | **`50 / 50 / 0`** |
| **C10** | `3.095` | `2.367` | **79.0% ✅** | **84.0%** | **74.0%** | `66.2 / 63 / 136` | `0.0%` (`4.5`) | `61 / 37 / 2` |
| **Mean / Peak** | — | **`2.357` (best)** | **`69.9%` (`84.0%` Peak)** | **`73.2%` (`88.0%` Peak)** | **`66.6%` (`80.0%` Peak)** | **`66.5` Peak Mean** | `0.0%` (`10.5` Peak) | `70,846` samples |

**Key Findings from `TF-B9`:**
- **+35.0 Percentage Point Peak Breakthrough over `TF-B8` (`84.0%` vs `49.0%`):** Restoring the $400\times$ policy cross-entropy gradient (`.neg().sum(-1).mean()`) and preventing `tanh` Value Head saturation (`tf.train.adam(0.001)`) transformed the exact same ConvNet + MCTS-50 architecture from a `43.0%` average (`TF-B8`) to a **`69.9%` 10-cycle average**, peaking at **`84.0%`** in Cycle 7 and finishing cleanly at **`79.0%`** in Cycle 10 (`84.0% as P1`, `74.0% as P2`) with **zero late-cycle collapses**.
- **Massive Territory Expansion (`Mean 66.2–66.5` vs `36–40` in `TF-B8` and `18–26` in `TF-B8-V2`):** In Cycles 6–10, the AI consistently builds contiguous territories averaging **`66+` triangles** (median `63–64`, max `135–148`), more than **$3\times$ larger** than the pre-audit `TF-B8-V2` network.
- **Strong Dual-Role Mastery:** By Cycle 7–10, the network dominates Random both as **First Mover P1 (`82%–88%`)** and as **Second Mover P2 (`72%–80%`)**, and in Cycle 9 achieved an exact `50W P1 / 50W P2 / 0D` equilibrium in MCTS self-play.

---

#### Theoretical Deep Dive: Why Initial Error "Sawtooths" Across Self-Play Cycles (Strategic Valleys & Phase Transitions)

A fascinating empirical phenomenon appears in the `TF-B9` training logs above: while the **within-cycle loss** drops smoothly every single cycle (e.g., `3.249 → 2.375` across the 10 epochs of Cycle 7), the **initial error at the start of each new cycle** oscillates in a distinct sawtooth pattern (`3.672 → 3.324 → 3.205 → 3.154 → 3.260 ⬆️ → 3.193 → 3.249 ⬆️ → 3.151`).

Does this oscillation in initial error mean the network is reaching **new valleys in the loss landscape where qualitatively different strategies are learned**? **Yes — and the mathematics of Generalized Policy Iteration explains exactly why.**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│     THE "SAWTOOTH" LOSS LANDSCAPE OF ALPHAZERO SELF-PLAY (TF-B9)             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Loss                                                                        │
│  3.7 ┼ ╲ C1                                                                  │
│      │  ╲   ╲ C2                                                             │
│  3.2 ┼   ╲   ╲   ╲ C3  ╲ C4    ▲ C5 Spike!       ▲ C7 Spike!                 │
│      │    ╲   ╲   ╲     ╲      │╲  (New P1       │╲  (New Counter-    ╲ C8   │
│  2.8 ┼     ╲   ╲   ╲     ╲     │ ╲  Enclosures)  │ ╲  Tactics)         ╲     │
│      │      ╲   ╲   ╲     ╲    │  ╲    ╲ C6      │  ╲                   ╲    │
│  2.3 ┼───────●───●───●─────●───┘───●────●────────┘───●───────────────────●── │
│              │             │       │                 │                       │
│         [Valley 1: Basic   ]   [Valley 2: P1    ] [Valley 3: Dual-Role ]     │
│         [Contiguous Growth ]   [Enclosure Traps ] [MCTS Counter-Play   ]     │
│         Win Rate: 52% → 64%    Win Rate: 74%      Win Rate: 84% (88%/80%)    │
└──────────────────────────────────────────────────────────────────────────────┘
```

##### 1. Supervised Learning vs. Non-Stationary Self-Play
In standard **supervised learning**, the dataset $\mathcal{D}$ is static. Because the data distribution never changes, the loss surface is fixed: the final error at the end of Epoch 10 (`2.371`) would be the exact initial error at the start of Epoch 11 (`2.371`).

In **MCTS Self-Play (Generalized Policy Iteration)**, the training loop alternates between two coupled operators:
1. **Policy Evaluation & Improvement (Within Cycle $k$):** For 10 epochs, the network parameters $\theta_k$ descend into the local valley of the *current* state distribution $\mathcal{D}_k$, reducing loss from $\sim 3.20 \to \sim 2.37$.
2. **Distribution Shift via Tree Search (Between Cycle $k$ and $k+1$):** In Cycle $k+1$, the newly trained network $\theta_k$ guides **MCTS-50** through 100 fresh self-play games. Because $\theta_k$ now plays stronger moves:
   - Both players steer the board into **previously unreachable state sub-manifolds** $\mathcal{D}_{k+1}$ (contested diagonal wedges, multi-cell bridges, and tighter endgames).
   - Inside those new board states, 50-simulation lookahead discovers **tactical refutations and multi-turn enclosure combos** that contradict the raw 1-ply intuition of $\theta_k$.

Therefore, `initialError` at the start of Cycle $k+1$ is **not** a continuation of Cycle $k$'s loss curve—it is the **Kullback-Leibler / Cross-Entropy "Surprise"** of network $\theta_k$ when confronted with MCTS's 50-step lookahead in newly unlocked board states:
$$\mathcal{L}_{\text{initial}}^{(k+1)} = \mathbb{E}_{s \sim \mathcal{D}_{k+1}} \left[ \underbrace{-\sum_{a=1}^{400} \pi_{\text{MCTS}}^{(k+1)}(a \mid s) \log \hat{p}_{\theta_k}(a \mid s)}_{\text{Tactical Surprise (MCTS found a new move)}} + \underbrace{\left(V_{\theta_k}(s) - z^{(k+1)}\right)^2}_{\text{Outcome Surprise (Who wins flipped)}} \right]$$

##### 2. Empirical Proof: Every `initialError` Spike Preceded a Major Win-Rate Breakout
Look closely at the `TF-B9` table above:
- **Cycles 1 → 4 (Descending into Valley 1):** `initialError` drops monotonically (`3.672 → 3.324 → 3.205 → 3.154`) while the AI learns fundamental contiguous placement. By Cycle 4, it reaches the bottom of Valley 1 and its win rate plateaus (`63.0% → 64.0%`).
- **Cycle 5 (Phase Transition 1 — `initialError` spikes `3.154 → 3.260 ⬆️`):** Armed with the C4 network, MCTS self-play breaks out of Valley 1 and discovers aggressive **P1 opening enclosures**. The old C4 weights are initially "surprised" by these new board states (`3.260`), but after 10 epochs of descending into **Valley 2** (`3.260 → 2.392`), the AI's win rate vs Random leaps by **+10 points to `74.0%`** (**`80.0%` as P1**), and its mean cluster size against the Heuristic opponent jumps $4\times$ (`2.5 → 10.5`)!
- **Cycle 7 (Phase Transition 2 — `initialError` spikes `3.193 → 3.249 ⬆️`):** Once P1 knows how to set enclosure traps (C5–C6), MCTS in Cycle 7 discovers **P2 defensive counter-tactics** that refute naive P1 expansion. Again, `initialError` spikes to `3.249` as the network enters **Valley 3**. Once digested (`3.249 → 2.375`), overall win rate surges by **+12 points to `84.0%`**, with **P1 reaching `88.0%`** and **P2 jumping from `64.0%` to `80.0%`**!

##### 3. Why the Replay Buffer Turns "Hopping Valleys" Into "Widening the Basin"
In naive self-play without an experience replay buffer, moving from Valley 1 $\to$ Valley 2 $\to$ Valley 3 often causes **catastrophic forgetting** (rock-paper-scissors cycling: the network learns Strategy B to beat Strategy A, forgets how to beat Random, and collapses).
- Because `TF-B9` trains every cycle on a **1:2 mixture** of **new self-play states (`~7,050` samples from Valley $k+1$)** and **historical Replay Buffer states (`~14,336` samples uniformly sampled across Valleys $1 \dots k$)**, the optimizer is forbidden from abandoning earlier valleys.
- When the combined loss converges down to a new global low (`2.357` in Cycle 8), it proves the ConvNet has found a **shared parameterization $\theta$ whose basin simultaneously satisfies the geometry of all previously discovered strategies**.

#### TF-B10: Breaking the Heuristic Wall via Mixed-Opponent MCTS Training (`50% Network / 50% Heuristic`)

**Diagnostic Root Cause — Why Pure Self-Play (`TF-B9`) Loses to `HeuristicPlayer`:**
A move-by-move trace between `TF-B9_network.json` (P1) and `HeuristicPlayer` (P2) revealed the exact tactical mechanism behind the Heuristic Wall:
- Because `executeMove` grants extra turns via `Math.floor((1 + claimed) / 2)`, **even a 1-triangle enclosure (`claimed = 1`) awards a full extra turn**.
- `HeuristicPlayer` explicitly evaluates 1-step enclosure claims (`+0.3 × claimed`), contiguous cluster growth (`+0.1 × largestCluster`), and relative lead (`+0.2 × (newLead - currentLead)`). Starting as early as **Move 3**, `HeuristicPlayer` chains 6–8 consecutive `claimed = 1` moves in a single turn sequence (`P2_clust` snowballing `1 → 3 → 5 → 19 → 55` by Move 38 while `P1_clust` remains at `1`), getting `96` moves to the AI's `14` moves across a 110-move game.
- In pure self-play (`TF-B9`), neither side played this hyper-aggressive early 1-triangle enclosure chain, so the network's policy and value heads never encountered (or learned to block/contest) early extra-turn snowballs.

**Configuration (`TF-B10`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001 \
  --opponent-mix mixed --network-ratio 0.5 --heuristic-ratio 0.5 --random-ratio 0.0 \
  --start-from dashboard/data/TF-B9_network.json \
  --output dashboard/data/TF-B10.json > dashboard/data/TF-B10.log 2>&1 < /dev/null &
```
- **Warm-Start:** Initialized from `dashboard/data/TF-B9_network.json` (`79.0%` final / `84.0%` peak vs Random).
- **Opponent Curriculum:** `50%` MCTS-50 Self-Play + `50%` `HeuristicPlayer` (`explorationFactor: 0.1`), exposing the AI to aggressive early enclosure chains while maintaining self-play stability via the 500K Replay Buffer.
- **Status:** ✅ **Completed** (`161.3` minutes, 10/10 cycles, **87.0% Peak & Final vs Random**, **81.8% 10-Cycle Mean**, published to `dashboard/data/TF-B10.json` and `trainedAI_network.json`).

**Complete 10-Cycle Trajectory (`TF-B10`):**

| Cycle | vs Random (Overall) | AI as P1 | AI as P2 | Mean Cluster vs Random | vs Heuristic | AI Cluster vs Heuristic (Mean / Max) | Loss (`Initial → Final`) | Self-Play Wins (P1/P2/D) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **78.0%** (78W/22L/0D) | 76.0% | 80.0% | 70.9 | 0.0% | 9.5 / 10 | `3.158 → 2.185` | 49 / 51 / 0 |
| **2** | **76.0%** (76W/21L/3D) | 74.0% | 78.0% | 66.5 | 0.0% | 6.0 / 7 | `3.015 → 2.196` | 45 / 55 / 0 |
| **3** | **79.0%** (79W/21L/0D) | 80.0% | 78.0% | 72.8 | 0.0% | **13.0 / 20** | `3.190 → 2.211` | 49 / 51 / 0 |
| **4** | **84.0%** (84W/16L/0D) | 80.0% | **88.0%** | 74.9 | 0.0% | 9.5 / 11 | `2.929 → 2.212` | 56 / 44 / 0 |
| **5** | **81.0%** (81W/18L/1D) | 80.0% | 82.0% | 69.5 | 0.0% | 5.5 / 6 | `3.130 → 2.240` | 47 / 52 / 1 |
| **6** | **87.0% 🚀** (87W/12L/1D) | **92.0% 🚀** | 82.0% | 74.3 | 0.0% | 6.5 / 9 | `3.067 → 2.243` | 48 / 51 / 1 |
| **7** | **86.0%** (86W/14L/0D) | 86.0% | 86.0% | 73.7 | 0.0% | 7.0 / 12 | `2.865 → 2.246` | 52 / 47 / 1 |
| **8** | **82.0%** (82W/18L/0D) | 86.0% | 78.0% | **79.2** | 0.0% | 5.0 / 8 | `3.222 → 2.252` | 46 / 54 / 0 |
| **9** | **78.0%** (78W/22L/0D) | 86.0% | 70.0% | 71.2 | 0.0% | 4.5 / 6 | `3.050 → 2.262` | 56 / 44 / 0 |
| **10** | **87.0% ✅** (87W/11L/2D) | **90.0%** | **84.0%** | 76.5 | 0.0% | 4.0 / 7 | `2.938 → 2.249` | 51 / 48 / 1 |

**Key Takeaways from `TF-B10`:**
1. **New All-Time TF.js Record vs Random (`87.0%` Peak & Final, `81.8%` 10-Cycle Mean):**
   - Overall 10-cycle average climbed from `69.9%` (`TF-B9`) to **`81.8%`** (`83.0%` as P1, `80.6%` as P2), peaking at **`92.0%` as P1** (Cycle 6) and **`88.0%` as P2** (Cycle 4).
   - Mean contiguous cluster size against Random reached **`79.2` triangles** in Cycle 8 (`76.5` in Cycle 10).
   - Standalone 1-ply browser network verification (`verify_game_network.cjs` without MCTS) improved from `68%` (`TF-B9`) to **`76%` vs Random (`76W / 24L / 0D`)**.
2. **Why `0.0%` vs `HeuristicPlayer` Persists Even With Mixed Opponent Training:**
   - During the 50 Heuristic games per cycle, `HeuristicPlayer` chains 1-triangle enclosures (`Math.floor((1 + claimed) / 2) = 1` extra turn per single claim) from Move 3 onward, taking ~85% of the moves (`~14–16` AI turns per game vs `~70` in self-play).
   - An inflection-point audit of `TF-B10` revealed that standard `MCTS.cjs` evaluates leaf states purely via the neural Value Head (which rated dangerous hole-creating moves at `+0.4794` vs safe moves at `+0.1060`). Without playing out forced enclosure cascades at the leaves, 50 MCTS simulations actively amplified single-move openings that `HeuristicPlayer` immediately exploited.

#### TF-B12: Quiescence MCTS + Mixed-Opponent Training (`50% Network / 50% Heuristic`)

**Architectural Principle — Let Quiescence Search Teach Both Enclosing Trains and Hole Avoidance:**
Rather than complicating the neural network with auxiliary risk heads or artificial TD($\lambda$) reward shaping, **TF-B12** keeps the clean 2-head ConvNet (`Policy + Value`) and upgrades the MCTS teacher to **`QuiescenceMCTS`** (`src/server/QuiescenceMCTS.cjs`):
1. **Enclosing Moves Are Never Risky (`claimed > 0` $\implies$ `risk = 0`):** Because any move with `claimed > 0` grants `Math.floor((1 + claimed) / 2) >= 1` extra turns, the active player retains the turn and can immediately chain further enclosures ("enclosing trains") or patch any holes created. `QuiescenceMCTS` gives enclosing moves a positive bonus (`enclosureBonusWeight`) at both the root prior and leaf evaluation.
2. **Penalize Single-Move Openings (`claimed === 0` with Opponent Enclosures):** Only non-enclosing moves that pass the turn to the opponent can be exploited. If a candidate move leaves enclosure opportunities for the opponent, `QuiescenceMCTS` plays out the deterministic greedy cascade at the leaf (`_runQuiescenceRollout`) and penalizes both the leaf Q-value and the root visit policy target (`_getRiskAdjustedPolicy`).
3. **Batched Pre-Cascade Expansion + Post-Cascade Evaluation (`searchBatched` / `_runBatch`):** Leaf nodes are always expanded using the policy of `leaf.grid` (pre-cascade), while leaf values are evaluated on the quiescent post-cascade board in a single combined `batchPredict` GPU call.

**Configuration (`TF-B12`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.001 \
  --opponent-mix mixed --network-ratio 0.5 --heuristic-ratio 0.5 --random-ratio 0.0 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.2 \
  --start-from dashboard/data/TF-B10_network.json \
  --output dashboard/data/TF-B12.json > dashboard/data/TF-B12.log 2>&1 < /dev/null &
```

#### TF-B13: Dual-Side Quiescence MCTS + 75/25 Curriculum (`89% Peak / 73% Final vs Heuristic`)

**Architectural & Curriculum Refinements:**
Building directly on `TF-B12_network.json`, **TF-B13** introduced three targeted improvements to consolidate strategic mastery against `HeuristicPlayer`:
1. **Dual-Side MCTS Self-Play Sample Collection (`!step.mctsPolicy` Filter):**
   In `TF-B9` through `TF-B12`, the MCTS sample loop filtered by `if (step.player !== aiPlayerNumber || !step.mctsPolicy) continue;`, discarding half of all MCTS-50 searches in network-vs-network games. Because every state in `hist` is already canonically encoded from `step.player`'s perspective (`1` = mover, `-1` = opponent) and `rewardForPlayer` is computed relative to `step.player`, both P1 and P2 MCTS moves are 100% perspective-consistent. Changing the filter to `if (!step.mctsPolicy) continue;` captured **both sides of network-vs-network MCTS games** while still excluding non-MCTS `HeuristicPlayer` moves—nearly doubling sample generation from **`~6,096` samples/cycle (`60,961` total in `TF-B12`)** to **`~11,589` samples/cycle (`115,892` total in `TF-B13`)**.
2. **75% Network / 25% Heuristic Curriculum + Fine-Tuning Schedule (`LR = 0.0005`, `riskPenalty = 0.25`):**
   Shifted the opponent mix to `75%` dual-side `QuiescenceMCTS-50` self-play and `25%` `HeuristicPlayer`, halved the Adam learning rate (`0.001` $\to$ `0.0005`), and raised `riskPenalty` (`0.20` $\to$ `0.25`) to sharpen punishment of single-move holes while stabilizing high-level self-play.
3. **100-Opening Random 2-Ply Protocol Across All 10 Cycles & Browser Parity:**
   Evaluated all 10 cycles using the 100-game random-first-2-plies protocol (50 as P1, 50 as P2) and updated `game.js` so browser AI/Heuristic games also play random first 2 plies (`this.totalMovesPlayed < 2`).

**Configuration (`TF-B13`):**
```bash
nohup node src/server/train_tfjs.cjs \
  --cycles 10 --games 100 --eval-games 100 \
  --mcts-sims 50 --architecture conv --learning-rate 0.0005 \
  --opponent-mix mixed --network-ratio 0.75 --heuristic-ratio 0.25 --random-ratio 0.0 \
  --quiescence-mode greedy --quiescence-depth 5 --risk-penalty 0.25 \
  --start-from dashboard/data/TF-B12_network.json \
  --output dashboard/data/TF-B13.json > dashboard/data/TF-B13.log 2>&1 < /dev/null &
```

---

### Phase 3: Value Head (Actor-Critic) — MERGED INTO PHASE 2

**Status:** ✅ Merged — Value head added as step 2.5 in the combined Phase 2 plan. Canonical slot encoding + conv layers + value head are being implemented together since they're deeply interdependent. A value head on top of the wrong encoding would give misleading signal; the canonical encoding makes the value head's job tractable.

**Original Problem:** Network only learns "what moves are good" (policy). It doesn't know "how good is my position" (value).

**Current:** Pure policy gradient — only learns from game outcomes.

**Proposed:** Actor-Critic — learn both policy and value simultaneously.

```
┌─────────────────────────────────────────────────────────────┐
│                    Actor-Critic Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Shared trunk (conv layers)                               │
│                      ↓                                      │
│            ┌────────────┴────────────┐                      │
│            ↓                         ↓                      │
│      Actor (Policy)             Critic (Value)               │
│   Input: Position              Input: Position               │
│   Output: Move probs            Output: Win prob [−1, +1]     │
│                                                             │
│   Training signal:              Training signal:             │
│   - Final outcome × policy     - TD(λ) from game outcomes   │
│   - MCTS improved policy       - Bootstrap from value net  │
│                                                             │
│   Benefit: Value head provides dense learning signal.      │
│   Even when you lose, you learn which positions were bad.   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 3.1 | Add value head to conv network (flatten → dense → tanh) | 45 min |
| 3.2 | Implement value loss: MSE(predicted, actual) | 30 min |
| 3.3 | Combine policy loss + value loss (weighted) | 30 min |
| 3.4 | Implement TD(λ) returns for value targets | 1 hour |
| 3.5 | Modify training loop: compute both losses | 45 min |
| 3.6 | Add value prediction to evaluation logging | 20 min |
| 3.7 | Test: Verify value predictions are in [−1, +1] | 15 min |
| 3.8 | Run experiment TF-B6 (actor-critic, 20 cycles) | 50 min |
| 3.9 | Analyze: correlation between value and outcome | 30 min |
| 3.10 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Value predictions correlate with outcomes (r > 0.5)
- [ ] Training converges faster (same loss in fewer iterations)
- [ ] vs Random improves to 65%+ (from Phase 2's 60%)
- [ ] vs Heuristic shows first wins (even 1% matters)

**M2 Ultra Optimization:**
- Larger batches for value loss (32-64)
- Use `tf.losses.meanSquaredError` with reduction=SUM
- Value loss weight: 0.5 (tune after initial results)

**Code Location:** `src/server/ActorCriticNetwork.js`

---

### Phase 4: Batch Normalization + Deeper Network

**Priority:** Medium (stability for deeper networks)

**Problem:** Deeper networks are harder to train. Gradients vanish or explode.

**Solution:** Batch normalization after each conv layer.

```
┌─────────────────────────────────────────────────────────────┐
│              Deep Residual Network with BatchNorm            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU    │
│                      ↓                                      │
│                Residual connection (add input)              │
│                      ↓                                      │
│   (Repeat 4-6 times)                                        │
│                      ↓                                      │
│            Policy Head + Value Head                         │
│                                                             │
│   BatchNorm benefits:                                       │
│   • 10x faster convergence                                  │
│   • Allows deeper networks without gradient issues          │
│   • More stable training (fewer hyperparameter tweaks)     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 4.1 | Add `tf.batchNormalization()` after each conv | 30 min |
| 4.2 | Implement residual block (conv → BN → ReLU → conv → BN → add skip) | 1 hour |
| 4.3 | Stack 4 residual blocks | 20 min |
| 4.4 | Test: Verify gradients don't explode (add gradient clipping) | 30 min |
| 4.5 | Run experiment TF-B7 (deep network, 20 cycles) | 60 min |
| 4.6 | Compare training curves vs Phase 3 | 20 min |
| 4.7 | Document results in TRAINING_BOOK | 20 min |

**Success Criteria:**
- [ ] Training reaches lower loss than Phase 3
- [ ] No NaN losses (gradient stability)
- [ ] vs Random 70%+ (from Phase 3's 65%)
- [ ] vs Heuristic 3-5% (from Phase 3's 1%)

**M2 Ultra Optimization:**
- BatchNorm works better with larger batches (64+)
- Use `training: true` in BatchNorm during training
- Lower learning rate (0.001 → 0.0005) for stability

**Code Location:** `src/server/ResidualBlock.js`

---

### Phase 5: Monte Carlo Tree Search (MCTS)

**Priority:** High (required to break the Heuristic Wall)

**Problem:** Network has no mechanism for "what if" reasoning. It sees the current position and outputs move probabilities, but cannot simulate consequences.

**Why this matters:** The Heuristic player evaluates 5-10 moves ahead through simulation. Our network sees only the current board. This is a fundamental asymmetry that no architecture can overcome without search.

---

#### MCTS Theory: A Complete Explanation

Monte Carlo Tree Search combines tree search with neural network evaluation. It's what made AlphaGo Zero superhuman.

**The Four Phases of MCTS:**

```
┌─────────────────────────────────────────────────────────────┐
│                 MCTS Search Loop                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐                                           │
│   │   SELECT    │  Traverse tree from root to leaf          │
│   │     ↓       │  using UCT formula                        │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │   EXPAND    │  Add child nodes at leaf                  │
│   │     ↓       │  Initialize with NN priors                │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │  EVALUATE   │  Use value network to score leaf          │
│   │     ↓       │  v = tanh output (-1 to +1)               │
│   └─────────────┘                                           │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────┐                                           │
│   │ BACKPROP    │  Update Q and N up to root               │
│   │             │  Q += v, N += 1                          │
│   └─────────────┘                                           │
│                                                             │
│   Repeat N times, then return improved policy              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**The UCT Formula (Upper Confidence Bound for Trees):**

```
UCT(action) = Q(action)/N(action) + c × P(action) × √(N(parent))/(1+N(action))
              ↑                              ↑
              │                              └── Policy prior from NN
              └── Average value from simulations

Where:
- Q = cumulative value from backpropagation
- N = visit count
- P = policy prior from neural network
- c = exploration constant (typically 1.0-2.0)
```

**Why UCT works:** The first term exploits good moves (high Q/N). The second term explores moves with high prior (P) but few visits (low N). This balances exploitation vs exploration.

---

#### MCTS Integration with Our Architecture

Our Phase 2 architecture is perfectly suited for MCTS:

| Component | Role in MCTS |
|-----------|-------------|
| Canonical encoding (10×10×18) | Fast state serialization for tree nodes |
| Policy head (400 sigmoid) | Move priors (P) for UCT formula |
| Value head (1 tanh) | Position evaluation (v) at leaves |
| BatchNorm layers | Stable inference during search |

**Data Flow:**

```
┌─────────────────────────────────────────────────────────────┐
│                 MCTS + Neural Network Pipeline               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Current board state (canonical tensor)              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Neural Network Forward Pass (ONE call per move)      │   │
│  │                                                       │   │
│  │  Input: [10, 10, 18] canonical tensor                │   │
│  │     ↓                                                 │   │
│  │  Conv layers (shared backbone)                       │   │
│  │     ↓                                                 │   │
│  │  ┌──────┴──────┐                                     │   │
│  │  ↓             ↓                                      │   │
│  │  Policy: 400   Value: 1                               │   │
│  │  sigmoid       tanh [-1, +1]                          │   │
│  │  P(move)       Position score                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  MCTS Search (N simulations):                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  For each simulation:                                 │   │
│  │                                                       │   │
│  │  1. SELECT: Traverse tree using UCT                   │   │
│  │     UCT = Q/N + c × P × √(N_parent)/(1+N)           │   │
│  │            ↑   ↑                                      │   │
│  │            │   └── Policy prior from NN              │   │
│  │            └── Average value from backprop            │   │
│  │                                                       │   │
│  │  2. EXPAND: Add child nodes with NN priors           │   │
│  │                                                       │   │
│  │  3. EVALUATE: At leaf → use value head               │   │
│  │     v = value_network(state)                         │   │
│  │                                                       │   │
│  │  4. BACKPROP: Update Q and N up the tree             │   │
│  │                                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Improved policy π(a) = N(child)/N(root)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

#### Simulation Count Trade-off

More simulations = better play, but slower:

| Sims | Time/move | Use Case |
|------|-----------|----------|
| 50 | ~2s | Training (fast iteration) |
| 200 | ~8s | Evaluation (quality check) |
| 1000 | ~40s | Final model (best performance) |

**Recommendation:** Start with 50 sims for training, evaluate at 200.

---

#### M2 Ultra Optimization: Batched Evaluation

The key insight: neural network calls are expensive, tree operations are cheap.

**Naive MCTS (slow):**
```javascript
for (let i = 0; i < simulations; i++) {
  const leaf = select(root);
  const value = evaluate(leaf);  // One NN call per sim
  backpropagate(leaf, value);
}
```

**Batched MCTS (fast):**
```javascript
async function parallelSearch(root, simulations, batchSize = 8) {
  const batches = Math.floor(simulations / batchSize);
  
  for (let b = 0; b < batches; b++) {
    // Collect leaves for batch evaluation
    const leaves = [];
    for (let i = 0; i < batchSize; i++) {
      const leaf = select(root);
      applyVirtualLoss(leaf);  // Mark as "in progress"
      leaves.push(leaf);
    }
    
    // ONE batched neural net call for all leaves
    const states = leaves.map(l => encodeCanonical(l.grid));
    const { policies, values } = await network.batchPredict(states);
    
    // Process all results
    for (let i = 0; i < batchSize; i++) {
      revertVirtualLoss(leaves[i]);
      expand(leaves[i], policies[i]);
      backpropagate(leaves[i], values[i]);
    }
  }
}
```

**Expected speedup:** 4-8x faster with batch size 8-16.

---

#### Virtual Loss for Parallel MCTS

When multiple threads explore the same node, we need to prevent them from all selecting the same path. Virtual loss solves this:

```javascript
function applyVirtualLoss(node) {
  node.Q -= 1;  // Temporarily reduce value
  node.N += 1;  // Temporarily increase visits
}

function revertVirtualLoss(node) {
  node.Q += 1;  // Restore value
  node.N -= 1;  // Restore visits
}
```

This makes a node look "worse" during parallel search, encouraging other threads to explore elsewhere.

---

#### Dirichlet Noise (Exploration)

At the root node, we add noise to policy priors to encourage exploration:

```javascript
function addDirichletNoise(priors, alpha = 0.03, epsilon = 0.25) {
  // Sample noise from Dirichlet distribution
  const noise = sampleDirichlet(alpha, priors.length);
  
  // Mix with priors
  return priors.map((p, i) => 
    (1 - epsilon) * p + epsilon * noise[i]
  );
}
```

**Why this matters:** Without noise, MCTS collapses to one line of play. With noise, it explores diverse strategies.

---

#### Training Pipeline Change

The key innovation from AlphaGo Zero: train the network to predict MCTS output.

```
Current (NN only):
  Self-play games → Store positions → Train network → Repeat

With MCTS (AlphaGo Zero style):
  Self-play games → MCTS improves policy → Store (state, π_MCTS, outcome) → Train network → Repeat
                    ↑                              ↑
                    └── Stronger supervision ──────┘
```

**What changes:**
- Input: Current board state (same)
- Target policy: π_MCTS (MCTS visit distribution) instead of raw network output
- Target value: Game outcome (same)

The network learns: "What would MCTS do?" not "What do I think?"

---

#### Expected Performance Impact

Based on AlphaGo Zero results and our current baseline:

| Metric | Current (NN only) | With MCTS-50 | With MCTS-200 |
|--------|-------------------|-------------|---------------|
| vs Random | 50% | 70-75% | 75-80% |
| vs Heuristic | 0% | **5-10%** | **10-15%** |
| Time/move | 50ms | 2-3s | 8-10s |

**The breakthrough:** MCTS provides the lookahead that the network cannot learn. The Heuristic player wins by planning enclosures 10-20 moves ahead. MCTS gives our network the same capability.

---

#### Risk Assessment

| Risk | Mitigation |
|------|------------|
| Slow training (MCTS per move) | Batch evaluation, start with 50 sims |
| Bug in tree logic (memory leak) | Unit tests, tree visualization, N tracking |
| Overfitting to MCTS policy | Dirichlet noise, temperature schedule |
| Still doesn't beat Heuristic | Increase simulations, tune c parameter |
| Implementation complexity | Start simple (sequential), optimize later |

---

```
┌─────────────────────────────────────────────────────────────┐
│                    MCTS Integration                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   1. Selection: Start from root, pick promising moves        │
│      • Use policy network to guide selection                 │
│      • Use UCT formula: Q + c × P × √N/(1+n)                │
│                                                             │
│   2. Expansion: Add new node when reaching leaf             │
│      • Use policy network to rank moves                     │
│                                                             │
│   3. Evaluation: At leaf, use value network                 │
│      • v = value_network(position)                         │
│                                                             │
│   4. Backpropagation: Update statistics up tree             │
│      • Q(node) += v                                         │
│      • N(node) += 1                                        │
│                                                             │
│   5. Return: After N simulations, return improved policy   │
│      • π(a) = N(child) / N(root)                            │
│                                                             │
│   Key insight: MCTS + NN > NN alone                        │
│   The tree does the planning, NN does the evaluation.       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation Steps:**

| Step | Task | Estimated Time |
|------|------|----------------|
| 5.1 | Implement `MCTSNode` class with parent/children pointers, Q, N, P fields | 1 hour |
| 5.2 | Implement `select()` with UCT formula and c parameter | 45 min |
| 5.3 | Implement `expand()` with policy network priors | 45 min |
| 5.4 | Implement `evaluate()` with value network | 30 min |
| 5.5 | Implement `backpropagate()` | 20 min |
| 5.6 | Implement `search(root, simulations)` main loop | 30 min |
| 5.7 | Add Dirichlet noise at root node | 20 min |
| 5.8 | Add temperature schedule for move selection | 15 min |
| 5.9 | Test: 100 simulations from start position, verify tree builds | 30 min |
| 5.10 | Implement batched evaluation for M2 Ultra optimization | 2 hours |
| 5.11 | Add virtual loss for parallel search | 1 hour |
| 5.12 | Integrate MCTS into self-play training loop | 1 hour |
| 5.13 | Run experiment TF-B8 (MCTS 50 sims, 10 cycles) | 2 hours |
| 5.14 | Run experiment TF-B9 (MCTS 200 sims, 10 cycles) | 4 hours |
| 5.15 | Analyze: compare MCTS vs no-MCTS performance | 30 min |
| 5.16 | Document results in TRAINING_BOOK | 30 min |
| **Total** | | **~16 hours** |

**Success Criteria:**
- [ ] MCTS completes 50 simulations in < 3 seconds
- [ ] MCTS improves policy vs raw network (KL divergence > 0.1)
- [ ] vs Random 70%+ (from current 50%)
- [ ] vs Heuristic **5%+** (from current 0%) — first wins!
- [ ] No memory leaks during extended search

**M2 Ultra Optimization:**
- Batch evaluate leaves (collect all leaves, evaluate in one forward pass)
- Use Web Workers for parallel simulations
- Virtual loss for parallel MCTS (add −1 to Q during search)

**Code Location:** `src/server/MCTS.js`

---

#### MCTS Implementation Details

**File Structure:**
```
src/server/MCTS.js
├── class MCTSNode {
│   ├── constructor(grid, move, parent, prior)
│   ├── select(c = 1.0)           // UCT selection
│   ├── expand(policy)           // Add children
│   ├── update(value)            // Backprop
│   └── isLeaf()                 // Check for expansion
├── class MCTS {
│   ├── constructor(network, config)
│   ├── search(grid, simulations) // Main entry
│   ├── selectPath(node)          // Selection phase
│   ├── evaluateBatch(nodes)      // Batch NN evaluation
│   └── getPolicy(node, temp)     // Visit count policy
└── exports: { MCTS, MCTSNode }
```

**MCTSNode API:**
```javascript
class MCTSNode {
  constructor(grid, move, parent, prior) {
    this.grid = grid;           // GridManager state
    this.move = move;           // {x, y, rotation} that led here
    this.parent = parent;
    this.children = new Map();  // move -> MCTSNode
    this.P = prior;             // Policy prior from NN
    this.Q = 0;                 // Average value
    this.N = 0;                 // Visit count
    this.player = grid.currentPlayer;
  }

  // UCT selection with exploration constant c
  select(c = 1.5) {
    let bestScore = -Infinity;
    let bestChild = null;
    
    for (const [move, child] of this.children) {
      const uct = child.Q + c * child.P * Math.sqrt(this.N) / (1 + child.N);
      if (uct > bestScore) {
        bestScore = uct;
        bestChild = child;
      }
    }
    return bestChild;
  }

  // Expand with policy vector from NN
  expand(policy) {
    const validMoves = this.grid.getValidMoves();
    for (const move of validMoves) {
      const idx = (move.y * 10 + move.x) * 4 + move.rotation;
      const prior = policy[idx];
      const newGrid = this.grid.clone();
      newGrid.placeTriangle(move.x, move.y, move.rotation, this.player);
      this.children.set(JSON.stringify(move), new MCTSNode(newGrid, move, this, prior));
    }
  }

  // Backpropagation
  update(value) {
    this.N += 1;
    this.Q += (value - this.Q) / this.N;  // Running average
  }
}
```

**MCTS Class API:**
```javascript
class MCTS {
  constructor(network, config = {}) {
    this.network = network;  // TFConvNetwork instance
    this.c = config.c || 1.5;  // Exploration constant
    this.alpha = config.alpha || 0.03;  // Dirichlet noise
    this.epsilon = config.epsilon || 0.25;  // Noise weight
  }

  async search(grid, simulations = 50) {
    const root = new MCTSNode(grid, null, null, 1.0);
    
    // Initial evaluation
    const { policy, value } = await this.network.run(
      encodeCanonical(grid, grid.currentPlayer)
    );
    
    // Add Dirichlet noise at root
    const noisyPolicy = this.addNoise(policy);
    root.expand(noisyPolicy);
    
    // Run simulations
    for (let i = 0; i < simulations; i++) {
      const path = this.selectPath(root);
      const leaf = path[path.length - 1];
      const { value } = await this.network.run(
        encodeCanonical(leaf.grid, leaf.player)
      );
      this.backpropagate(path, value);
    }
    
    return this.getPolicy(root);
  }

  getPolicy(node, temperature = 1.0) {
    const visits = [];
    for (const [move, child] of node.children) {
      visits.push({ move, count: child.N });
    }
    
    if (temperature === 0) {
      // Argmax
      return visits.sort((a, b) => b.count - a.count)[0].move;
    }
    
    // Softmax with temperature
    const probs = visits.map(v => Math.pow(v.count, 1/temperature));
    const sum = probs.reduce((a, b) => a + b, 0);
    const normalized = probs.map(p => p / sum);
    
    // Sample
    const r = Math.random();
    let acc = 0;
    for (let i = 0; i < visits.length; i++) {
      acc += normalized[i];
      if (r < acc) return visits[i].move;
    }
    return visits[visits.length - 1].move;
  }
}
```

**Configuration Parameters:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| c (exploration) | 1.5 | 0.5-3.0 | Higher = more exploration |
| alpha (Dirichlet) | 0.03 | 0.01-0.1 | Higher = more noise at root |
| epsilon (noise weight) | 0.25 | 0.1-0.5 | Higher = more exploration |
| temperature | 1.0 | 0.0-2.0 | Higher = more random move selection |
| batchSize | 8 | 1-32 | Parallel leaf evaluation |

**Temperature Schedule:**
- First 10 moves: temperature = 1.0 (exploratory)
- Later moves: temperature = 0.0 (greedy)

This prevents the AI from playing the same opening every game.

---

### Phase 5.5: Quiescence MCTS — Why It Improves Over Simple MCTS

**Status:** ✅ Completed (`src/server/QuiescenceMCTS.cjs`) — **Broke the Heuristic Wall in `TF-B12` (`63%` vs Heuristic), Reached `89%` Peak in `TF-B13`, and Achieved `90.0%` Peak (`88%` P1 / `92%` P2) & `80.1%` 10-Cycle Avg vs Heuristic in `TF-B14` (`100%` vs Random across all 30 cycles)**

#### 1. The Horizon Effect & Why Simple MCTS Fails Against Deterministic Cascades
In `TF-B10` (Simple MCTS-50 + 50% Heuristic opponents), the AI reached a record **87% vs Random**, yet remained at **0% vs `HeuristicPlayer`** (averaging only ~14 turns to Heuristic's ~96 turns per game). An inflection-point audit of `TF-B10` exposed two structural blind spots in simple MCTS (`MCTS.cjs`):

1. **The Horizon Effect at Unquiet Leaves:**
   Simple MCTS expands a leaf node $s_L$ after a single candidate move and immediately queries the neural Value Head $V_\theta(s_L)$. If $s_L$ is a **"noisy" (tactically unstable) position**—specifically, a single move (`claimed = 0`) that passes the turn to the opponent while leaving a 3-sided enclosure hole—`HeuristicPlayer` will deterministically claim that enclosure with probability $P(\text{capture}) = 1.0$, earn an extra turn via $\lfloor (1 + \text{claimed})/2 \rfloor$, and chain a 5–10 move capture cascade.
2. **Self-Play Value Head Inversion on Single-Move Openings:**
   Because the Value Head was trained primarily on states where local cluster density correlated with winning against non-cascading play, $V_\theta(s_L)$ rated a hole-creating blunder at **`+0.4794`** while rating a safe move at **`+0.1060`**. With 50 simulations spread across ~35 legal moves, simple MCTS never searched deep enough down the forced opponent cascade to see the collapse—instead, UCT **amplified** the `+0.4794` blunder (`13` visits vs `1–2` visits for safe moves).

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Simple MCTS vs. Quiescence MCTS at a Leaf Node              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SIMPLE MCTS (Blind to Forced Cascades):                                │
│  Root ──(Move A: claimed=0, leaves hole)──► Leaf s_L                    │
│    • Evaluates raw V_θ(s_L) = +0.4794 ("cluster grew, looks great!")    │
│    • Backpropagates +0.4794 → UCT visits Move A 13 times                │
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
  Claiming even a single enclosed triangle (`claimed = 1`) awards $\lfloor 2/2 \rfloor = 1$ full extra turn, meaning **the active player retains the turn (`nextPlayer === mover`)**. Even if placing that triangle creates a new 3-sided pocket elsewhere on the board, the AI immediately moves again—allowing it to either **chain an enclosing train in its own favor** or patch the hole before the opponent ever gets a turn. Therefore, `calculateEnclosureRisk` and `calculateChildNodeRisk` strictly enforce `risk = 0` whenever `claimed > 0 || nextPlayer === mover`.
- **Single-Move Openings (`claimed === 0`) Carry All Opponent Risk:**
  Only when `claimed === 0` (and `extraTurns === 0`) does the turn pass to the opponent. If that single move leaves enclosure opportunities for the opponent, the opponent can immediately exploit them and snowball extra turns.

#### 3. Four Mechanisms That Make `QuiescenceMCTS` a Superior Teacher
Recall that during evaluation (`getAIMove`) and browser play (`AIPlayer.js`), the AI uses a **single forward pass of the Policy Head** (no MCTS at test time). MCTS is the **training teacher** that generates target distributions $\pi_{\text{MCTS}}$ and value targets for the 2-head ConvNet. `QuiescenceMCTS` directly upgrades the teacher in four ways without needing a 3rd network head or artificial TD($\lambda$) reward shaping:

1. **Root Prior Shaping (`_expandRootWithRisk`):**
   At the root node, every legal child move records `child.claimed` and `child.enclosureRisk`.
   - Enclosing moves (`child.claimed > 0`) receive a multiplicative prior boost, ensuring the 50 MCTS simulations immediately explore offensive **enclosing trains**.
   - Hole-creating single moves (`child.claimed === 0` and `risk > 0`) have their prior scaled down by $\max(0.05, 1 - 2 \cdot \text{risk} \cdot w_{\text{risk}})$, preventing wasted simulations on obvious blunders.
2. **Pre-Cascade Leaf Expansion + Post-Cascade Quiescent Value (`_runBatch`):**
   During batched MCTS (`searchBatched`), every non-terminal leaf `leaf` is:
   - **Expanded** using the neural policy $P_\theta(\text{leaf.grid})$ of the true **pre-cascade** board so child moves and priors match `leaf.grid`.
   - **Evaluated** by running a deterministic greedy cascade (`_runQuiescenceRollout`, up to `quiescenceDepth = 5`) to reach a quiet board `qEval.grid`, then combining the neural value $V_\theta(\text{qEval.grid})$ with tactical cascade adjustments (`+claimed × enclosureBonusWeight` for AI claims, `-claimed × riskPenaltyWeight` for opponent claims):
     $$V_{\text{leaf}} = \text{clip}_{[-1, 1]}\!\left(V_\theta(s_{\text{quiet}}) + w_{\text{bonus}}\sum \text{claimed}_{\text{AI}} - w_{\text{risk}}\sum \text{claimed}_{\text{opp}}\right)$$
   - Both pre-cascade states (for expansion) and post-cascade states (for value) are packed into a **single combined `network.batchPredict(allStates)` GPU call**.
3. **Direct Policy Target Distillation (`_getRiskAdjustedPolicy`):**
   After all 50 simulations finish, the root visit distribution $\pi(a) \propto N(s_0, a)$ is further refined by rewarding enclosing moves (`child.claimed > 0`) and penalizing risky non-enclosing moves (`child.enclosureRisk.risk > 0`). Because the ConvNet Policy Head trains directly via cross-entropy against $\pi_{\text{MCTS}}$, **the 1-ply Policy Head internalizes both offensive enclosing trains and defensive hole avoidance**.
4. **$O(1)$ Local Neighborhood Enclosure Detection (`countWouldBeEnclosed`):**
   Because `isTriangleEnclosed(cx, cy, rot)` only depends on cell `(cx, cy)` and its 4 orthogonal neighbors, placing a triangle at `(mx, my)` can only enclose `(mx, my)` itself or one of its 4 orthogonal neighbors. Checking just those 5 cells in-place (pushing/popping the candidate triangle without cloning the grid or rebuilding clusters) produces **100% identical results** to full-board simulation while running **3.2× faster** (~140ms per 50-sim move on M2 Ultra).

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
- **Protocol Upgrade (Cycle 8 Onwards & Permanent Default in `train_tfjs.cjs`):**
  Starting at Cycle 8 of `TF-B12` (and now permanently baked into `src/server/train_tfjs.cjs` and `scripts/eval_random_2ply.cjs`), `Eval vs Heuristic` plays **random moves for the first 2 plies (`m < 2`, i.e., 1 random opening triangle per player)** before switching to `exploration = 0` greedy policy vs deterministic `HeuristicPlayer`. This tests the trained network across **100 genuinely distinct opening board configurations** while keeping all mid-game and endgame play strictly deterministic.
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
   - In the current engine (`src/core/GridManager.js` and `game.js`), **edge adjacency** (`getAdjacentTriangles`) and **enclosure detection** (`isTriangleEnclosed`) wrap around the `10×10` grid toroidally via `(x ± 1 + cols) % cols` and `(y ± 1 + rows) % rows`.
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
│     WHY ZERO-PADDING BLINDED TF-B14 AT THE BORDER vs. TOROIDALPAD2D (TF-B15) │
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

**Code Location:** `src/server/ModelPool.js`

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

## 12. Toroidal 3D Visualization, Dual-View Projection & Human Telemetry {#12-toroidal-visualization}

To make the $T^2$ topology intuitive for human players while preserving the mathematical integrity of the core engine and AI tensor pipeline, Triantica enforces a strict **Engine-View Separation Architecture**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 STRICT ENGINE / VIEW-PROJECTION SEPARATION                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CORE ENGINE & AI LAYER (`GridManager.js`, `ConvNetwork.js`, `MCTS.cjs`)    │
│   ───────────────────────────────────────────────────────────────────────    │
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

### Continuous Learning from Human Play (`HumanGameLogger`)
With `TF-B15` deployed to production (`trainedAI_network.json`) across **Easy**, **Medium** (`99.0%` Raw 1-ply ConvNet), and **Hard** (`100.0%` ConvNet + Tactical Safety Filter), the browser and server runtimes include `HumanGameLogger` (`/api/human-games`). Every completed human match records canonical 18-channel states, chosen actions, enclosure claims, and final outcomes—enabling future human-in-the-loop replay mixing without altering the core self-play pipeline.

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
│   ├── TRAINING_BOOK.md           # This comprehensive guide
│   ├── AI_ARCHITECTURE.md         # Deep-dive tensor & network reference
│   ├── EXECUTION_ROADMAP.md       # 2D release & 3D manifold generalization roadmap
│   └── QUIESCENCE_MCTS.md         # Quiescence MCTS mathematical specification
└── trainedAI_network.json         # Production champion weights (TF-B15 Cycle 7: 99% Raw / 100% +Safety)
```

---

## Appendix C: CLI & Dashboard Telemetry Parameters

### 1. Training CLI Flags (`src/server/train_tfjs.cjs`)

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

*Last updated: 2026-10-31*
*Version: 4.0 (Authoritative Complete Edition — Phase 1 to Phase 6 Full Torus $T^2$ Mastery)*
*Framework: TensorFlow.js (`@tensorflow/tfjs-node`) on Apple Silicon (AMX / Accelerate)*
*Hardware: Mac Studio M2 Ultra (24-core CPU, 60-core GPU, 64GB Unified Memory)*
