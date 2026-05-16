Here is a structured breakdown of your project, "Triantica," designed to refresh your memory on its architecture, logic, and machine learning components.

### **1. Project Overview**

**Triantica** is a grid-based strategy game built with **Phaser 3**.

* **Goal:** Players place triangles on a $10 \times 10$ grid to form the largest contiguous cluster of their color.
* **Mechanic:** Each grid cell can hold up to two triangles. Connecting triangles merges "clusters." Enclosing an empty space claims it and grants an extra turn.
* **AI:** Includes a neural network (Brain.js) that learns via self-play (Reinforcement Learning).

---

### **2. File Structure & Key Responsibilities**

| File | Primary Responsibility |
| --- | --- |
| **`index.html`** | Entry point. Loads dependencies (`phaser.js`, `brain.js`) and game scripts. |
| **`game.js`** | **The Game Engine.** Contains the Phaser Scenes (`Boot`, `Menu`, `Game`), visual rendering, input handling, and the rules logic (`GridManager`). |
| **`AIPlayer.js`** | **The Brain.** Contains the Neural Network logic, data representation, move prediction, and self-play training loop. |

---

### **3. Logic Breakdown**

#### **A. The Rules Engine (`GridManager` inside `game.js`)**

This class manages the "truth" of the game board, independent of visuals.

* **Data Structure:** The grid is a 2D array where each cell contains a list of triangles.
* **Cluster System (Scoring):**
* Uses a **Union-Find** style approach.
* **`createClusterMap()`**: Tracks which cluster ID every triangle belongs to.
* **`updateClustersForNewTriangle(x, y, ...)`**: When a piece is placed, it checks neighbors. If they belong to different clusters, it **merges** them into one larger cluster ID.


* **Enclosure Mechanic:**
* **`claimEnclosedTriangles()`**: Scans the board for empty slots surrounded by existing triangles. If found, it auto-fills them for the current player and grants `extraTurns`.



#### **B. The Visual Layer (`GameScene` inside `game.js`)**

This handles what the player sees and touches.

* **State Machine:** Handles modes like `HvH` (Human vs Human), `HvAI`, and `AvA`.
* **The "Bridge":** The function **`triggerAIMove()`** is critical.
1. It asks `AIPlayer` for a move.
2. It waits (delayed call) to simulate "thinking" time.
3. It visually places the triangle and updates the `GridManager`.



---

### **4. Machine Learning Elements (`AIPlayer.js`)**

The AI uses a **Feedforward Neural Network** (via `brain.js`) trained using a custom implementation of reinforcement learning.

#### **I. Input & Output (The "Eye" and "Hand")**

* **Input (`_getStateRepresentation`)**: A flat array representing the $10 \times 10$ grid.
* Values are normalized: `1` (Self), `-1` (Opponent), `0` (Empty).
* Rotations are normalized from 0-3 to `0.0` - `0.75`.


* **Output**: A massive array of size $10 \times 10 \times 4$ (400 outputs).
* Each index corresponds to a specific move: `(row * cols + col) * 4 + rotation`.
* The value (0.0 to 1.0) represents the **confidence** that this move yields a reward.



#### **II. Decision Making (`predictMove`)**

The AI uses an **Epsilon-Greedy** strategy:

1. **Exploitation (90%):** Runs the neural network. It sorts all 400 outputs by score and picks the highest one that is **valid** (legal).
2. **Exploration (10%):** Ignores the brain and picks a random legal move to discover new strategies.

#### **III. The Training Loop (`runSelfPlayGame`)**

This is the most complex part of the code. It allows the AI to train itself in the browser.

1. **Simulation:** The AI plays a full game against a clone of itself.
2. **Reward Calculation:**
* **Immediate Reward:** Did this move increase my cluster size? Did I capture enclosed space? (Calculated by `calculateImmediateRewardVector`).
* **Final Outcome:** Did I win ($+1$) or lose ($-1$)?


3. **Backpropagation:**
* It takes the final game result and "discounts" it backwards.
* Moves leading to a win get a small bonus added to their target score.
* Moves leading to a loss get a penalty.


4. **Training:** The collected data (State + Calculated Reward) is fed into `network.train()`.

### **5. Critical Functions Cheat Sheet**

| Function | Location | What it does |
| --- | --- | --- |
| **`GridManager.updateClustersForNewTriangle`** | `game.js` | **The Core Mechanic.** Handles merging groups of triangles so scoring works. |
| **`AIPlayer.calculateImmediateRewardVector`** | `AIPlayer.js` | **The "Planner".** Simulates *every* possible legal move on a temp board to see which one gives immediate points. |
| **`AIPlayer.trainNetwork`** | `AIPlayer.js` | **The Learner.** Feeds the history of the self-play game into Brain.js to adjust weights. |
| **`importNetwork`** | `AIPlayer.js` | **Persistence.** Tries to load a trained brain from a local JSON file or `localStorage`. |
