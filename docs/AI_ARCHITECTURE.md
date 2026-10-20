# AI Architecture Technical Reference

**Version:** 1.0  
**Last Updated:** 2026-10-18  
**Status:** Current implementation analysis

---

## Neural Network Specification

### Architecture Overview

```
Input Layer:     401 neurons (flattened state + player flag)
Hidden Layers:   Configurable by difficulty
Output Layer:    400 neurons (move confidence scores)
Activation:      Sigmoid (all layers)
```

### Difficulty Configurations

| Difficulty | Hidden Layers | Total Weights | Approx Size |
|------------|---------------|---------------|-------------|
| Easy       | [100]         | 401×100 + 100×400 = 80,100 | ~320 KB |
| Medium     | [200]         | 401×200 + 200×400 = 160,200 | ~640 KB |
| Hard       | [267, 160]    | 401×267 + 267×160 + 160×400 = 160,667 | ~640 KB |

Current saved network: ~5MB (suggests additional metadata or different config)

---

## Input Encoding

### State Vector Construction

```javascript
// Pseudocode for state representation
const state = [];

for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
        const cell = grid[y][x];
        const t1 = cell.triangles[0];  // First triangle (if any)
        const t2 = cell.triangles[1];  // Second triangle (if any)
        
        // Encode player ownership
        state.push(normalizePlayer(t1?.player));    // 1=AI, -1=opponent, 0=empty
        state.push(normalizeRotation(t1?.rotation)); // 0.0-0.75 or -1
        
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation));
    }
}

// Add current player context
state.push(currentPlayer === aiPlayerNumber ? 1 : -1);
```

### Encoding Functions

```javascript
normalizePlayer(p) {
    if (!p) return 0;
    return p === this.playerNumber ? 1 : -1;
}

normalizeRotation(r) {
    if (r === undefined || r === -1) return -1;
    return r / 4.0;  // Maps {0,1,2,3} → {0.0, 0.25, 0.5, 0.75}
}
```

### Input Vector Layout

```
Index 0-3:     Cell (0,0) - [P1, R1, P2, R2]
Index 4-7:     Cell (1,0) - [P1, R1, P2, R2]
...
Index 396-399: Cell (9,9) - [P1, R1, P2, R2]
Index 400:     Current player flag
```

---

## Output Encoding

### Output Vector Interpretation

```javascript
// Output index → move mapping
const i = outputIndex;
const rotation = i % 4;           // 0-3
const cellIndex = Math.floor(i / 4);
const x = cellIndex % cols;        // 0-9
const y = Math.floor(cellIndex / cols); // 0-9
```

### Output Value Semantics

| Value | Meaning |
|-------|---------|
| ~0.01 | Invalid move (default for illegal positions) |
| ~0.50 | Neutral expected value |
| ~0.99 | Highly confident good move |

### Move Selection Process

```javascript
predictMove() {
    // Epsilon-greedy: 10% exploration
    if (Math.random() < explorationFactor) {
        return getRandomValidMove();
    }
    
    // Get network output
    const output = network.run(state);
    
    // Sort all moves by confidence
    const moves = output.map((score, i) => ({
        ...indexToMove(i),
        score
    })).sort((a, b) => b.score - a.score);
    
    // Return highest-scoring valid move
    for (const move of moves) {
        if (isValidMove(move)) return move;
    }
    
    // Fallback
    return getRandomValidMove();
}
```

---

## Reward Calculation

### Immediate Reward Components

The AI evaluates every legal move by simulating it on a temporary board:

```javascript
calculateImmediateRewardVector(state, player) {
    const output = new Array(400).fill(0.01);  // Default: invalid
    
    for (each cell position) {
        for (each rotation 0-3) {
            if (isLegalMove(x, y, rotation)) {
                // Create temporary board
                const tempGrid = clone(currentGrid);
                
                // Simulate placement
                tempGrid.placeTriangle(x, y, rotation, player);
                const claims = tempGrid.claimEnclosedTriangles();
                
                // Calculate reward
                const claimReward = claims * 0.3;
                const clusterReward = tempGrid
                    .getConnectedClusterSize(x, y, rotation) * 0.1;
                
                const totalReward = claimReward + clusterReward;
                
                // Scale to [0.01, 0.99]
                output[index] = scaleReward(totalReward);
            }
        }
    }
    
    return output;
}
```

### Reward Scaling

```javascript
// REALISTIC bounds (Exp 10+ — not theoretical maximums!)
const maxClaimBonus = 1.5;    // max ~5 claims × 0.3
const maxClusterBonus = 10.0;  // realistic ~100 triangles × 0.1
const maxLeadBonus = 4.0;      // realistic ±20 lead delta × 0.2 (Exp 10+)
const maxImmediateReward = 1.5 + 10.0 + 4.0;  // = 15.5
const minReward = -5.0;        // realistic worst case
const rewardRange = maxImmediateReward - minReward;  // = 20.5

scaledReward = (reward - minReward) / rewardRange;
scaledReward = Math.max(0.01, Math.min(0.99, scaledReward));
```

> ⚠️ **Critical:** Never use theoretical maximums for normalization. The lead bonus theoretical max is 40 (200×0.2) but realistic max is ~4. Using theoretical values compresses 95% of the output range into unused territory, causing fake convergence.

### Final Outcome Integration

After a self-play game completes, the training data is adjusted:

```javascript
// Backward iteration through game history
for (let t = totalMoves - 1; t >= 0; t--) {
    const step = gameHistory[t];
    
    // Linear discount based on position
    const discountFactor = (t + 1) / totalMoves;
    const scaledFinalReward = finalOutcome * discountFactor;
    
    // Combine with immediate rewards
    const outputVector = [...step.immediateRewardVector];
    const combinedReward = outputVector[step.actualMoveIndex] + scaledFinalReward;
    
    // Re-scale to [0.01, 0.99]
    outputVector[step.actualMoveIndex] = scaleCombinedReward(combinedReward);
    
    trainingData.push({
        input: step.state,
        output: outputVector
    });
}
```

---

## Training Loop

### Self-Play Game Execution

```javascript
async runSelfPlayGame() {
    const gameHistory = [];
    const opponentAI = new AIPlayer(gridManager, difficulty, opponentNumber);
    opponentAI.network.fromJSON(this.network.toJSON());
    
    while (!gridManager.isGridFull()) {
        const state = getCurrentState();
        const currentPlayerAI = getCurrentPlayer();
        
        // Calculate immediate rewards for training labels
        const immediateRewards = calculateImmediateRewardVector(state, player);
        
        // Select move
        let move;
        if (networkLoaded && movesCount > 0) {
            if (Math.random() >= immediacyFactor) {
                move = currentPlayerAI.predictMove();  // Use network
            } else {
                move = selectByImmediateReward(immediateRewards);  // Greedy
            }
        } else {
            move = currentPlayerAI.getRandomValidMove();  // First move or untrained
        }
        
        // Execute move
        gridManager.selectedRotation = move.rotation;
        gridManager.placeTriangle(move.x, move.y);
        const claims = gridManager.claimEnclosedTriangles();
        
        // Record step
        gameHistory.push({
            state,
            move,
            player: currentPlayer,
            claimed: claims,
            immediateRewardVector: immediateRewards,
            actualMoveIndex: moveToIndex(move)
        });
        
        // Handle turn switching
        gridManager.extraTurns += (claims + 1) / 2;
        if (gridManager.extraTurns === 0) {
            switchPlayer();
        }
    }
    
    // Determine winner and process rewards
    const winner = determineWinner();
    processGameHistory(gameHistory, winner);
    gridManager.reset();
}
```

### Batch Training

```javascript
trainNetwork(iterations = 500, learningRate = 0.1) {
    const result = network.train(trainingData, {
        iterations,
        learningRate,
        errorThresh: 0.00093,
        log: (details) => console.log(`Iteration ${details.iterations}: Error ${details.error}`),
        logPeriod: 10
    });
    
    if (result.error < 1) {
        isNetworkLoaded = true;
    }
    
    return result;
}
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| State encoding | O(rows × cols) | < 1 ms |
| Network forward pass | O(weights) | ~5-10 ms |
| Move validation | O(1) per move | < 1 ms |
| Immediate reward calc | O(moves × weight) | ~200-500 ms |
| Training iteration | O(samples × weights) | varies |

### Memory Usage

- Network weights: 160K parameters × 8 bytes ≈ 1.3 MB
- Training data buffer: variable (depends on games stored)
- Per-game state: 401 floats × 200 moves ≈ 320 KB

### Bottlenecks

1. **Immediate reward calculation**: Simulates every legal move
2. **Training iterations**: Can be slow with large datasets
3. **Grid cloning**: JSON.stringify/parse for deep copy

---

## Known Issues & Limitations

### Current Implementation

1. **No metrics logging**: Can't track performance over time
2. **Blocking training**: Runs on main thread, freezes UI
3. **Fixed hyperparameters**: Can't tune without code changes
4. **No early stopping**: Runs all iterations regardless
5. **Memory leak potential**: Training data accumulates

### Brain.js Limitations

1. **No GPU acceleration**: All CPU-based
2. **Limited architecture options**: Only feedforward/LSTM
3. **No batch normalization**: Manual scaling required
4. **No learning rate scheduling**: Fixed rate

---

## Extension Points

### Adding Metrics

```javascript
class AIPlayer {
    constructor() {
        this.metrics = {
            gamesPlayed: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            totalMoves: 0,
            totalClaims: 0,
            avgDecisionTime: 0,
            trainingErrors: []
        };
    }
    
    recordGameResult(winner) {
        this.metrics.gamesPlayed++;
        if (winner === this.playerNumber) this.metrics.wins++;
        else if (winner === 0) this.metrics.draws++;
        else this.metrics.losses++;
    }
}
```

### Web Worker Training

```javascript
// worker.js
self.onmessage = (e) => {
    const { trainingData, config } = e.data;
    const net = new brain.NeuralNetwork(config.architecture);
    const result = net.train(trainingData, config.trainOpts);
    self.postMessage({ weights: net.toJSON(), result });
};
```

### Configurable Hyperparameters

```javascript
const AI_CONFIG = {
    explorationFactor: 0.10,
    explorationMode: 'perMove',    // 'perMove' | 'perGame' (Exp 10+)
    immediacyFactor: 0.20,
    claimRewardFactor: 0.30,
    clusterRewardFactor: 0.10,
    largestLeadFactor: 0.20,       // Exp 10+ — largest-cluster-lead reward
    finalRewardWeight: 1.0,
    trainingIterations: 500,
    learningRate: 0.10
};
```

---

## References

- [Brain.js Documentation](https://github.com/BrainJS/brain.js)
- [Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book.html)
- [Phaser 3 Documentation](https://photonstorm.github.io/phaser3-docs/)
