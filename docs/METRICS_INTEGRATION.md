# Metrics Integration Guide

## Adding MetricsLogger to AIPlayer

### Step 1: Include the Logger

Add to `index.html`:

```html
<script src="MetricsLogger.js"></script>
```

### Step 2: Initialize in AIPlayer

```javascript
class AIPlayer {
    constructor(gridManager, difficulty = 'medium', playerNumber = 2) {
        // ... existing code ...
        
        // Add metrics logging
        this.metrics = new MetricsLogger(`exp-${Date.now()}`);
    }
}
```

### Step 3: Track Game Results

In `runSelfPlayGame()`:

```javascript
async runSelfPlayGame() {
    this.metrics.startGame();
    
    // ... existing game loop ...
    
    // Track each move
    const moveStart = performance.now();
    const move = currentPlayerAI.predictMove();
    const moveTime = performance.now() - moveStart;
    this.metrics.recordMove(moveTime);
    
    // Track claims
    const claims = this.gridManager.claimEnclosedTriangles();
    this.metrics.recordClaims(claims);
    
    // ... at game end ...
    
    this.metrics.endGame(winner);
    this.metrics.printSummary();
}
```

### Step 4: Track Training

In `trainNetwork()`:

```javascript
trainNetwork(iterations = 500, learningRate = 0.1) {
    this.metrics.startTraining(this.trainingData.length);
    
    const result = this.network.train(this.trainingData, {
        iterations,
        learningRate,
        // ... existing options ...
        
        // Add callback for iteration tracking
        callback: (details) => {
            this.metrics.logIteration(details.iterations, details.error);
        }
    });
    
    this.metrics.endTraining(result.error, result.iterations);
}
```

### Step 5: Export Results

```javascript
// In browser console after games:
ai.metrics.downloadLog();

// Or get JSON directly:
const summary = ai.metrics.getSummary();
```

## Quick Test Script

Run 100 self-play games with metrics:

```javascript
async function runBenchmark(ai, numGames = 100) {
    console.log(`Running ${numGames} games...`);
    
    for (let i = 0; i < numGames; i++) {
        await ai.runSelfPlayGame();
        
        if (i % 10 === 0) {
            console.log(`Progress: ${i}/${numGames}`);
        }
    }
    
    ai.metrics.printSummary();
    ai.metrics.downloadLog();
}
```

## Metrics Explained

| Metric | What it measures | Why it matters |
|--------|------------------|----------------|
| Win Rate | % of games won by each player | Overall AI quality |
| Decision Time | ms per move prediction | Network efficiency |
| Training Error | Loss after training | Learning progress |
| Claims/Game | Average enclosures | Strategic behavior |
| Game Length | Moves per game | Play style indicator |

## Interpretation Guide

### Good Signs
- Win rate > 60% vs random opponent
- Decision time < 20ms
- Training error decreasing over sessions
- Consistent claim counts

### Warning Signs
- Win rate < 50% vs random
- Decision time > 100ms (network too large?)
- Training error plateaued or increasing
- Wildly varying game lengths
