# Metrics Integration Guide

## Metrics in TF.js Training Pipeline

The TF.js training pipeline (`src/server/train_tfjs.cjs`) includes comprehensive metrics tracking built in. No additional setup needed — all metrics are automatically recorded per cycle.

### Tracked Metrics

| Metric | Location | What it measures |
|--------|----------|------------------|
| `openingDiversity` | `selfPlay` | Shannon entropy of first-move distribution (bits) |
| `boardDiversity` | `selfPlay` | Shannon entropy of final board states (bits) |
| `initialError` | `training` | Loss before training on new cycle data |
| `finalError` | `training` | Loss after training |
| `errorSnapshots` | `training` | Per-iteration loss curve (array) |
| `aiWinRate` | `evaluation` | Win % vs Random opponent |
| `winsByOpponent` | `selfPlay` | Win breakdown by opponent type |
| `scoresByOpponent` | `selfPlay` | Score breakdown by opponent type |
| `opponentTypeCounts` | `selfPlay` | Number of games per opponent type |

### Running with Auditing

```bash
# Enable game auditing to catch training bugs
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --audit true \
  --audit-rate 0.05 \
  --output dashboard/data/experiment.json
```

The audit system validates:
- Games play to completion (~170 moves)
- Both players make moves (no zero-move games)
- Grid is fully filled at game end
- Final cluster sizes are consistent

### Interpreting Metrics

#### Diversity Metrics
- **Opening diversity**: 6.6 bits = maximum (100 unique openings). Below 5.0 = network is stuck in a rut.
- **Board diversity**: Higher is better — means the network is exploring varied strategies.

#### Error Metrics
- **initialError**: Should spike when new data arrives, then `finalError` should be lower after training.
- **Crash-to-zero error**: If `finalError` reaches 0.000000, the network is overfitting. Reduce `--iterations`.
- **Healthy curve**: initialError ~0.01-0.02, finalError ~0.002-0.004.

#### Win Rate
- **vs Random**: Best baseline = 91% (exp20, brain.js). TF.js baselines ~50% (fresh weights, same architecture).
- **vs Heuristic**: Currently 0% across all experiments — the "Heuristic Wall."

### Dashboard

Training results are viewable on the dashboard:
**https://augusto-espinel.github.io/Triantica/dashboard/**

To update dashboard data, output or copy training JSON to `dashboard/data/` and update `dashboard/experiments.json`, then run `./scripts/publish-dashboard.sh` and push to `main`.

### Adding Custom Metrics

To add a new metric, edit `src/server/train_tfjs.cjs`:

1. **Compute the metric** in the cycle loop (around line 700+):
```javascript
const myMetric = computeMyMetric(gameData);
console.log(`📊 My Metric: ${myMetric.toFixed(4)}`);
```

2. **Add to cycle output** in the cycle summary object:
```javascript
selfPlay: {
  // ... existing fields
  myMetric: myMetric
}
```

3. **Update the dashboard** to display the new field in `dashboard/index.html`.
