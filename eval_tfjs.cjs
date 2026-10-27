const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// TF.js Neural Network wrapper
class TFNeuralNetwork {
  constructor(config = {}) {
    this.inputSize = config.inputSize || 401;
    this.hiddenLayers = config.hiddenLayers || [200];
    this.outputSize = config.outputSize || 400;
    this.model = null;
    this._buildModel();
  }

  _buildModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [this.inputSize],
      units: this.hiddenLayers[0],
      activation: 'sigmoid'
    }));
    this.model.add(tf.layers.dense({
      units: this.outputSize,
      activation: 'sigmoid'
    }));
  }

  run(input) {
    return tf.tidy(() => {
      const t = tf.tensor2d([input], [1, input.length]);
      const out = this.model.predict(t);
      return Array.from(out.dataSync());
    });
  }

  fromJSON(json) {
    tf.tidy(() => {
      for (let i = 0; i < this.model.layers.length; i++) {
        const layer = json.layers[i + 1] || json.layers[i];
        if (!layer?.weights) return;
        const src = layer.weights[0]?.length || 0;
        const tgt = layer.weights.length || 0;
        const k = [];
        for (let s = 0; s < src; s++) {
          k.push([]);
          for (let t = 0; t < tgt; t++) k[s].push(layer.weights[t][s]);
        }
        this.model.layers[i].setWeights([
          tf.tensor2d(k, [src, tgt]),
          tf.tensor1d(layer.biases)
        ]);
      }
    });
  }
}

async function main() {
  const { GridManager } = await import('./src/core/GridManager.js');

  // Load exp20 weights
  const net = new TFNeuralNetwork();
  const weights = JSON.parse(fs.readFileSync(process.argv[2] || 'dashboard/data/TF-B1_pure_network.json', 'utf-8'));
  net.fromJSON(weights);

  // State encoding (relative to current player)
  const getState = (grid) => {
    const state = [];
    for (let r = 0; r < grid.rows; r++) {
      for (let c = 0; c < grid.cols; c++) {
        const cell = grid.grid[r][c];
        for (const t of [cell.triangles[0], cell.triangles[1]]) {
          state.push(t ? (t.player === grid.currentPlayer ? 1 : -1) : 0);
          state.push(t?.rotation !== undefined ? t.rotation / 4 : -1);
        }
      }
    }
    state.push(1); // Always self in relative representation
    return state;
  };

  // Move validation - delegate to GridManager
  const isValid = (grid, x, y, rot) => {
    return grid.isValidMove(x, y, rot);
  };

  const getRandomMove = (grid) => {
    const valid = [];
    for (let y = 0; y < grid.rows; y++) {
      for (let x = 0; x < grid.cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (isValid(grid, x, y, r)) valid.push({ x, y, rotation: r });
        }
      }
    }
    return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
  };

  const getAIMove = (grid) => {
    const state = getState(grid);
    const output = net.run(state);
    const moves = [];
    for (let i = 0; i < 400; i++) {
      const rot = i % 4;
      const idx = Math.floor(i / 4);
      moves.push({ x: idx % grid.cols, y: Math.floor(idx / grid.cols), rotation: rot, score: output[i] });
    }
    moves.sort((a, b) => b.score - a.score);
    for (const m of moves) {
      if (isValid(grid, m.x, m.y, m.rotation)) return m;
    }
    return getRandomMove(grid);
  };

  // Run evaluation
  console.log('Evaluating exp20_mixed (TF.js loaded weights) vs Random: 100 games');
  const grid = new GridManager(10, 10);
  let wins = 0, losses = 0, draws = 0;

  for (let g = 0; g < 100; g++) {
    grid.reset();
    let cur = 1, m = 0;
    while (!grid.isGridFull() && m < 300) {
      grid.currentPlayer = cur;
      const move = cur === 1 ? getAIMove(grid) : getRandomMove(grid);
      if (!move) break;
      const execRes = grid.executeMove(move.x, move.y, move.rotation, cur);
      if (!execRes.success) break;
      cur = execRes.nextPlayer;
      m++;
    }
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    if (s1 > s2) wins++; else if (s2 > s1) losses++; else draws++;

    if ((g + 1) % 10 === 0) process.stdout.write(`\r${g + 1}/100: W:${wins} L:${losses} D:${draws}`);
  }
  console.log(`\n\nResult: ${wins}% vs Random (${wins}W / ${losses}L / ${draws}D)`);
}

main().catch(e => console.error('Error:', e));
