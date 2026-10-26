#!/usr/bin/env node
/**
 * Evaluate a trained network against Random opponent
 * Usage: node scripts/evaluate.cjs <network.json>
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Import GridManager
const { GridManager } = require(path.join(__dirname, '../src/core/GridManager.js'));

class TFNeuralNetwork {
  constructor(config = {}) {
    this.inputSize = config.inputSize || 401;
    this.hiddenLayers = config.hiddenLayers || [200];
    this.outputSize = config.outputSize || 400;
    this.activation = config.activation || 'sigmoid';
    this.model = null;
  }

  fromJSON(json) {
    tf.tidy(() => {
      this.inputSize = json.options?.inputSize || json.sizes?.[0] || 401;
      this.hiddenLayers = json.options?.hiddenLayers || (json.sizes ? json.sizes.slice(1, -1) : [200]);
      this.outputSize = json.options?.outputSize || json.sizes?.[json.sizes.length - 1] || 400;
      this.activation = json.options?.activation || 'sigmoid';

      this.model = tf.sequential();
      this.model.add(tf.layers.dense({
        inputShape: [this.inputSize],
        units: this.hiddenLayers[0],
        activation: this.activation
      }));
      
      this.model.add(tf.layers.dense({
        units: this.outputSize,
        activation: this.activation
      }));
      
      this.model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });

      // Load weights
      for (let i = 0; i < this.model.layers.length; i++) {
        const brainLayer = json.layers[i] || json.layers[i + 1];
        if (!brainLayer?.weights) continue;
        
        const src = brainLayer.weights[0]?.length || 0;
        const tgt = brainLayer.weights.length || 0;
        
        const k = [];
        for (let s = 0; s < src; s++) {
          k.push([]);
          for (let t = 0; t < tgt; t++) {
            k[s].push(brainLayer.weights[t][s]);
          }
        }
        
        this.model.layers[i].setWeights([
          tf.tensor2d(k, [src, tgt]),
          tf.tensor1d(brainLayer.biases)
        ]);
      }
    });
  }

  run(input) {
    return tf.tidy(() => {
      const t = tf.tensor2d([input], [1, input.length]);
      const out = this.model.predict(t);
      return Array.from(out.dataSync());
    });
  }
}

function getStateRelative(grid) {
  const state = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      const cell = grid.grid[y][x];
      const t1 = cell.triangles[0];
      const t2 = cell.triangles[1];
      state.push(t1?.player === 1 ? 1 : (t1?.player === 2 ? -1 : 0));
      state.push(t1?.rotation ?? -1);
      state.push(t2?.player === 1 ? 1 : (t2?.player === 2 ? -1 : 0));
      state.push(t2?.rotation ?? -1);
    }
  }
  state.push(1);
  return state;
}

function isValidMove(grid, x, y, rotation) {
  return grid.isValidMove(x, y, rotation);
}

function getAIMove(grid, output) {
  const moves = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      for (let rot = 0; rot < 4; rot++) {
        const idx = (y * grid.cols + x) * 4 + rot;
        if (isValidMove(grid, x, y, rot)) {
          moves.push({ x, y, rotation: rot, score: output[idx] });
        }
      }
    }
  }
  if (moves.length === 0) return null;
  moves.sort((a, b) => b.score - a.score);
  return moves[0];
}

function getRandomMove(grid) {
  const moves = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      for (let rot = 0; rot < 4; rot++) {
        if (isValidMove(grid, x, y, rot)) {
          moves.push({ x, y, rotation: rot });
        }
      }
    }
  }
  if (moves.length === 0) return null;
  return moves[Math.floor(Math.random() * moves.length)];
}

async function main() {
  const networkFile = process.argv[2] || 'trainedAI_network.json';
  
  console.log(`Loading network from ${networkFile}...`);
  const data = JSON.parse(fs.readFileSync(networkFile));
  
  const network = new TFNeuralNetwork();
  network.fromJSON(data);
  
  console.log(`Network: ${data.sizes.join(' -> ')}`);
  console.log("\nRunning evaluation vs Random opponent...");
  
  const grid = new GridManager(10, 10);
  let wins = 0, losses = 0, draws = 0;
  const games = 100;
  
  for (let g = 0; g < games; g++) {
    grid.reset();
    let cur = 1, m = 0;
    const maxM = grid.rows * grid.cols * 3;
    
    while (!grid.isGridFull() && m < maxM) {
      grid.currentPlayer = cur;
      const state = getStateRelative(grid);
      
      let move;
      if (cur === 1) {
        const output = network.run(state);
        move = getAIMove(grid, output);
      } else {
        move = getRandomMove(grid);
      }
      
      if (!move) break;
      
      const execRes = grid.executeMove(move.x, move.y, move.rotation, cur);
      if (!execRes.success) break;
      cur = execRes.nextPlayer;
      m++;
    }
    
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    if (s1 > s2) wins++;
    else if (s2 > s1) losses++;
    else draws++;
    
    if ((g + 1) % 10 === 0) process.stdout.write(`${g + 1}/${games} games done\r`);
  }
  
  console.log(`\n\nResults: ${wins}W / ${losses}L / ${draws}D`);
  console.log(`Win rate: ${(wins / games * 100).toFixed(1)}%`);
}

main();
