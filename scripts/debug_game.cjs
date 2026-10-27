#!/usr/bin/env node
/**
 * Debug script: Run one AI vs Heuristic game with full move logging
 * 
 * Usage: node scripts/debug_game.cjs
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Dynamic imports
let GridManager, HeuristicPlayer;

// TFNeuralNetwork class (same as train_tfjs.cjs)
class TFNeuralNetwork {
  constructor(config = {}) {
    this.inputSize = config.inputSize || 401;
    this.hiddenLayers = config.hiddenLayers || [200];
    this.outputSize = config.outputSize || 400;
    this.activation = config.activation || 'sigmoid';
    this.model = null;
    this._buildModel();
  }

  _buildModel() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({
      inputShape: [this.inputSize],
      units: this.hiddenLayers[0],
      activation: this.activation
    }));
    for (let i = 1; i < this.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.hiddenLayers[i],
        activation: this.activation
      }));
    }
    this.model.add(tf.layers.dense({
      units: this.outputSize,
      activation: this.activation
    }));
    this.model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });
  }

  run(input) {
    // Simpler approach: use model.predict directly
    const inputTensor = tf.tensor1d(input);
    const output = this.model.predict(inputTensor.reshape([1, input.length]));
    const result = Array.from(output.dataSync());
    inputTensor.dispose();
    output.dispose();
    return result;
  }

  fromJSON(json) {
    tf.tidy(() => {
      this.inputSize = json.options?.inputSize || json.sizes?.[0] || 401;
      this.hiddenLayers = json.options?.hiddenLayers || (json.sizes ? json.sizes.slice(1, -1) : [200]);
      this.outputSize = json.options?.outputSize || json.sizes?.[json.sizes.length - 1] || 400;
      this.activation = json.options?.activation || 'sigmoid';
      this._buildModel();

      for (let i = 0; i < this.model.layers.length; i++) {
        const brainLayer = json.layers[i];
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
}

async function main() {
  // Load modules
  const gridModule = await import('../src/core/GridManager.js');
  GridManager = gridModule.GridManager;
  const heuristicModule = await import('../src/players/HeuristicPlayer.js');
  HeuristicPlayer = heuristicModule.HeuristicPlayer;

  console.log('🎮 Debug Game: AI vs Heuristic\n');

  // Load latest trained network
  const networkPath = path.resolve(__dirname, '../dashboard/data/TF-B1_pure_network.json');
  let network = null;
  
  if (fs.existsSync(networkPath)) {
    const data = JSON.parse(fs.readFileSync(networkPath, 'utf-8'));
    network = new TFNeuralNetwork();
    network.fromJSON(data);
    console.log(`📂 Loaded network from ${path.basename(networkPath)}\n`);
  } else {
    console.log('⚠️  No trained network found, using random moves\n');
  }

  // Run a single game with full logging
  const grid = new GridManager(10, 10);
  const heuristic = new HeuristicPlayer(2, { explorationFactor: 0 });
  
  let cur = 1;
  let moveNum = 0;
  const trace = [];
  
  console.log('Move | Player | Type      | Position | Rot | Enclosed | Cluster | Extra');
  console.log('-'.repeat(75));
  
  // Only show first 30 moves in detail
  let verboseLimit = 30;
  
  while (!grid.isGridFull() && moveNum < 300) {
    moveNum++;
    grid.currentPlayer = cur;
    
    let move, moveType;
    if (cur === 1) {
      // AI move
      if (network) {
        console.log(`\n  === AI thinking (Move ${moveNum}) ===`);
        move = getAIMove(network, grid, 0); // No exploration
        moveType = 'AI';
      } else {
        move = getRandomMove(grid);
        moveType = 'RAND';
      }
    } else {
      // Heuristic move
      move = heuristic.getMove(grid);
      moveType = 'HEUR';
    }
    
    if (!move) {
      console.log(`No valid move for P${cur} at move ${moveNum}`);
      break;
    }
    
    // Place and track
    const prevClusters = { ...grid.clusters };
    grid.selectedRotation = move.rotation;
    const success = grid.placeTriangle(move.x, move.y, move.rotation, cur);
    
    if (!success) {
      console.log(`❌ FAILED: P${cur} (${moveType}) at (${move.x},${move.y}) r${move.rotation}`);
      continue;
    }
    
    const enclosed = grid.claimEnclosedTriangles(cur);
    const clusterSize = grid.getConnectedClusterSize(move.x, move.y, move.rotation);
    
    // Extra turns
    const prevExtra = grid.extraTurns;
    grid.extraTurns += Math.floor((1 + enclosed) / 2);
    if (grid.extraTurns > 0) grid.extraTurns--;
    else cur = cur === 1 ? 2 : 1;
    
    // Log move
    const logLine = `${String(moveNum).padStart(4)} | P${cur}     | ${moveType.padEnd(9)} | (${move.x},${move.y})    | ${move.rotation}   | ${String(enclosed).padStart(7)} | ${String(clusterSize).padStart(7)} | ${grid.extraTurns > 0 ? 'YES' : 'no'}`;
    trace.push(logLine);
    if (moveNum <= verboseLimit) {
      console.log(logLine);
    } else if (moveNum === verboseLimit + 1) {
      console.log('... (remaining moves truncated)');
    }
  }
  
  // Final scores
  const s1 = grid.getLargestContiguous(1);
  const s2 = grid.getLargestContiguous(2);
  
  console.log('\n' + '='.repeat(75));
  console.log(`Final Score: AI(P1)=${s1}, Heuristic(P2)=${s2}`);
  console.log(`Winner: ${s1 > s2 ? 'AI' : (s2 > s1 ? 'Heuristic' : 'Draw')}`);
  console.log(`Total moves: ${moveNum}`);
  
  // Print cluster summary
  console.log('\n📊 Cluster Summary:');
  for (const [id, info] of Object.entries(grid.clusters)) {
    console.log(`  Cluster ${id}: P${info.player}, size=${info.size}`);
  }
  
  // Print board visualization
  console.log('\n📋 Final Board:');
  printBoard(grid);
}

function getStateFixed(grid, aiPlayerNumber) {
  const state = [];
  const { rows, cols } = grid;
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid.grid[r][c];
      const t1 = cell.triangles[0], t2 = cell.triangles[1];
      
      const pv = t => {
        if (!t) return 0;
        return t.player === aiPlayerNumber ? 1 : -1;
      };
      
      const rv = r => r === undefined || r === -1 ? -1 : r / 4;
      
      state.push(pv(t1), rv(t1?.rotation ?? -1), pv(t2), rv(t2?.rotation ?? -1));
    }
  }
  state.push(1);
  return state;
}

function getAIMove(network, grid, exploration) {
  if (Math.random() < exploration) return getRandomMove(grid);
  
  // Use fixed perspective from AI's view
  const aiPlayer = grid.currentPlayer;
  const state = getStateFixed(grid, aiPlayer);
  
  const output = network.run(state);
  
  // Sort by score
  const moves = [];
  for (let i = 0; i < output.length; i++) {
    const rotation = i % 4;
    const cellIndex = Math.floor(i / 4);
    const x = cellIndex % 10;
    const y = Math.floor(cellIndex / 10);
    moves.push({ x, y, rotation, score: output[i] });
  }
  moves.sort((a, b) => b.score - a.score);
  
  // DEBUG: Print top 10 predictions
  console.log('   Top predictions:');
  let count = 0;
  for (const move of moves) {
    const valid = grid.isValidMove(move.x, move.y, move.rotation);
    console.log(`     (${move.x},${move.y}) r${move.rotation} score=${move.score.toFixed(3)} ${valid ? '✓' : '✗'}`);
    count++;
    if (count >= 10) break;
  }
  
  // Find first valid
  for (const move of moves) {
    if (grid.isValidMove(move.x, move.y, move.rotation)) {
      return move;
    }
  }
  
  return getRandomMove(grid);
}

function getRandomMove(grid) {
  const valid = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      for (let r = 0; r < 4; r++) {
        if (grid.isValidMove(x, y, r)) valid.push({ x, y, rotation: r });
      }
    }
  }
  return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
}

function printBoard(grid) {
  // Visualize: each cell shows which triangles are present
  const symbols = {
    '00': '· ', // empty
    '10': '◢ ', // P1 first triangle only
    '01': '·◣', // P2 first triangle only  
    '11': '██', // both players in same cell (shouldn't happen normally)
  };
  
  for (let y = 0; y < grid.rows; y++) {
    let row = '';
    for (let x = 0; x < grid.cols; x++) {
      const cell = grid.grid[y][x];
      const t1 = cell.triangles[0];
      const t2 = cell.triangles[1];
      
      if (!t1 && !t2) {
        row += '· ';
      } else if (t1 && !t2) {
        row += t1.player === 1 ? '1 ' : '2 ';
      } else if (!t1 && t2) {
        row += t2.player === 1 ? '1 ' : '2 ';
      } else {
        // Both triangles present
        if (t1.player === t2.player) {
          row += t1.player === 1 ? '█1' : '█2';
        } else {
          row += '██';  // Mixed cell
        }
      }
    }
    console.log(row);
  }
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
