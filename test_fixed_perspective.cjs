#!/usr/bin/env node
/**
 * Quick test of fixed perspective training
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function main() {
  console.log('=== Fixed Perspective Training Test ===\n');
  
  // Import modules
  const gridModule = await import('./src/core/GridManager.js');
  const GridManager = gridModule.GridManager;
  const heuristicModule = await import('./src/players/HeuristicPlayer.js');
  const HeuristicPlayer = heuristicModule.HeuristicPlayer;
  
  console.log('✓ Modules loaded');
  
  // Simple TF.js network for testing
  class SimpleNetwork {
    constructor() {
      this.model = tf.sequential();
      this.model.add(tf.layers.dense({
        inputShape: [401],
        units: 200,
        activation: 'sigmoid'
      }));
      this.model.add(tf.layers.dense({
        units: 400,
        activation: 'sigmoid'
      }));
      this.model.compile({
        optimizer: tf.train.adam(0.1),
        loss: 'meanSquaredError'
      });
    }
    
    run(state) {
      return tf.tidy(() => {
        const input = tf.tensor2d([state], [1, 401]);
        const output = this.model.predict(input);
        return Array.from(output.dataSync());
      });
    }
    
    async train(states, targets) {
      const xs = tf.tensor2d(states);
      const ys = tf.tensor2d(targets);
      const result = await this.model.fit(xs, ys, {
        epochs: 10,
        batchSize: Math.min(32, states.length),
        verbose: 0
      });
      xs.dispose();
      ys.dispose();
      return result.history.loss[0];
    }
  }
  
  const network = new SimpleNetwork();
  const grid = new GridManager(10, 10);
  
  // Fixed perspective state encoding
  function getStateFixed(grid, aiPlayerNumber) {
    const state = [];
    const { rows, cols } = grid;
    const opponent = aiPlayerNumber === 1 ? 2 : 1;
    
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
  
  function getRandomMove(grid) {
    const valid = [];
    for (let y = 0; y < grid.rows; y++) {
      for (let x = 0; x < grid.cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (grid.isValidMove(x, y, r)) {
            valid.push({ x, y, rotation: r });
          }
        }
      }
    }
    return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
  }
  
  function getAIMove(network, grid, aiPlayerNumber) {
    const state = getStateFixed(grid, aiPlayerNumber);
    const output = network.run(state);
    const { rows, cols } = grid;
    
    // Find best valid move
    let best = null, bestVal = -Infinity;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (grid.isValidMove(x, y, r)) {
            const idx = (y * cols + x) * 4 + r;
            if (output[idx] > bestVal) {
              bestVal = output[idx];
              best = { x, y, rotation: r };
            }
          }
        }
      }
    }
    return best || getRandomMove(grid);
  }
  
  // Play a game
  function playGame(network, grid, aiPlayerNumber, exploration = 0.3) {
    grid.reset();
    let curP = 1;
    const history = [];
    const maxMoves = grid.rows * grid.cols * 3;
    let m = 0;
    
    while (!grid.isGridFull() && m < maxMoves) {
      grid.currentPlayer = curP;
      const isAI = curP === aiPlayerNumber;
      
      let move;
      if (isAI && Math.random() > exploration) {
        move = getAIMove(network, grid, aiPlayerNumber);
      } else {
        move = getRandomMove(grid);
      }
      
      if (!move) break;
      
      const state = getStateFixed(grid, aiPlayerNumber);
      const moveIdx = (move.y * grid.cols + move.x) * 4 + move.rotation;
      
      grid.selectedRotation = move.rotation;
      if (!grid.placeTriangle(move.x, move.y)) break;
      
      grid.claimEnclosedTriangles();
      
      history.push({ state, moveIdx, player: curP });
      
      // Extra turn logic
      grid.extraTurns += Math.floor((1 + 0) / 2);
      if (grid.extraTurns > 0) grid.extraTurns--;
      else curP = curP === 1 ? 2 : 1;
      
      m++;
    }
    
    // Determine winner
    let winner = 0;
    if (grid.isGridFull()) {
      const s1 = grid.getLargestContiguous(1);
      const s2 = grid.getLargestContiguous(2);
      winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
    }
    
    return { history, winner, aiPlayerNumber };
  }
  
  // Training loop
  console.log('\n🔄 Training with fixed perspective...\n');
  
  let bestWinRate = 0;
  
  for (let cycle = 1; cycle <= 5; cycle++) {
    const states = [], targets = [];
    let aiWins = 0, totalGames = 0;
    
    // Self-play: AI alternates between P1 and P2
    for (let g = 0; g < 20; g++) {
      const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;
      const { history, winner, aiPlayerNumber: aiP } = playGame(network, grid, aiPlayerNumber, 0.3);
      
      // Build training data from AI's perspective only
      for (const step of history) {
        if (step.player === aiP) {
          const finalReward = winner === aiP ? 0.99 : (winner === 0 ? 0.5 : 0.01);
          const output = new Array(400).fill(0.01);
          output[step.moveIdx] = finalReward;
          states.push(step.state);
          targets.push(output);
        }
      }
      
      // Track wins
      totalGames++;
      if (winner === aiP) aiWins++;
    }
    
    // Train
    const loss = await network.train(states, targets);
    const winRate = (aiWins / totalGames * 100).toFixed(1);
    
    if (parseFloat(winRate) > bestWinRate) bestWinRate = parseFloat(winRate);
    
    console.log(`Cycle ${cycle}: Loss=${loss.toFixed(6)}, Win Rate vs Random=${winRate}% (${aiWins}/${totalGames})`);
  }
  
  // Evaluation: vs Random (100 games, no exploration)
  console.log('\n📊 Final Evaluation (vs Random, 100 games):');
  let evalWins = 0;
  
  for (let g = 0; g < 100; g++) {
    const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;
    const { winner, aiPlayerNumber: aiP } = playGame(network, grid, aiPlayerNumber, 0);
    if (winner === aiP) evalWins++;
  }
  
  console.log(`   Win Rate: ${(evalWins / 100 * 100).toFixed(1)}%`);
  
  // Evaluation: vs Heuristic (100 games)
  console.log('\n📊 Final Evaluation (vs Heuristic, 100 games):');
  let heurWins = 0;
  const heuristic = new HeuristicPlayer(2, {});
  
  for (let g = 0; g < 100; g++) {
    const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;
    grid.reset();
    let curP = 1;
    let m = 0;
    const maxMoves = grid.rows * grid.cols * 3;
    
    while (!grid.isGridFull() && m < maxMoves) {
      grid.currentPlayer = curP;
      const isAI = curP === aiPlayerNumber;
      
      let move;
      if (isAI) {
        move = getAIMove(network, grid, aiPlayerNumber);
      } else {
        move = heuristic.getMove(grid);
      }
      
      if (!move) break;
      
      grid.selectedRotation = move.rotation;
      if (!grid.placeTriangle(move.x, move.y)) break;
      
      grid.claimEnclosedTriangles();
      
      grid.extraTurns += Math.floor((1 + 0) / 2);
      if (grid.extraTurns > 0) grid.extraTurns--;
      else curP = curP === 1 ? 2 : 1;
      
      m++;
    }
    
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    const winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
    
    if (winner === aiPlayerNumber) heurWins++;
  }
  
  console.log(`   Win Rate: ${(heurWins / 100 * 100).toFixed(1)}%`);
  
  console.log('\n✅ Test complete');
  console.log(`\n📈 Best training win rate: ${bestWinRate}%`);
  console.log(`📊 Final vs Random: ${(evalWins / 100 * 100).toFixed(1)}%`);
  console.log(`📊 Final vs Heuristic: ${(heurWins / 100 * 100).toFixed(1)}%`);
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
