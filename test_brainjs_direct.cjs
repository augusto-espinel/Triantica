#!/usr/bin/env node
/**
 * Test original brain.js network performance
 */

const brain = require('brain.js');
const fs = require('fs');
const path = require('path');

// Import GridManager
const { GridManager } = require('./src/core/GridManager.js');

function getStateFixed(grid, aiPlayerNumber) {
  const state = [];
  for (let y = 0; y < grid.rows; y++) {
    for (let x = 0; x < grid.cols; x++) {
      const cell = grid.grid[y][x];
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
        if (grid.isValidMove(x, y, r)) valid.push({ x, y, rotation: r });
      }
    }
  }
  return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
}

function getAIMove(network, grid, aiPlayerNumber) {
  const state = getStateFixed(grid, aiPlayerNumber);
  const output = network.run(state);
  
  const moves = [];
  for (let i = 0; i < output.length; i++) {
    const rot = i % 4;
    const x = Math.floor(i / 4) % 10;
    const y = Math.floor(i / 40);
    moves.push({ x, y, rotation: rot, score: output[i] });
  }
  
  moves.sort((a, b) => b.score - a.score);
  
  for (const m of moves) {
    if (grid.isValidMove(m.x, m.y, m.rotation)) return m;
  }
  
  return getRandomMove(grid);
}

class HeuristicPlayer {
  constructor(playerNumber) {
    this.playerNumber = playerNumber;
  }
  
  getMove(grid) {
    const valid = [];
    for (let y = 0; y < grid.rows; y++) {
      for (let x = 0; x < grid.cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (grid.isValidMove(x, y, r)) {
            const temp = grid.clone();
            temp.currentPlayer = this.playerNumber;
            temp.selectedRotation = r;
            temp.placeTriangle(x, y, this.playerNumber);
            const claimed = temp.claimEnclosedTriangles(this.playerNumber);
            const clusterSize = temp.getConnectedClusterSize(x, y, r);
            valid.push({ x, y, rotation: r, claimed, clusterSize, score: claimed * 10 + clusterSize });
          }
        }
      }
    }
    
    if (!valid.length) return null;
    valid.sort((a, b) => b.score - a.score);
    return valid[0];
  }
}

async function main() {
  console.log('=== Original Brain.js Network Evaluation ===\n');
  
  // Load network
  const networkPath = path.join(__dirname, 'dist', 'trainedAI_network.json');
  console.log('Loading network from:', networkPath);
  
  const data = JSON.parse(fs.readFileSync(networkPath, 'utf8'));
  const network = new brain.NeuralNetwork();
  network.fromJSON(data);
  
  console.log('Network loaded successfully');
  console.log('Architecture:', data.sizes?.join(' -> ') || 'unknown');
  console.log('');
  
  const grid = new GridManager(10, 10);
  
  // Test vs Random
  console.log('📊 Testing vs Random (100 games)...');
  let wins = 0, losses = 0, draws = 0;
  
  for (let g = 0; g < 100; g++) {
    grid.reset();
    let cur = 1;
    const aiPlayer = (g % 2 === 0) ? 1 : 2;
    
    while (!grid.isGridFull()) {
      grid.currentPlayer = cur;
      
      let move;
      if (cur === aiPlayer) {
        move = getAIMove(network, grid, aiPlayer);
      } else {
        move = getRandomMove(grid);
      }
      
      if (!move) break;
      
      grid.selectedRotation = move.rotation;
      if (!grid.placeTriangle(move.x, move.y)) break;
      
      const claimed = grid.claimEnclosedTriangles();
      if (claimed > 0) {
        // Extra turn
      } else {
        cur = cur === 1 ? 2 : 1;
      }
    }
    
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    
    const aiScore = aiPlayer === 1 ? s1 : s2;
    const oppScore = aiPlayer === 1 ? s2 : s1;
    
    if (aiScore > oppScore) wins++;
    else if (oppScore > aiScore) losses++;
    else draws++;
    
    if ((g + 1) % 20 === 0) process.stdout.write(`.`);
  }
  
  console.log(`\n   Wins: ${wins} | Losses: ${losses} | Draws: ${draws}`);
  console.log(`   Win Rate: ${wins}%\n`);
  
  // Test vs Heuristic
  console.log('📊 Testing vs Heuristic (100 games)...');
  wins = 0; losses = 0; draws = 0;
  const heuristic = new HeuristicPlayer(2);
  
  for (let g = 0; g < 100; g++) {
    grid.reset();
    let cur = 1;
    const aiPlayer = (g % 2 === 0) ? 1 : 2;
    const heurPlayer = aiPlayer === 1 ? 2 : 1;
    
    const heuristicForGame = new HeuristicPlayer(heurPlayer);
    
    while (!grid.isGridFull()) {
      grid.currentPlayer = cur;
      
      let move;
      if (cur === aiPlayer) {
        move = getAIMove(network, grid, aiPlayer);
      } else {
        move = heuristicForGame.getMove(grid);
      }
      
      if (!move) break;
      
      grid.selectedRotation = move.rotation;
      if (!grid.placeTriangle(move.x, move.y)) break;
      
      const claimed = grid.claimEnclosedTriangles();
      if (claimed > 0) {
        // Extra turn
      } else {
        cur = cur === 1 ? 2 : 1;
      }
    }
    
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    
    const aiScore = aiPlayer === 1 ? s1 : s2;
    const oppScore = aiPlayer === 1 ? s2 : s1;
    
    if (aiScore > oppScore) wins++;
    else if (oppScore > aiScore) losses++;
    else draws++;
    
    if ((g + 1) % 20 === 0) process.stdout.write(`.`);
  }
  
  console.log(`\n   Wins: ${wins} | Losses: ${losses} | Draws: ${draws}`);
  console.log(`   Win Rate: ${wins}%\n`);
  
  console.log('=== Summary ===');
  console.log('If win rate vs Random is NOT ~90%, there may be a bug in:');
  console.log('  - State encoding');
  console.log('  - Move selection logic');
  console.log('  - Network weights transfer');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
