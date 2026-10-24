#!/usr/bin/env node
/**
 * SINGLE GAME TRACE - Traces reward values through one complete game
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');

// Import GridManager
async function main() {
  const { GridManager } = await import(path.resolve('./src/core/GridManager.js'));
  
  console.log('═══════════════════════════════════════════════════════════════════════════');
  console.log('            SINGLE GAME REWARD TRACE                                         ');
  console.log('═══════════════════════════════════════════════════════════════════════════\n');

  const grid = new GridManager(10, 10);
  
  // Create simple network
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [401], units: 200, activation: 'sigmoid' }));
  model.add(tf.layers.dense({ units: 400, activation: 'sigmoid' }));
  model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });

  // Trace a short game
  const hist = [];
  const aiPlayer = 1;
  let curP = 1;
  let m = 0;
  
  while (!grid.isGridFull() && m < 20) { // Only 20 moves
    grid.currentPlayer = curP;
    
    // Calculate immediate reward vector
    const immediateVector = calculateImmediateRewardVector(grid, curP);
    
    // Get valid moves
    const validMoves = [];
    for (let y = 0; y < 10; y++) {
      for (let x = 0; x < 10; x++) {
        for (let r = 0; r < 4; r++) {
          if (isValidMove(grid, x, y, r)) {
            validMoves.push({ x, y, rotation: r });
          }
        }
      }
    }
    
    if (!validMoves.length) break;
    
    // Pick random move
    const move = validMoves[Math.floor(Math.random() * validMoves.length)];
    const moveIdx = (move.y * 10 + move.x) * 4 + move.rotation;
    
    console.log(`Move ${m}: Player ${curP} plays (${move.x},${move.y}) r${move.rotation}`);
    console.log(`  Valid moves: ${validMoves.length}`);
    console.log(`  Chosen move index: ${moveIdx}`);
    
    // Show immediate reward values
    const sortedRewards = immediateVector
      .map((v, i) => ({ idx: i, val: v }))
      .filter(x => x.val > 0.015)
      .sort((a, b) => b.val - a.val);
    
    console.log(`  Top 5 immediate rewards:`);
    for (const r of sortedRewards.slice(0, 5)) {
      const x = Math.floor(r.idx / 4) % 10;
      const y = Math.floor(r.idx / 40);
      const rot = r.idx % 4;
      console.log(`    (${x},${y}) r${rot}: ${r.val.toFixed(4)}`);
    }
    
    const chosenImmediateReward = immediateVector[moveIdx];
    console.log(`  Chosen move immediate reward: ${chosenImmediateReward.toFixed(4)}`);
    
    // Store sample
    hist.push({
      moveNum: m,
      player: curP,
      moveIdx,
      chosenImmediateReward,
      immediateVector: [...immediateVector],
      validMoves: validMoves.length
    });
    
    // Make move
    grid.selectedRotation = move.rotation;
    grid.placeTriangle(move.x, move.y);
    grid.claimEnclosedTriangles();
    
    // Next player
    if (grid.extraTurns > 0) {
      grid.extraTurns--;
    } else {
      curP = curP === 1 ? 2 : 1;
    }
    m++;
  }
  
  // Determine winner
  const s1 = grid.getLargestContiguous(1);
  const s2 = grid.getLargestContiguous(2);
  const winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
  
  console.log(`\n═══════════════════════════════════════════════════════════════════════════`);
  console.log(`                    GAME RESULT                                              `);
  console.log(`═══════════════════════════════════════════════════════════════════════════\n`);
  console.log(`Final scores: P1=${s1}, P2=${s2}`);
  console.log(`Winner: ${winner === 0 ? 'DRAW' : `Player ${winner}`}`);
  
  // Show training targets
  console.log(`\n═══════════════════════════════════════════════════════════════════════════`);
  console.log(`                TRAINING TARGETS (for AI = Player ${aiPlayer})                `);
  console.log(`═══════════════════════════════════════════════════════════════════════════\n`);
  
  const finalRewardWeight = 1.0;
  const minC = 0.01 - finalRewardWeight; // -0.99
  const maxC = 0.99 + finalRewardWeight; // 1.99
  const range = maxC - minC; // 2.98
  
  console.log(`Normalization: range=[${minC.toFixed(2)}, ${maxC.toFixed(2)}], span=${range.toFixed(2)}\n`);
  
  for (const step of hist) {
    if (step.player !== aiPlayer) {
      console.log(`Move ${step.moveNum}: Player ${step.player} (opponent) - SKIPPED`);
      continue;
    }
    
    const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
    const combinedReward = step.chosenImmediateReward + rewardForPlayer * finalRewardWeight;
    
    let scaledCombined = (combinedReward - minC) / range;
    scaledCombined = Math.max(0.01, Math.min(0.99, scaledCombined));
    
    console.log(`Move ${step.moveNum}: AI (Player ${step.player})`);
    console.log(`  immediate = ${step.chosenImmediateReward.toFixed(4)}`);
    console.log(`  winReward = ${rewardForPlayer.toFixed(1)}`);
    console.log(`  combined = ${combinedReward.toFixed(4)}`);
    console.log(`  normalized = ${scaledCombined.toFixed(4)}`);
    
    // Verify the final training target
    const finalTarget = scaledCombined;
    console.log(`  ✅ TRAINING TARGET for move ${step.moveIdx}: ${finalTarget.toFixed(4)}`);
    
    if (winner === aiPlayer && finalTarget < 0.5) {
      console.log(`  ❌ BUG: Won but target is LOW (${finalTarget.toFixed(4)})`);
    }
    if (winner !== aiPlayer && winner !== 0 && finalTarget > 0.1) {
      console.log(`  ❌ BUG: Lost but target is HIGH (${finalTarget.toFixed(4)})`);
    }
    console.log('');
  }
}

// Helper functions (copied from train_tfjs.cjs)
function isValidMove(grid, x, y, rot) {
  return grid.isValidMove(x, y, rot);
}

function calculateImmediateRewardVector(grid, player) {
  const { rows, cols } = grid;
  const outputVector = new Array(rows * cols * 4).fill(0.01);
  
  const claimRewardFactor = 0.3;
  const connectedClusterRewardFactor = 0.1;
  
  const maxPossibleClaimBonus = 1.5;
  const maxPossibleClusterSize = rows * cols * 2;
  const maxPossibleClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;
  const minImmediateReward = 0;
  const maxImmediateReward = maxPossibleClaimBonus + maxPossibleClusterBonus;
  const rewardRange = maxImmediateReward - minImmediateReward;
  
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      for (let rotation = 0; rotation < 4; rotation++) {
        const moveIndex = (y * cols + x) * 4 + rotation;
        
        if (!isValidMove(grid, x, y, rotation)) continue;
        
        const tempManager = grid.clone();
        tempManager.currentPlayer = player;
        const simSuccess = tempManager.placeTriangle(x, y, rotation, player);
        
        if (simSuccess) {
          const simClaimed = tempManager.claimEnclosedTriangles(player);
          const hypoClaimReward = simClaimed * claimRewardFactor;
          
          const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);
          const hypoClusterReward = simClusterSize * connectedClusterRewardFactor;
          
          const hypotheticalReward = hypoClaimReward + hypoClusterReward;
          
          let scaledReward = rewardRange !== 0
            ? (hypotheticalReward - minImmediateReward) / rewardRange
            : 0.5;
          
          outputVector[moveIndex] = Math.max(0.01, Math.min(0.99, scaledReward));
        }
      }
    }
  }
  
  return outputVector;
}

main().catch(e => console.error('Error:', e));
