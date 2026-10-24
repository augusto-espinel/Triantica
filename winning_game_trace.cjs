#!/usr/bin/env node
/**
 * Check if winning games produce correct training targets
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');

async function main() {
  const { GridManager } = await import(path.resolve('./src/core/GridManager.js'));
  
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('        WINNING GAME REWARD TRACE                              ');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const grid = new GridManager(10, 10);
  
  // Simulate a scenario where P1 wins
  // P1: score 10, P2: score 5
  // To guarantee this, we'll play specific moves
  
  const hist = [];
  const aiPlayer = 1;
  let m = 0;
  
  // Play a complete game where P1 dominates
  let curP = 1;
  while (!grid.isGridFull() && m < 200) {
    grid.currentPlayer = curP;
    
    const immediateVector = calculateImmediateRewardVector(grid, curP);
    const validMoves = getValidMoves(grid);
    
    if (!validMoves.length) break;
    
    // P1 tries to maximize, P2 is random
    let move;
    if (curP === 1) {
      // Find best immediate reward move for P1
      const sorted = validMoves.map(mv => ({
        ...mv,
        reward: immediateVector[(mv.y * 10 + mv.x) * 4 + mv.rotation]
      })).sort((a, b) => b.reward - a.reward);
      move = sorted[0];
    } else {
      move = validMoves[Math.floor(Math.random() * validMoves.length)];
    }
    
    const moveIdx = (move.y * 10 + move.x) * 4 + move.rotation;
    
    hist.push({
      moveNum: m,
      player: curP,
      moveIdx,
      immediateReward: immediateVector[moveIdx],
      validMoves: validMoves.length
    });
    
    grid.selectedRotation = move.rotation;
    grid.placeTriangle(move.x, move.y);
    grid.claimEnclosedTriangles(curP);
    
    if (grid.extraTurns > 0) grid.extraTurns--;
    else curP = curP === 1 ? 2 : 1;
    
    m++;
  }
  
  const s1 = grid.getLargestContiguous(1);
  const s2 = grid.getLargestContiguous(2);
  const winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
  
  console.log(`Game finished: P1=${s1}, P2=${s2}, Winner=${winner}`);
  
  if (winner !== 1) {
    console.log('\n⚠️  P1 did not win. Re-run to get a winning game.');
  }
  
  // Check training targets for AI moves
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('        TRAINING TARGETS FOR WINNING MOVES                      ');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  
  const minC = 0.01 - 1.0;
  const maxC = 0.99 + 1.0;
  const range = maxC - minC;
  
  let correctTargets = 0;
  let totalTargets = 0;
  
  for (const step of hist) {
    if (step.player !== aiPlayer) continue;
    
    totalTargets++;
    const rewardForPlayer = winner === step.player ? 1.0 : -1.0;
    const combinedReward = step.immediateReward + rewardForPlayer * 1.0;
    let scaledTarget = (combinedReward - minC) / range;
    scaledTarget = Math.max(0.01, Math.min(0.99, scaledTarget));
    
    const isCorrect = (winner === aiPlayer && scaledTarget > 0.5) || 
                       (winner !== aiPlayer && winner !== 0 && scaledTarget < 0.2);
    
    if (isCorrect) correctTargets++;
    
    console.log(`Move ${step.moveNum}: immediate=${step.immediateReward.toFixed(4)}, ` +
                `winReward=${rewardForPlayer.toFixed(1)}, ` +
                `target=${scaledTarget.toFixed(4)} ` +
                `${isCorrect ? '✅' : '❌'}`);
  }
  
  console.log(`\n${correctTargets}/${totalTargets} targets correct`);
  
  if (correctTargets < totalTargets) {
    console.log('\n❌ SIGNAL BUG DETECTED!');
    console.log('   Some training targets are pointing in the wrong direction!');
  } else {
    console.log('\n✅ All signals point in correct direction');
  }
}

function getValidMoves(grid) {
  const valid = [];
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      for (let r = 0; r < 4; r++) {
        if (grid.isValidMove(x, y, r)) {
          valid.push({ x, y, rotation: r });
        }
      }
    }
  }
  return valid;
}

function calculateImmediateRewardVector(grid, player) {
  const outputVector = new Array(400).fill(0.01);
  
  const claimRewardFactor = 0.3;
  const connectedClusterRewardFactor = 0.1;
  const maxPossibleClaimBonus = 1.5;
  const maxPossibleClusterSize = 200;
  const maxPossibleClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;
  const minImmediateReward = 0;
  const maxImmediateReward = maxPossibleClaimBonus + maxPossibleClusterBonus;
  const rewardRange = maxImmediateReward - minImmediateReward;
  
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      for (let rotation = 0; rotation < 4; rotation++) {
        const moveIndex = (y * 10 + x) * 4 + rotation;
        if (!grid.isValidMove(x, y, rotation)) continue;
        
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
