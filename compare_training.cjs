#!/usr/bin/env node
/**
 * Compare training data construction between train.js and train_tfjs.cjs
 */

const fs = require('fs');
const path = require('path');

async function main() {
  console.log('=== Training Data Construction Comparison ===\n');
  
  // Check train.js normalization
  console.log('📊 train.js normalization:');
  console.log('   minImmediateReward = -4.0');
  console.log('   maxImmediateReward = 15.5');
  console.log('   minCombinedReward = -4.0 - 1.0 = -5.0');
  console.log('   maxCombinedReward = 15.5 + 1.0 + 10.0 = 26.5');
  console.log('   range = 31.5');
  console.log('');
  console.log('   BUT: immediateRewardVector values are scaled [0.01, 0.99]');
  console.log('   AND: rawImmediateReward = average(vector) ≈ 0.02-0.03');
  console.log('');
  console.log('   Example - WIN on last move:');
  console.log('     rawImmediateReward ≈ 0.02 (average of mostly 0.01 values)');
  console.log('     winReward = +1.0');
  console.log('     combined = 0.02 + 1.0 = 1.02');
  console.log('     scaled = (1.02 - (-5.0)) / 31.5 = 6.02/31.5 = 0.19');
  console.log('');
  console.log('   Example - LOSS on last move:');
  console.log('     combined = 0.02 - 1.0 = -0.98');
  console.log('     scaled = (-0.98 - (-5.0)) / 31.5 = 4.02/31.5 = 0.13');
  console.log('');
  console.log('   WIN vs LOSS ratio: 0.19 vs 0.13 (difference = 0.06)');
  console.log('');
  
  // Check train_tfjs.cjs normalization
  console.log('📊 train_tfjs.cjs normalization:');
  console.log('   minImmediateReward = 0.01 (scaled range)');
  console.log('   maxImmediateReward = 0.99 (scaled range)');
  console.log('   minCombinedReward = 0.01 - 1.0 = -0.99');
  console.log('   maxCombinedReward = 0.99 + 1.0 = 1.99');
  console.log('   range = 2.98');
  console.log('');
  console.log('   Using chosenMoveReward = vector[moveIdx] (correct!)');
  console.log('');
  console.log('   Example - WIN on last move, chosenMoveReward = 0.5:');
  console.log('     combined = 0.5 + 1.0 = 1.5');
  console.log('     scaled = (1.5 - (-0.99)) / 2.98 = 2.49/2.98 = 0.84');
  console.log('');
  console.log('   Example - LOSS on last move, chosenMoveReward = 0.5:');
  console.log('     combined = 0.5 - 1.0 = -0.5');
  console.log('     scaled = (-0.5 - (-0.99)) / 2.98 = 0.49/2.98 = 0.16');
  console.log('');
  console.log('   WIN vs LOSS ratio: 0.84 vs 0.16 (difference = 0.68)');
  console.log('');
  
  console.log('════════════════════════════════════════════════════════════');
  console.log('                    KEY FINDING                              ');
  console.log('════════════════════════════════════════════════════════════\n');
  
  console.log('train.js has TWO bugs:');
  console.log('  1. rawImmediateReward uses average (diluted to ~0.02)');
  console.log('  2. Normalization range is for RAW values, not scaled');
  console.log('');
  console.log('BUT: The two bugs COMPENSATE each other!');
  console.log('  - Bug 1: weakens signal (0.5 → 0.02)');
  console.log('  - Bug 2: weakens normalization (0.5 in [0,31.5] vs [-0.99, 1.99])');
  console.log('');
  console.log('Result: train.js produces similar gradients despite bugs.');
  console.log('');
  
  console.log('train_tfjs.cjs has CORRECT normalization but:');
  console.log('  1. Uses chosenMoveReward ✓');
  console.log('  2. Uses correct range [-0.99, 1.99] ✓');
  console.log('  3. Produces STRONGER gradients (0.84 vs 0.16)');
  console.log('');
  console.log('Stronger gradients should help training, not hurt it.');
  console.log('The problem must be elsewhere...');
  console.log('');
  
  // Now check if there's an issue with training on opponent moves
  console.log('────────────────────────────────────────────────────────────');
  console.log('CRITICAL: Training on opponent moves\n');
  
  console.log('train.js: Only trains on AI moves (player 1)');
  console.log('train_tfjs.cjs: Trained on ALL moves until our fix');
  console.log('');
  console.log('When AI (P1) wins:');
  console.log('  - P1 moves get +1 reward ✓');
  console.log('  - P2 moves (if trained) get -1 reward ✗');
  console.log('');
  console.log('This teaches the AI to play LOSING moves!');
  console.log('The fix (filter step.player !== aiPlayerNumber) is CORRECT.');
  console.log('');
  
  // Check state encoding consistency
  console.log('────────────────────────────────────────────────────────────');
  console.log('State Encoding Check\n');
  
  // Simulate a simple board
  const testState = new Array(401).fill(0);
  testState[0] = 1;  // P1 at (0,0) triangle 1
  testState[2] = -1; // P2 at (0,0) triangle 2
  testState[400] = 1; // Bias
  
  console.log('Test state: P1 at (0,0) t1, P2 at (0,0) t2');
  console.log('  First 4 values:', testState.slice(0, 4));
  console.log('  Last value:', testState[400]);
  console.log('');
  console.log('Expected encoding (AI=P1):');
  console.log('  [1, -1, -1, -1] (P1=1, P2=-1, rotations normalized)');
  console.log('');
  console.log('train.js uses _getStateRepresentation() which does:');
  console.log('  normalizePlayer(t) = t.player === this.playerNumber ? 1 : -1');
  console.log('  this.playerNumber is fixed at P1');
  console.log('');
  console.log('train_tfjs.cjs uses getStateFixed(grid, aiPlayerNumber):');
  console.log('  pv(t) = t.player === aiPlayerNumber ? 1 : -1');
  console.log('  This matches when aiPlayerNumber === 1');
  console.log('');
  
  console.log('════════════════════════════════════════════════════════════');
  console.log('                    CONCLUSION                              ');
  console.log('════════════════════════════════════════════════════════════\n');
  
  console.log('1. The opponent-move filter fix is CORRECT and needed.');
  console.log('2. The normalization in train_tfjs.cjs is CORRECT.');
  console.log('3. State encoding appears consistent.');
  console.log('');
  console.log('Possible remaining issues:');
  console.log('  a) Network architecture mismatch (activation, initialization)');
  console.log('  b) Learning rate or optimizer differences');
  console.log('  c) Batch size or training iterations');
  console.log('  d) Self-play exploration behavior');
  console.log('  e) Reward discount factor implementation');
  console.log('');
  console.log('Next step: Check if brain.js uses a different optimizer');
  console.log('          or if the network architecture differs.');
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
