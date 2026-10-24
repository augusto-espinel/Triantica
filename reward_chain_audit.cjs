#!/usr/bin/env node
/**
 * REWARD CHAIN AUDIT — train.js vs train_tfjs.cjs
 */

async function main() {
  console.log('════════════════════════════════════════════════════════════════');
  console.log('           REWARD CHAIN AUDIT — train.js vs train_tfjs.cjs      ');
  console.log('════════════════════════════════════════════════════════════════\n');

  console.log('━━━ BUG #1: MSE GRADIENT DOMINATION ━━━\n');
  console.log('train.js (brain.js) uses cross-entropy loss + online training.');
  console.log('train_tfjs.cjs uses MSE loss + batch training.');
  console.log('');
  console.log('With MSE, for each sample there are 400 output targets:');
  console.log('  399 targets ≈ 0.01 (illegal/unchosen moves)');
  console.log('  1 target   = 0.681 (chosen winning move)');
  console.log('');
  console.log('The 399 "suppress everything" gradients DOMINATE the 1 "boost chosen" gradient.');
  console.log('The network learns "dont play illegal" but barely learns "play THIS".');
  console.log('FIX: Gradient isolation — set unchosen move targets = current prediction.\n');

  console.log('━━━ BUG #2: DISCOUNT FACTOR NOT IN ORIGINAL ━━━\n');
  console.log('train_tfjs.cjs adds: discountFactor = (t + 1) / totalMoves');
  console.log('train.js has NO discount factor — all moves get full win/loss reward.');
  console.log('');
  console.log('This artificially weakens the signal for early-game moves.');
  console.log('FIX: Remove the discount factor entirely.\n');

  console.log('━━━ SUMMARY OF FIXES NEEDED ━━━\n');
  console.log('1. Gradient isolation: unchosen moves → zero gradient');
  console.log('2. Remove discount factor');
  console.log('3. Keep momentum SGD + LR 0.3 (already done)');
}

main();
