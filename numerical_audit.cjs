#!/usr/bin/env node
/**
 * NUMERICAL REWARD CHAIN AUDIT
 * Simulate exactly what train.js (brain.js) vs train_tfjs.cjs (TF.js) computes
 * for a single training sample, step by step.
 */

// Simulate a typical move in the middle of a game
// Board has ~60% cells occupied, ~40 valid moves

// === BRAIN.JS PATH (train.js) ===
console.log('═══════════════════════════════════════════════════════════════');
console.log('         BRAIN.JS (train.js) — STEP BY STEP                ');
console.log('═══════════════════════════════════════════════════════════════\n');

// 1. Immediate reward vector (from AIPlayer.calculateImmediateRewardVector)
// Typical values: ~350 illegal (0.01), ~50 valid moves with small rewards
const rewardRange = 21.5; // maxPossibleClaim(1.5) + maxPossibleCluster(20*0.1)
const numIllegal = 350;
const numValid = 50;

// Typical valid move rewards (raw, before scaling)
const validRewards = [];
for (let i = 0; i < numValid; i++) {
  const claim = Math.random() < 0.3 ? 1 : 0; // 30% chance of claiming 1 triangle
  const cluster = 1 + Math.floor(Math.random() * 5); // cluster size 1-5
  validRewards.push(claim * 0.3 + cluster * 0.1);
}

// Scale to [0.01, 0.99]
const scaledValidRewards = validRewards.map(r => Math.max(0.01, Math.min(0.99, r / rewardRange)));

// Build full vector
const immediateVector = new Array(400).fill(0.01);
let validIdx = 0;
for (let i = 0; i < 400; i++) {
  if (Math.random() < numValid / 400 && validIdx < numValid) {
    immediateVector[i] = scaledValidRewards[validIdx++];
  }
}

// 2. rawImmediateReward = average of entire vector
const rawImmediateReward = immediateVector.reduce((s, v) => s + v, 0) / 400;
console.log(`rawImmediateReward (average of 400 values) = ${rawImmediateReward.toFixed(6)}`);

// 3. Choose a move (say index 42, which is valid)
const chosenMoveIdx = 42;
const chosenMoveValue = immediateVector[chosenMoveIdx];
console.log(`chosenMoveValue = immediateVector[42] = ${chosenMoveValue.toFixed(4)}`);

// 4. Win/loss reward
const winner = 1; // AI won
const finalRewardWeight = 1.0;
const winReward = winner === 1 ? 1.0 : -1.0;
console.log(`winReward = ${winReward}`);

// 5. Combined reward (NO discount in train.js)
const combinedReward_brain = rawImmediateReward + winReward * finalRewardWeight;
console.log(`combinedReward = ${rawImmediateReward.toFixed(6)} + ${winReward} * ${finalRewardWeight} = ${combinedReward_brain.toFixed(6)}`);

// 6. Normalization (BUGGY range from train.js)
const minImmediate = -4.0; // train.js uses this for range calculation
const maxImmediate = 15.5;
const minCombined_brain = minImmediate - finalRewardWeight; // -5.0
const maxCombined_brain = maxImmediate + finalRewardWeight + 0; // 16.5 (no cluster)
const range_brain = maxCombined_brain - minCombined_brain; // 21.5

const scaled_brain = Math.max(0.01, Math.min(0.99,
  (combinedReward_brain - minCombined_brain) / range_brain
));
console.log(`Normalization: [${minCombined_brain}, ${maxCombined_brain}], range=${range_brain}`);
console.log(`scaledCombined = (${combinedReward_brain.toFixed(6)} - (${minCombined_brain})) / ${range_brain} = ${scaled_brain.toFixed(6)}`);

// 7. Build output vector for training
// Without gradient isolation: only override chosen move, rest stays as immediate reward
const outputVector_brain = [...immediateVector];
outputVector_brain[chosenMoveIdx] = scaled_brain;

console.log(`\nOutput vector stats:`);
console.log(`  chosen move target = ${scaled_brain.toFixed(6)}`);
console.log(`  valid move targets = ${scaledValidRewards.slice(0, 5).map(v => v.toFixed(4))}...`);
console.log(`  illegal move targets = 0.01`);
console.log(`  non-chosen targets: ${numValid - 1} valid (${scaledValidRewards.filter((_, i) => i !== 0).slice(0, 3).map(v => v.toFixed(4))}...) + ${numIllegal} illegal (0.01)`);

// 8. brain.js training: online, one sample at a time
// Loss = MSE per sample = (1/400) * sum((pred_i - target_i)^2)
// Gradient for output i = (pred_i - target_i) * sigmoid_derivative
// For sigmoid: gradient = (pred_i - target_i) * pred_i * (1 - pred_i)
console.log(`\nbrain.js gradient analysis (assuming pred ≈ 0.5 everywhere):`);
let totalGradSquared = 0;
let chosenGrad = 0;
for (let i = 0; i < 400; i++) {
  const pred = 0.5;
  const target = outputVector_brain[i];
  const grad = (pred - target) * pred * (1 - pred); // sigmoid derivative
  totalGradSquared += grad * grad;
  if (i === chosenMoveIdx) chosenGrad = grad;
}
console.log(`  Gradient for chosen move: ${chosenGrad.toFixed(6)}`);
console.log(`  Total gradient magnitude: ${Math.sqrt(totalGradSquared).toFixed(6)}`);
console.log(`  Chosen move fraction: ${(Math.abs(chosenGrad) / Math.sqrt(totalGradSquared) * 100).toFixed(1)}%`);

console.log('\n');

// === TF.JS PATH (train_tfjs.cjs) ===
console.log('═══════════════════════════════════════════════════════════════');
console.log('         TF.js (train_tfjs.cjs) — STEP BY STEP               ');
console.log('═══════════════════════════════════════════════════════════════\n');

// Same immediate reward vector
const immediateVector_tf = [...immediateVector];

// 2. chosenMoveReward = vector[chosenMoveIdx]
const chosenMoveReward_tf = immediateVector_tf[chosenMoveIdx];
console.log(`chosenMoveReward = ${chosenMoveReward_tf.toFixed(6)}`);

// 3. Combined reward (NO discount factor — we removed it)
const combinedReward_tf = chosenMoveReward_tf + winReward * finalRewardWeight;
console.log(`combinedReward = ${chosenMoveReward_tf.toFixed(6)} + ${winReward} * ${finalRewardWeight} = ${combinedReward_tf.toFixed(6)}`);

// 4. Normalization (CORRECT range for scaled values)
const minCombined_tf = 0.01 - finalRewardWeight; // -0.99
const maxCombined_tf = 0.99 + finalRewardWeight; // 1.99
const range_tf = maxCombined_tf - minCombined_tf; // 2.98

const scaled_tf = Math.max(0.01, Math.min(0.99,
  (combinedReward_tf - minCombined_tf) / range_tf
));
console.log(`Normalization: [${minCombined_tf}, ${maxCombined_tf}], range=${range_tf}`);
console.log(`scaledCombined = (${combinedReward_tf.toFixed(6)} - (${minCombined_tf})) / ${range_tf} = ${scaled_tf.toFixed(6)}`);

// 5. Build output vector
const outputVector_tf = [...immediateVector_tf];
outputVector_tf[chosenMoveIdx] = scaled_tf;

console.log(`\nOutput vector stats:`);
console.log(`  chosen move target = ${scaled_tf.toFixed(6)}`);
console.log(`  valid move targets (same as brain.js) = ${scaledValidRewards.slice(0, 5).map(v => v.toFixed(4))}...`);
console.log(`  illegal move targets = 0.01`);

// 6. TF.js training: batch, MSE
// Loss = (1/N) * (1/400) * sum_samples sum_outputs (pred - target)^2
// Gradient for output i = (pred_i - target_i) / (N * 400)
console.log(`\nTF.js gradient analysis (assuming pred ≈ 0.5 everywhere, N=1 sample):`);
let totalGradSquared_tf = 0;
let chosenGrad_tf = 0;
for (let i = 0; i < 400; i++) {
  const pred = 0.5;
  const target = outputVector_tf[i];
  const grad = (pred - target); // MSE gradient (no sigmoid derivative since we use sigmoid activation, TF handles it)
  totalGradSquared_tf += grad * grad;
  if (i === chosenMoveIdx) chosenGrad_tf = grad;
}
console.log(`  Gradient for chosen move: ${chosenGrad_tf.toFixed(6)}`);
console.log(`  Total gradient magnitude: ${Math.sqrt(totalGradSquared_tf).toFixed(6)}`);
console.log(`  Chosen move fraction: ${(Math.abs(chosenGrad_tf) / Math.sqrt(totalGradSquared_tf) * 100).toFixed(1)}%`);

console.log('\n');

// === COMPARISON ===
console.log('═══════════════════════════════════════════════════════════════');
console.log('                    COMPARISON                              ');
console.log('═══════════════════════════════════════════════════════════════\n');

console.log(`brain.js: chosen target = ${scaled_brain.toFixed(6)}, chosen gradient = ${chosenGrad.toFixed(6)}`);
console.log(`TF.js:    chosen target = ${scaled_tf.toFixed(6)}, chosen gradient = ${chosenGrad_tf.toFixed(6)}`);
console.log('');

// The CRITICAL difference: non-chosen move targets
console.log('Non-chosen move targets:');
console.log('  brain.js: same immediate reward values (0.01 for illegal, 0.02-0.05 for valid unchosen)');
console.log('  TF.js:    same immediate reward values (IDENTICAL to brain.js)');
console.log('');
console.log('→ The output vectors are IDENTICAL in structure!');
console.log('  The only difference is the chosen move target value:');
console.log(`  brain.js: ${scaled_brain.toFixed(6)} (compressed by buggy range)`);
console.log(`  TF.js:    ${scaled_tf.toFixed(6)} (spread by correct range)`);
console.log('');

// The REAL difference is in how the gradients flow through the sigmoid
console.log('KEY: brain.js multiplies gradient by sigmoid_derivative = pred * (1 - pred)');
console.log('     TF.js with sigmoid output + MSE also computes this internally.');
console.log('');
console.log('The difference is brain.js uses ONLINE training (one sample at a time)');
console.log('while TF.js uses BATCH training, averaging gradients over many samples.');
console.log('');
console.log('In batch training with N samples:');
console.log('  - Each sample has 1 chosen move target at ~0.68 and 399 targets at ~0.01');
console.log('  - The network learns from the AVERAGE of all these signals');
console.log('  - The "suppress everything to 0.01" signal dominates');
console.log('  - The "boost chosen move to 0.68" signal is 399x diluted');
console.log('');
console.log('In online training (brain.js):');
console.log('  - Each sample updates weights IMMEDIATELY');
console.log('  - No averaging across samples');
console.log('  - The chosen move signal is applied at full strength');
