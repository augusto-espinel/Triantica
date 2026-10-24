#!/usr/bin/env node
/**
 * REWARD SIGNAL FORENSIC AUDIT
 * Traces the exact reward values through the training pipeline.
 */

console.log('═══════════════════════════════════════════════════════════════');
console.log('           REWARD SIGNAL FORENSIC AUDIT                        ');
console.log('═══════════════════════════════════════════════════════════════\n');

// SCENARIO: AI wins with minimal immediate reward
const winner = 1; // AI wins
const chosenIdx = 42;
const aiPlayer = 1;

// Simulate immediate reward vector (from AIPlayer.calculateImmediateRewardVector)
const vector = new Array(400).fill(0.01); // Illegal moves
for (let i = 0; i < 50; i++) vector[i * 8] = 0.02 + Math.random() * 0.03;
vector[chosenIdx] = 0.02; // Chosen move (minimal)

// Step 1: What gets stored?
const avgImmediate = vector.reduce((s,v) => s+v, 0) / 400;
const chosenReward = vector[chosenIdx];

console.log('STEP 1: DATA COLLECTION');
console.log('train.js stores: avgImmediateReward =', avgImmediate.toFixed(4));
console.log('train_tfjs stores: chosenMoveReward =', chosenReward.toFixed(4));
console.log('');

// Step 2: Combined reward
const winReward = winner === aiPlayer ? 1.0 : -1.0;
const combined_train = avgImmediate + winReward;
const combined_tf = chosenReward + winReward;

console.log('STEP 2: COMBINED REWARD');
console.log('train.js: combined =', avgImmediate.toFixed(4), '+', winReward, '=', combined_train.toFixed(4));
console.log('train_tfjs: combined =', chosenReward.toFixed(4), '+', winReward, '=', combined_tf.toFixed(4));
console.log('');

// Step 3: Normalization
// train_tfjs uses [0.01, 0.99] scaled range
const minC = 0.01 - 1.0; // -0.99
const maxC = 0.99 + 1.0; // 1.99
const range = maxC - minC; // 2.98

const scaled_train = Math.max(0.01, Math.min(0.99, (combined_train - minC) / range));
const scaled_tf = Math.max(0.01, Math.min(0.99, (combined_tf - minC) / range));

console.log('STEP 3: NORMALIZATION');
console.log('Range: [', minC.toFixed(2), ',', maxC.toFixed(2), '] span =', range.toFixed(2));
console.log('train.js scaled:', scaled_train.toFixed(4));
console.log('train_tfjs scaled:', scaled_tf.toFixed(4));
console.log('');

// Step 4: Training target
console.log('STEP 4: TRAINING TARGET');
console.log('For chosen move idx', chosenIdx, ':');
console.log('  train.js target =', scaled_train.toFixed(4));
console.log('  train_tfjs target =', scaled_tf.toFixed(4));
console.log('');

// Step 5: What about non-chosen moves?
console.log('STEP 5: NON-CHOSEN MOVE TARGETS');
console.log('Illegal move (idx 0): target =', vector[0].toFixed(4));
console.log('Valid unchosen (idx 8): target =', vector[8].toFixed(4));
console.log('');

// Critical analysis
console.log('═══════════════════════════════════════════════════════════════');
console.log('                    SIGNAL ANALYSIS                             ');
console.log('═══════════════════════════════════════════════════════════════\n');

const illegal_target = 0.01;
const valid_unchosen_target = 0.03;
const chosen_target = scaled_tf;

const pred = 0.5; // Network prediction (sigmoid midpoint)
const sigmoid_deriv = pred * (1 - pred); // 0.25

const illegal_error = pred - illegal_target;
const valid_error = pred - valid_unchosen_target;
const chosen_error = pred - chosen_target;

const illegal_grad = illegal_error * sigmoid_deriv;
const valid_grad = valid_error * sigmoid_deriv;
const chosen_grad = chosen_error * sigmoid_deriv;

console.log('Gradient magnitudes (pred=0.5):');
console.log('  Chosen move:   ', chosen_grad.toFixed(4));
console.log('  Illegal move:  ', illegal_grad.toFixed(4));
console.log('  Valid unchosen:', valid_grad.toFixed(4));
console.log('');

const num_illegal = 350;
const num_valid_unchosen = 49;
const num_chosen = 1;

const total_grad_sq = num_illegal * illegal_grad**2 
                    + num_valid_unchosen * valid_grad**2
                    + num_chosen * chosen_grad**2;

const total_grad = Math.sqrt(total_grad_sq);
const chosen_pct = Math.abs(chosen_grad) / total_grad * 100;

console.log('Total gradient contribution:');
console.log('  Illegal (350×):     ', (num_illegal * Math.abs(illegal_grad)).toFixed(2));
console.log('  Valid unchosen (49×):', (num_valid_unchosen * Math.abs(valid_grad)).toFixed(2));
console.log('  Chosen (1×):        ', Math.abs(chosen_grad).toFixed(4));
console.log('');
console.log('  Total magnitude:', total_grad.toFixed(4));
console.log('  Chosen move fraction:', chosen_pct.toFixed(1) + '%');
console.log('');

if (chosen_pct < 5) {
  console.log('❌ CRITICAL: Chosen move contributes <5% of gradient!');
  console.log('   The suppress-everything signal dominates learning.');
} else if (chosen_pct < 20) {
  console.log('⚠️  WARNING: Chosen move contributes <20% of gradient.');
  console.log('   Learning signal is weak but may work with more iterations.');
} else {
  console.log('✅ OK: Chosen move contributes significant gradient.');
}

console.log('\n═══════════════════════════════════════════════════════════════');
console.log('                    KEY FINDING                                  ');
console.log('═══════════════════════════════════════════════════════════════\n');

console.log('The non-chosen VALID moves have targets around 0.02-0.05,');
console.log('which creates a gradient pushing them TOWARD those values.');
console.log('');
console.log('This teaches the network: "valid moves should output 0.02-0.05"');
console.log('But we want: "winning moves should output high, losing moves low"');
console.log('');
console.log('The immediate reward signal is interfering with win probability learning!');
