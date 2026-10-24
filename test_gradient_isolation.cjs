#!/usr/bin/env node
/**
 * Test if TF.js gradient isolation works correctly
 */

const tf = require('@tensorflow/tfjs-node');

async function testGradientIsolation() {
  console.log('═══════════════════════════════════════════════════════════════════');
  console.log('        TF.js GRADIENT ISOLATION VERIFICATION                      ');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  // Create a simple model
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [10], units: 10, activation: 'sigmoid' }));
  model.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }));

  // Test 1: Standard MSE loss with 1 high target, 9 low targets
  console.log('TEST 1: Standard MSE (1 high target, 9 low)\n');
  
  model.compile({ optimizer: tf.train.sgd(0.5), loss: 'meanSquaredError' });
  
  const xs = tf.tensor2d([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]);
  const ys = tf.tensor2d([[0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]);
  
  const h1 = await model.fit(xs, ys, { epochs: 1, verbose: 0 });
  console.log(`  Loss: ${h1.history.loss[0].toFixed(6)}`);
  console.log(`  Expected MSE: ((0.5-0.8)² + 9×(0.5-0.1)²) / 10 = ${((0.09 + 9*0.16)/10).toFixed(6)}`);
  console.log('');

  // Get predictions after 1 epoch
  const pred1 = model.predict(xs);
  const pred1Data = pred1.dataSync();
  console.log(`  Predictions: [${Array.from(pred1Data).map(v => v.toFixed(3)).join(', ')}]`);
  console.log(`  Target was:   [0.800, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100]`);
  
  pred1.dispose();
  xs.dispose();
  ys.dispose();

  // Test 2: Check if custom loss with weights works
  console.log('\nTEST 2: Custom loss with closure over weights\n');
  
  const model2 = tf.sequential();
  model2.add(tf.layers.dense({ inputShape: [10], units: 10, activation: 'sigmoid' }));
  model2.add(tf.layers.dense({ units: 10, activation: 'sigmoid' }));

  // Weight mask: only first output matters
  const weightMask = tf.tensor2d([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [1, 10]);
  
  const customLoss = (yTrue, yPred) => {
    // Weighted MSE: only first output counts
    const diff = yPred.sub(yTrue).square();
    const weighted = diff.mul(weightMask);
    return weighted.mean();
  };
  
  model2.compile({ optimizer: tf.train.sgd(0.5), loss: customLoss });
  
  const xs2 = tf.tensor2d([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]);
  const ys2 = tf.tensor2d([[0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]);
  
  const h2 = await model2.fit(xs2, ys2, { epochs: 1, verbose: 0 });
  console.log(`  Loss: ${h2.history.loss[0].toFixed(6)}`);
  console.log(`  Expected: (0.5-0.8)² / 10 = 0.009 (only first output, averaged over 10)`);
  console.log('  OR: (0.5-0.8)² = 0.09 (if not averaged over all outputs)');
  console.log('');

  const pred2 = model2.predict(xs2);
  const pred2Data = pred2.dataSync();
  console.log(`  Predictions: [${Array.from(pred2Data).map(v => v.toFixed(3)).join(', ')}]`);
  
  pred2.dispose();
  xs2.dispose();
  ys2.dispose();
  weightMask.dispose();

  console.log('\n═══════════════════════════════════════════════════════════════════');
  console.log('                         VERIFICATION                                 ');
  console.log('═══════════════════════════════════════════════════════════════════\n');

  // In Test 2, if gradient isolation works, first output should move toward 0.8
  // while others stay near 0.5 (not pushed to 0.1)
  console.log('If gradient isolation works:');
  console.log('  - First output should move TOWARD 0.8 (the target)');
  console.log('  - Other outputs should stay near 0.5 (not pushed to 0.1)');
  console.log('');
  console.log('If standard MSE:');
  console.log('  - All outputs move toward their targets');
  console.log('  - First moves to 0.8, others move to 0.1');
}

testGradientIsolation().catch(e => console.error('Test failed:', e));
