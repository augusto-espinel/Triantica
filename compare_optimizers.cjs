#!/usr/bin/env node
/**
 * Direct comparison: brain.js ArthurDeviationWeights vs TF.js momentum
 * 
 * Tests the SAME network architecture on the SAME data with both optimizers
 * to measure the weight magnitude and convergence difference.
 */
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

// Create identical networks
function createNetwork() {
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [401],
    units: 200,
    activation: 'sigmoid',
    useBias: true
  }));
  model.add(tf.layers.dense({
    units: 400,
    activation: 'sigmoid',
    useBias: true
  }));
  model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
  return model;
}

// ArthurDeviationWeights: change = lr * grad + momentum * lastChange; weight += change
class ArthurOptimizer {
  constructor(model, lr = 0.1, momentum = 0.1) {
    this.lr = lr;
    this.momentum = momentum;
    this.changes = model.getWeights().map(w => tf.zerosLike(w));
  }
  
  step(model) {
    const weights = model.getWeights();
    const newWeights = [];
    const newChanges = [];
    
    // Compute gradients manually
    const { grads } = tf.variableGrads(() => {
      // Just return a dummy - we need external gradients
      return tf.scalar(0);
    });
    
    // Actually, we need to use model.optimizer.computeGradients
    // Let's use a different approach: train one step and compare
    
    for (let l = 0; l < weights.length; l++) {
      // For now just track weights
      newWeights.push(weights[l].clone());
      newChanges.push(this.changes[l].clone());
    }
    
    return { newWeights, newChanges };
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('  Arthur Deviation Weights vs TF.js Momentum — Weight Comparison');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  // Generate simple synthetic training data
  const N = 1000;
  const inputs = [];
  const targets = [];
  for (let i = 0; i < N; i++) {
    const inp = new Array(401).fill(0).map(() => Math.random() * 0.1);
    inp[400] = 1;
    const out = new Array(400).fill(0.01);
    const chosenIdx = Math.floor(Math.random() * 400);
    out[chosenIdx] = 0.8; // One "winning" move
    inputs.push(inp);
    targets.push(out);
  }
  
  // === Method 1: TF.js momentum optimizer (batch training) ===
  console.log('--- Method 1: TF.js momentum optimizer (batch=256, lr=0.1, mom=0.1) ---');
  const model1 = createNetwork();
  model1.compile({
    optimizer: tf.train.momentum(0.1, 0.1),
    loss: 'meanSquaredError'
  });
  
  const xs = tf.tensor2d(inputs, [N, 401]);
  const ys = tf.tensor2d(targets, [N, 400]);
  
  const before1 = model1.getWeights().map(w => w.dataSync());
  const w1Stats_before = before1.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs)),
    min: Math.min(...Array.from(w))
  }));
  
  await model1.fit(xs, ys, { epochs: 30, batchSize: 256, shuffle: true, verbose: 0 });
  
  const after1 = model1.getWeights().map(w => w.dataSync());
  const w1Stats_after = after1.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs)),
    min: Math.min(...Array.from(w))
  }));
  
  console.log('  Weight changes (batch=256):');
  for (let l = 0; l < w1Stats_before.length; l++) {
    const delta = w1Stats_after[l].mean - w1Stats_before[l].mean;
    console.log(`    Layer ${l}: |w|_avg ${w1Stats_before[l].mean.toFixed(6)} → ${w1Stats_after[l].mean.toFixed(6)} (Δ=${delta.toFixed(6)}), max=${w1Stats_after[l].max.toFixed(6)}`);
  }
  
  // === Method 2: TF.js momentum optimizer (online, batch=1) ===
  console.log('\n--- Method 2: TF.js momentum optimizer (batch=1, lr=0.1, mom=0.1) ---');
  const model2 = createNetwork();
  model2.compile({
    optimizer: tf.train.momentum(0.1, 0.1),
    loss: 'meanSquaredError'
  });
  
  // Copy same initial weights
  model2.setWeights(model1.getWeights().map(w => w.clone()));
  // Re-compile after setWeights
  model2.compile({
    optimizer: tf.train.momentum(0.1, 0.1),
    loss: 'meanSquaredError'
  });
  
  const before2 = model2.getWeights().map(w => w.dataSync());
  const w2Stats_before = before2.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs))
  }));
  
  await model2.fit(xs, ys, { epochs: 30, batchSize: 1, shuffle: true, verbose: 0 });
  
  const after2 = model2.getWeights().map(w => w.dataSync());
  const w2Stats_after = after2.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs))
  }));
  
  console.log('  Weight changes (batch=1):');
  for (let l = 0; l < w2Stats_before.length; l++) {
    const delta = w2Stats_after[l].mean - w2Stats_before[l].mean;
    console.log(`    Layer ${l}: |w|_avg ${w2Stats_before[l].mean.toFixed(6)} → ${w2Stats_after[l].mean.toFixed(6)} (Δ=${delta.toFixed(6)}), max=${w2Stats_after[l].max.toFixed(6)}`);
  }
  
  // === Method 3: Manual Arthur-style online training ===
  console.log('\n--- Method 3: Arthur Deviation Weights (manual online, lr=0.1, mom=0.1) ---');
  const model3 = createNetwork();
  model3.compile({
    optimizer: tf.train.sgd(0), // dummy
    loss: 'meanSquaredError'
  });
  
  // Copy same initial weights
  model3.setWeights(model1.getWeights().map(w => w.clone()));
  
  const before3 = model3.getWeights().map(w => w.dataSync());
  const w3Stats_before = before3.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs))
  }));
  
  // Arthur-style: train one sample at a time with per-weight momentum
  let changes = model3.getWeights().map(w => tf.zerosLike(w).arraySync());
  const lr = 0.1;
  const mom = 0.1;
  
  for (let iter = 0; iter < 30; iter++) {
    // Shuffle
    const indices = Array.from({length: N}, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    // Process each sample
    for (const idx of indices) {
      const inputTensor = tf.tensor2d([inputs[idx]], [1, 401]);
      const targetTensor = tf.tensor2d([targets[idx]], [1, 400]);
      
      // Compute gradients
      const f = () => {
        const pred = model3.predict(inputTensor);
        return tf.losses.meanSquaredError(targetTensor, pred);
      };
      
      const { grads } = tf.variableGrads(f);
      
      // Apply Arthur update
      const weights = model3.getWeights();
      const newWeights = [];
      
      const varNames = Object.keys(grads);
      let layerIdx = 0;
      for (const name of varNames) {
        const g = grads[name];
        const w = weights[layerIdx];
        const gArr = g.arraySync();
        const wArr = w.arraySync();
        
        // Flatten, update, reshape
        const shape = w.shape;
        const wFlat = Array.isArray(wArr[0]) ? wArr.flat(Infinity) : wArr;
        const gFlat = Array.isArray(gArr[0]) ? gArr.flat(Infinity) : gArr;
        
        // Ensure changes array exists
        if (!changes[layerIdx]) {
          changes[layerIdx] = new Array(wFlat.length).fill(0);
        }
        if (changes[layerIdx].length !== wFlat.length) {
          changes[layerIdx] = new Array(wFlat.length).fill(0);
        }
        
        const newW = [];
        const newC = [];
        for (let i = 0; i < wFlat.length; i++) {
          const change = lr * gFlat[i] + mom * changes[layerIdx][i];
          newC.push(change);
          newW.push(wFlat[i] + change);
        }
        
        changes[layerIdx] = newC;
        
        if (shape.length === 2) {
          const reshaped = [];
          for (let r = 0; r < shape[0]; r++) {
            reshaped.push(newW.slice(r * shape[1], (r + 1) * shape[1]));
          }
          newWeights.push(tf.tensor2d(reshaped, shape));
        } else {
          newWeights.push(tf.tensor1d(newW));
        }
        
        g.dispose();
        w.dispose();
        layerIdx++;
      }
      
      model3.setWeights(newWeights);
      
      inputTensor.dispose();
      targetTensor.dispose();
      
      // Dispose grads
      Object.values(grads).forEach(g => g.dispose());
    }
    
    // Progress
    if ((iter + 1) % 10 === 0) process.stdout.write(`  iter ${iter + 1}/30\n`);
  }
  
  const after3 = model3.getWeights().map(w => w.dataSync());
  const w3Stats_after = after3.map(w => ({
    mean: Array.from(w).reduce((a,b) => a+Math.abs(b), 0) / w.length,
    max: Math.max(...Array.from(w).map(Math.abs))
  }));
  
  console.log('  Weight changes (Arthur online):');
  for (let l = 0; l < w3Stats_before.length; l++) {
    const delta = w3Stats_after[l].mean - w3Stats_before[l].mean;
    console.log(`    Layer ${l}: |w|_avg ${w3Stats_before[l].mean.toFixed(6)} → ${w3Stats_after[l].mean.toFixed(6)} (Δ=${delta.toFixed(6)}), max=${w3Stats_after[l].max.toFixed(6)}`);
  }
  
  // === Summary ===
  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('  SUMMARY: Final weight magnitudes after 30 iterations');
  console.log('═══════════════════════════════════════════════════════════════\n');
  
  console.log('  Layer 0 (weights 401→200):');
  console.log(`    Batch=256:  |w|_avg = ${w1Stats_after[0].mean.toFixed(6)}, max = ${w1Stats_after[0].max.toFixed(6)}`);
  console.log(`    Batch=1:    |w|_avg = ${w2Stats_after[0].mean.toFixed(6)}, max = ${w2Stats_after[0].max.toFixed(6)}`);
  console.log(`    Arthur:     |w|_avg = ${w3Stats_after[0].mean.toFixed(6)}, max = ${w3Stats_after[0].max.toFixed(6)}`);
  
  console.log('\n  Layer 1 (bias 200):');
  console.log(`    Batch=256:  |w|_avg = ${w1Stats_after[1].mean.toFixed(6)}, max = ${w1Stats_after[1].max.toFixed(6)}`);
  console.log(`    Batch=1:    |w|_avg = ${w2Stats_after[1].mean.toFixed(6)}, max = ${w2Stats_after[1].max.toFixed(6)}`);
  console.log(`    Arthur:     |w|_avg = ${w3Stats_after[1].mean.toFixed(6)}, max = ${w3Stats_after[1].max.toFixed(6)}`);
  
  console.log('\n  Layer 2 (weights 200→400):');
  console.log(`    Batch=256:  |w|_avg = ${w1Stats_after[2].mean.toFixed(6)}, max = ${w1Stats_after[2].max.toFixed(6)}`);
  console.log(`    Batch=1:    |w|_avg = ${w2Stats_after[2].mean.toFixed(6)}, max = ${w2Stats_after[2].max.toFixed(6)}`);
  console.log(`    Arthur:     |w|_avg = ${w3Stats_after[2].mean.toFixed(6)}, max = ${w3Stats_after[2].max.toFixed(6)}`);
  
  console.log('\n  Layer 3 (bias 400):');
  console.log(`    Batch=256:  |w|_avg = ${w1Stats_after[3].mean.toFixed(6)}, max = ${w1Stats_after[3].max.toFixed(6)}`);
  console.log(`    Batch=1:    |w|_avg = ${w2Stats_after[3].mean.toFixed(6)}, max = ${w2Stats_after[3].max.toFixed(6)}`);
  console.log(`    Arthur:     |w|_avg = ${w3Stats_after[3].mean.toFixed(6)}, max = ${w3Stats_after[3].max.toFixed(6)}`);
  
  xs.dispose();
  ys.dispose();
}

main().catch(e => console.error(e));
