#!/usr/bin/env node
/**
 * Check if trained network produces sensible outputs
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

async function main() {
  // Load the network
  const networkPath = 'dashboard/data/TF-B1_pure_network.json';
  if (!fs.existsSync(networkPath)) {
    console.log('Network file not found');
    return;
  }

  const networkData = JSON.parse(fs.readFileSync(networkPath, 'utf-8'));

  // Build model from saved weights
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    inputShape: [401],
    units: 200,
    activation: 'sigmoid',
    kernelInitializer: 'zeros',
    biasInitializer: 'zeros'
  }));
  
  model.add(tf.layers.dense({
    units: 400,
    activation: 'sigmoid',
    kernelInitializer: 'zeros',
    biasInitializer: 'zeros'
  }));

  // Load weights from brain.js format
  // layers[0] = input layer (no weights)
  // layers[1] = first hidden layer (input → 200)
  // layers[2] = output layer (200 → 400)
  
  if (networkData.layers && networkData.layers.length >= 2) {
    // Layer 1: 401 inputs → 200 outputs
    const l1_weights = networkData.layers[0]?.weights || [];
    const l1_biases = networkData.layers[0]?.biases || [];
    
    if (l1_weights.length > 0) {
      // brain.js: weights[target][source], TF.js: weights[source][target]
      const w1 = [];
      for (let s = 0; s < 401; s++) {
        w1.push([]);
        for (let t = 0; t < 200; t++) {
          w1[s].push(l1_weights[t]?.[s] || 0);
        }
      }
      
      model.layers[0].setWeights([
        tf.tensor2d(w1, [401, 200]),
        tf.tensor1d(l1_biases)
      ]);
    }

    // Layer 2: 200 inputs → 400 outputs
    const l2_weights = networkData.layers[1]?.weights || [];
    const l2_biases = networkData.layers[1]?.biases || [];
    
    if (l2_weights.length > 0) {
      const w2 = [];
      for (let s = 0; s < 200; s++) {
        w2.push([]);
        for (let t = 0; t < 400; t++) {
          w2[s].push(l2_weights[t]?.[s] || 0);
        }
      }
      
      model.layers[1].setWeights([
        tf.tensor2d(w2, [200, 400]),
        tf.tensor1d(l2_biases)
      ]);
    }
  }

  console.log('Testing trained network outputs:\n');

  // Test 1: Empty board
  console.log('TEST 1: Empty board (first move)');
  const emptyState = new Array(401).fill(0);
  emptyState[400] = 1; // Bias
  
  const pred1 = model.predict(tf.tensor2d([emptyState], [1, 401]));
  const out1 = Array.from(pred1.dataSync());
  pred1.dispose();
  
  console.log(`  Output range: ${Math.min(...out1).toFixed(3)} - ${Math.max(...out1).toFixed(3)}`);
  console.log(`  Top 5 moves:`);
  const indexed1 = out1.map((v, i) => ({i, v})).sort((a, b) => b.v - a.v);
  for (const m of indexed1.slice(0, 5)) {
    const x = Math.floor(m.i / 4) % 10;
    const y = Math.floor(m.i / 40);
    const r = m.i % 4;
    console.log(`    (${x},${y}) r${r}: ${m.v.toFixed(3)}`);
  }

  // Test 2: Mid-game position
  console.log('\nTEST 2: Mid-game position');
  const midState = new Array(401).fill(0);
  // Add some pieces (simplified - just marking cells as occupied)
  for (let i = 0; i < 50; i++) {
    midState[i * 4 + Math.floor(Math.random() * 4)] = Math.random() < 0.5 ? 1 : -1;
  }
  midState[400] = 1;
  
  const pred2 = model.predict(tf.tensor2d([midState], [1, 401]));
  const out2 = Array.from(pred2.dataSync());
  pred2.dispose();
  
  console.log(`  Output range: ${Math.min(...out2).toFixed(3)} - ${Math.max(...out2).toFixed(3)}`);
  console.log(`  Top 5 moves:`);
  const indexed2 = out2.map((v, i) => ({i, v})).sort((a, b) => b.v - a.v);
  for (const m of indexed2.slice(0, 5)) {
    const x = Math.floor(m.i / 4) % 10;
    const y = Math.floor(m.i / 40);
    const r = m.i % 4;
    console.log(`    (${x},${y}) r${r}: ${m.v.toFixed(3)}`);
  }

  // Test 3: Check for reasonable distribution
  console.log('\nANALYSIS:');
  const avgVal = out1.reduce((s, v) => s + v, 0) / 400;
  const variance = out1.reduce((s, v) => s + (v - avgVal) ** 2, 0) / 400;
  const stdDev = Math.sqrt(variance);
  
  console.log(`  Mean output: ${avgVal.toFixed(3)}`);
  console.log(`  Std deviation: ${stdDev.toFixed(3)}`);
  
  if (stdDev < 0.05) {
    console.log('  ⚠️  LOW VARIANCE - network may not be trained properly');
    console.log('     All outputs are nearly identical');
  } else if (stdDev > 0.3) {
    console.log('  ✅ Good variance - network is discriminating between moves');
  } else {
    console.log('  ⚠️  Moderate variance - network has some discrimination');
  }
}

main().catch(e => console.error('Error:', e));
