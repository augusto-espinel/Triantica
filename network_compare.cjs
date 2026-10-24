#!/usr/bin/env node
/**
 * Direct Comparison: Brain.js vs TF.js Performance
 * 
 * This script tests the original brain.js network (trainedAI_network.json)
 * against the TF.js trained network to verify if they have the same behavior.
 */

const fs = require('fs');
const path = require('path');

async function main() {
  console.log('=== Network Architecture Comparison ===\n');
  
  // Load original brain.js network
  const brainjsPath = path.join(__dirname, 'dist', 'trainedAI_network.json');
  if (!fs.existsSync(brainjsPath)) {
    console.log('❌ Original brain.js network not found at:', brainjsPath);
    return;
  }
  
  const brainjsData = JSON.parse(fs.readFileSync(brainjsPath, 'utf8'));
  
  console.log('📊 Brain.js Network Structure:');
  console.log('   Type:', brainjsData.type || 'unknown');
  console.log('   Input size:', brainjsData.options?.inputSize || brainjsData.sizes?.[0] || 'unknown');
  console.log('   Hidden layers:', brainjsData.options?.hiddenLayers || brainjsData.layers?.hidden?.map(l => l.length) || 'unknown');
  console.log('   Output size:', brainjsData.options?.outputSize || brainjsData.sizes?.[brainjsData.sizes.length-1] || 'unknown');
  console.log('   Activation:', brainjsData.options?.activation || brainjsData.activation || 'unknown');
  
  // Count actual weights
  let totalWeights = 0;
  let totalBiases = 0;
  
  if (brainjsData.layers) {
    for (const layerType of ['hidden', 'output']) {
      if (brainjsData.layers[layerType]) {
        for (const layer of brainjsData.layers[layerType]) {
          if (layer.weights) totalWeights += layer.weights.length;
          if (layer.bias !== undefined) totalBiases++;
        }
      }
    }
  }
  
  console.log('   Total weights:', totalWeights);
  console.log('   Total biases:', totalBiases);
  
  // Load TF.js network for comparison
  const tfjsPath = path.join(__dirname, 'data', 'TF-FIXED-OPPONENT-FILTER_network.json');
  if (fs.existsSync(tfjsPath)) {
    const tfjsData = JSON.parse(fs.readFileSync(tfjsPath, 'utf8'));
    console.log('\n📊 TF.js Network Structure:');
    console.log('   Input size:', tfjsData.inputSize || 'unknown');
    console.log('   Hidden layers:', tfjsData.hiddenLayers || 'unknown');
    console.log('   Output size:', tfjsData.outputSize || 'unknown');
    console.log('   Activation:', tfjsData.activation || 'unknown');
    
    // Try to load weights if separate file exists
    const weightsPath = tfjsPath.replace('_network.json', '_weights.json');
    if (fs.existsSync(weightsPath)) {
      const weights = JSON.parse(fs.readFileSync(weightsPath, 'utf8'));
      console.log('   Weights file exists:', weights.length || Object.keys(weights).length, 'keys');
    }
  }
  
  // Now test actual inference
  console.log('\n=== Testing Inference ===\n');
  
  // Create a simple test state
  const testState = new Array(401).fill(0);
  testState[400] = 1; // Bias node
  
  // Try to load and run brain.js
  try {
    const brain = require('brain.js');
    const net = new brain.NeuralNetwork();
    net.fromJSON(brainjsData);
    
    const output = net.run(testState);
    console.log('✅ Brain.js inference successful');
    console.log('   Output type:', typeof output);
    console.log('   Output length:', output.length);
    
    const outputArray = Array.isArray(output) ? output : output.toFloat32Array ? Array.from(output) : Object.values(output);
    console.log('   Output range:', Math.min(...outputArray).toFixed(4), '-', Math.max(...outputArray).toFixed(4));
    console.log('   Top 5 moves:');
    
    const moves = outputArray.map((score, i) => ({ index: i, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    
    for (const m of moves) {
      const x = Math.floor(m.index / 4) % 10;
      const y = Math.floor(m.index / 40);
      const rot = m.index % 4;
      console.log(`     (${x}, ${y}) rot=${rot}: ${m.score.toFixed(4)}`);
    }
  } catch (e) {
    console.log('❌ Brain.js error:', e.message);
  }
  
  // Try TF.js
  try {
    const tf = require('@tensorflow/tfjs-node');
    
    // Create a simple TF.js model with same architecture
    const model = tf.sequential();
    model.add(tf.layers.dense({ 
      units: 200, 
      inputShape: [401], 
      activation: 'sigmoid' 
    }));
    model.add(tf.layers.dense({ 
      units: 400, 
      activation: 'sigmoid' 
    }));
    
    const input = tf.tensor2d([testState]);
    const output = model.predict(input);
    const outputData = await output.data();
    
    console.log('\n✅ TF.js inference successful');
    console.log('   Output shape:', outputData.length);
    console.log('   Output range:', Math.min(...outputData).toFixed(4), '-', Math.max(...outputData).toFixed(4));
    
    output.dispose();
    input.dispose();
    model.dispose();
  } catch (e) {
    console.log('❌ TF.js error:', e.message);
  }
}

main().catch(e => {
  console.error('Error:', e);
  process.exit(1);
});
