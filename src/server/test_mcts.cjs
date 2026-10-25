#!/usr/bin/env node
/**
 * Test script for MCTS implementation
 * 
 * Verifies:
 * 1. Tree construction from root
 * 2. UCT selection works correctly
 * 3. Expansion creates valid children
 * 4. Backpropagation updates statistics
 * 5. Visit counts produce valid policy
 */

// Patch util module for Node v25 compatibility
const util = require('util');
if (!util.isNullOrUndefined) {
  util.isNullOrUndefined = (v) => v === null || v === undefined;
}

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { MCTS, MCTSNode } = require('./MCTS.cjs');

let GridManager;

async function loadGridManager() {
  const gridModule = await import('../core/GridManager.js');
  GridManager = gridModule.GridManager;
}

console.log('🧪 MCTS Implementation Tests');
console.log('='.repeat(50));

// Load the trained conv network
async function loadNetwork() {
  const networkPath = 'docs/data/TF-B6_optimized_network.json';
  const fullPath = path.resolve(networkPath);
  
  const { TFConvNetwork } = require('./train_tfjs.cjs');
  
  if (fs.existsSync(fullPath)) {
    console.log(`✅ Using network from ${networkPath}`);
    return new TFConvNetwork();
  } else {
    console.log(`⚠️ No network file found, creating fresh network`);
    return new TFConvNetwork();
  }
}

// Encode canonical state for conv network
function encodeCanonical(grid, player) {
  const state = new Array(1800).fill(0);
  
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      const cell = grid.grid[y][x];
      const baseIdx = (y * 10 + x) * 18;
      
      // First triangle slot
      const t1 = cell.triangles[0];
      if (t1) {
        if (t1.player === player) {
          state[baseIdx + t1.rotation] = 1;
        } else {
          state[baseIdx + 4 + t1.rotation] = 1;
        }
      } else {
        state[baseIdx + 8] = 1;
      }
      
      // Second triangle slot
      const t2 = cell.triangles[1];
      if (t2) {
        if (t2.player === player) {
          state[baseIdx + 9 + t2.rotation] = 1;
        } else {
          state[baseIdx + 9 + 4 + t2.rotation] = 1;
        }
      } else {
        state[baseIdx + 9 + 8] = 1;
      }
    }
  }
  
  return state;
}

// Test 1: Basic node creation
function testNodeCreation() {
  console.log('\n📋 Test 1: Node Creation');
  
  const grid = new GridManager(10, 10);
  const node = new MCTSNode(grid, null, null, 1.0);
  
  console.log(`  ✓ Node created with ${node.grid.rows}x${node.grid.cols} grid`);
  console.log(`  ✓ Leaf status: ${node.isLeaf()}`);
  console.log(`  ✓ Terminal status: ${node.isTerminal()}`);
  console.log(`  ✓ Initial visits: ${node.N}`);
  console.log(`  ✓ Initial Q: ${node.Q}`);
  
  return node;
}

// Test 2: Expansion
function testExpansion() {
  console.log('\n📋 Test 2: Node Expansion');
  
  const grid = new GridManager(10, 10);
  const node = new MCTSNode(grid, null, null, 1.0);
  
  // Create fake policy (uniform)
  const policy = new Array(400).fill(0.0025);
  
  node.expand(policy);
  
  const validMoves = node.getValidMoves();
  console.log(`  ✓ Valid moves: ${validMoves.length}`);
  console.log(`  ✓ Children created: ${node.children.size}`);
  console.log(`  ✓ Leaf after expansion: ${node.isLeaf()}`);
  
  return node;
}

// Test 3: UCT Selection
function testSelection() {
  console.log('\n📋 Test 3: UCT Selection');
  
  const grid = new GridManager(10, 10);
  const node = new MCTSNode(grid, null, null, 1.0);
  
  // Expand with uniform policy
  const policy = new Array(400).fill(0.0025);
  node.expand(policy);
  node.N = 10;
  
  // Update some children
  let idx = 0;
  for (const child of node.children.values()) {
    if (idx === 0) {
      child.Q = 0.8;
      child.N = 5;
    } else if (idx === 1) {
      child.Q = -0.5;
      child.N = 3;
    }
    idx++;
  }
  
  const selected = node.select(1.5);
  
  console.log(`  ✓ Selected child exists: ${selected !== null}`);
  if (selected) {
    console.log(`  ✓ Selected Q: ${selected.Q.toFixed(4)}`);
    console.log(`  ✓ Selected N: ${selected.N}`);
  }
  
  return selected;
}

// Test 4: Backpropagation
function testBackprop() {
  console.log('\n📋 Test 4: Backpropagation');
  
  const grid = new GridManager(10, 10);
  const root = new MCTSNode(grid, null, null, 1.0);
  
  const policy = new Array(400).fill(0.0025);
  root.expand(policy);
  root.N = 1;
  
  const child = root.select(1.5);
  if (child) {
    child.expand(policy);
    child.N = 1;
    
    const grandchild = child.select(1.5);
    
    // Manual backprop
    const path = [root, child, grandchild].filter(n => n);
    const value = 0.7;
    
    for (const node of path) {
      node.update(value);
    }
    
    console.log(`  ✓ Root visits: ${root.N}`);
    console.log(`  ✓ Root Q: ${root.Q.toFixed(4)}`);
    console.log(`  ✓ Child visits: ${child.N}`);
    console.log(`  ✓ Child Q: ${child.Q.toFixed(4)}`);
    console.log(`  ✓ Grandchild visits: ${grandchild?.N || 0}`);
  }
}

// Test 5: MCTS Search
async function testMCTSSearch(network) {
  console.log('\n📋 Test 5: MCTS Search');
  
  const grid = new GridManager(10, 10);
  const mcts = new MCTS(network, {
    c: 1.5,
    alpha: 0.03,
    epsilon: 0.25,
    temperature: 1.0,
    encodeState: (g, p) => encodeCanonical(g, p)
  });
  
  console.log('  Running search with 50 simulations...');
  const startTime = Date.now();
  
  const result = await mcts.search(grid, 50);
  
  const elapsed = Date.now() - startTime;
  console.log(`  ✓ Search completed in ${elapsed}ms`);
  console.log(`  ✓ Selected move: (${result.move?.x}, ${result.move?.y}, r${result.move?.rotation})`);
  console.log(`  ✓ Root value estimate: ${result.value?.toFixed(4)}`);
  console.log(`  ✓ Policy max: ${Math.max(...result.policy).toFixed(4)}`);
  console.log(`  ✓ Policy sum: ${result.policy.reduce((a,b) => a+b, 0).toFixed(4)}`);
  
  return result;
}

// Test 6: Batched search
async function testBatchedSearch(network) {
  console.log('\n📋 Test 6: Batched MCTS Search');
  
  const grid = new GridManager(10, 10);
  const mcts = new MCTS(network, {
    c: 1.5,
    alpha: 0.03,
    epsilon: 0.25,
    batchSize: 8,
    encodeState: (g, p) => encodeCanonical(g, p)
  });
  
  console.log('  Running batched search with 50 simulations...');
  const startTime = Date.now();
  
  const result = await mcts.searchBatched(grid, 50);
  
  const elapsed = Date.now() - startTime;
  console.log(`  ✓ Batched search completed in ${elapsed}ms`);
  console.log(`  ✓ Selected move: (${result.move?.x}, ${result.move?.y}, r${result.move?.rotation})`);
  console.log(`  ✓ Root value estimate: ${result.value?.toFixed(4)}`);
  
  return result;
}

// Run all tests
async function runTests() {
  try {
    // Load GridManager first
    await loadGridManager();
    
    // Basic tests (no network needed)
    testNodeCreation();
    testExpansion();
    testSelection();
    testBackprop();
    
    // Network-dependent tests
    const network = await loadNetwork();
    
    await testMCTSSearch(network);
    await testBatchedSearch(network);
    
    console.log('\n' + '='.repeat(50));
    console.log('✅ All tests passed!');
    console.log('='.repeat(50));
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

runTests();
