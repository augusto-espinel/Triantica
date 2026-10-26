#!/usr/bin/env node
/**
 * TF.js Training Pipeline for Triantica AI
 * 
 * Supports:
 * - Pure network self-play
 * - Mixed opponents (network + random + heuristic)
 * - Cluster reward with TD(λ)
 * - Full dashboard metrics output
 * - Game auditing for bug detection
 * 
 * Usage: node src/server/train_tfjs.cjs --cycles 10 --games 100 --output data/TF-A1_training.json
 * 
 * Requires Node v20 LTS for tfjs-node native bindings (Metal GPU on M2 Ultra)
 */

// Patch util module for Node v25 compatibility with TF.js
const util = require('util');
if (!util.isNullOrUndefined) {
  util.isNullOrUndefined = (v) => v === null || v === undefined;
}

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { ReplayBuffer } = require('./ReplayBuffer.cjs');
const { MCTS } = require('./MCTS.cjs');

// Dynamic imports for ES modules (must be awaited)
let GridManager, GameAuditor, HeuristicPlayer;
let importPromise = (async () => {
  const gridModule = await import('../core/GridManager.js');
  GridManager = gridModule.GridManager;
  const auditorModule = await import('./GameAuditor.js');
  GameAuditor = auditorModule.GameAuditor;
  const heuristicModule = await import('../players/HeuristicPlayer.js');
  HeuristicPlayer = heuristicModule.HeuristicPlayer;
})();

// ============================================================
// TF.js Neural Network
// ============================================================

class TFNeuralNetwork {
  constructor(config = {}) {
    this.inputSize = config.inputSize || 401;
    this.hiddenLayers = config.hiddenLayers || [200];
    this.outputSize = config.outputSize || 400;
    this.activation = config.activation || 'sigmoid';
    this.model = null;
    this._buildModel();
  }

  _buildModel() {
    this.model = tf.sequential();
    
    this.model.add(tf.layers.dense({
      inputShape: [this.inputSize],
      units: this.hiddenLayers[0],
      activation: this.activation,
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros'
    }));
    
    for (let i = 1; i < this.hiddenLayers.length; i++) {
      this.model.add(tf.layers.dense({
        units: this.hiddenLayers[i],
        activation: this.activation,
        kernelInitializer: 'glorotUniform',
        biasInitializer: 'zeros'
      }));
    }
    
    this.model.add(tf.layers.dense({
      units: this.outputSize,
      activation: this.activation,
      kernelInitializer: 'glorotUniform',
      biasInitializer: 'zeros'
    }));
    
    this.model.compile({
      optimizer: tf.train.momentum(0.01, 0.9), // Optimized for M2 Ultra batching
      loss: 'meanSquaredError'
    });
  }

  run(input) {
    return tf.tidy(() => {
      const inputTensor = tf.tensor2d([input], [1, input.length]);
      const output = this.model.predict(inputTensor);
      return Array.from(output.dataSync());
    });
  }

  async trainBatchWithIsolation(inputs, outputs, chosenIndices, iterations = 10, learningRate = 0.05, batchSize = 256) {
    // Gradient isolation: compute loss ONLY on the chosen move per sample.
    // 
    // CRITICAL: model.fit() with custom loss function doesn't properly handle
    // closure-captured tensors. Use manual gradient computation instead.
    //
    // Optimized for M2 Ultra: configurable batch size for GPU parallelism.
    
    const optimizer = tf.train.momentum(learningRate, 0.9);
    const start = Date.now();
    const lossHistory = [];
    const actualBatchSize = Math.min(batchSize, inputs.length);
    
    for (let epoch = 0; epoch < iterations; epoch++) {
      // Shuffle indices each epoch
      const indices = Array.from({length: inputs.length}, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      let epochLoss = 0;
      let batchCount = 0;
      
      for (let b = 0; b < inputs.length; b += actualBatchSize) {
        const batchIndices = indices.slice(b, b + actualBatchSize);
        
        // Compute gradients manually
        const grads = optimizer.computeGradients(() => {
          const batchXs = tf.tensor2d(
            batchIndices.map(i => inputs[i]),
            [batchIndices.length, inputs[0].length]
          );
          
          const preds = this.model.predict(batchXs);
          const predData = preds.dataSync();
          
          // Build targets: copy predictions, override chosen moves with targets
          const outputSize = outputs[0].length;
          const targetData = new Float32Array(batchIndices.length * outputSize);
          
          for (let s = 0; s < batchIndices.length; s++) {
            const sampleIdx = batchIndices[s];
            const chosenIdx = chosenIndices[sampleIdx];
            
            // Copy prediction (zero gradient for unchosen moves)
            for (let o = 0; o < outputSize; o++) {
              targetData[s * outputSize + o] = predData[s * outputSize + o];
            }
            
            // Override chosen move with target value
            targetData[s * outputSize + chosenIdx] = outputs[sampleIdx][chosenIdx];
          }
          
          const batchYs = tf.tensor2d(targetData, [batchIndices.length, outputSize]);
          
          // Compute loss (MSE, but only chosen moves have non-zero error)
          const loss = preds.sub(batchYs).square().mean();
          
          batchXs.dispose();
          batchYs.dispose();
          preds.dispose();
          
          return loss;
        });
        
        // Apply gradients
        optimizer.applyGradients(grads.grads);
        
        const lossVal = grads.value.dataSync()[0];
        epochLoss += lossVal;
        batchCount++;
        
        // Dispose gradient tensors
        for (const tensor of Object.values(grads.grads)) {
          if (tensor) tensor.dispose();
        }
        grads.value.dispose();
      }
      
      lossHistory.push(epochLoss / batchCount);
    }
    
    return {
      initialError: lossHistory[0],
      finalError: lossHistory[lossHistory.length - 1],
      errorHistory: lossHistory,
      iterations,
      timeMs: Date.now() - start
    };
  }

  async trainBatch(inputs, outputs, iterations = 10, learningRate = 0.05, batchSize = 256) {
    const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const ys = tf.tensor2d(outputs, [outputs.length, outputs[0].length]);
    
    this.model.compile({
      optimizer: tf.train.momentum(learningRate, 0.9),
      loss: 'meanSquaredError'
    });
    
    const start = Date.now();
    const history = await this.model.fit(xs, ys, {
      epochs: iterations,
      batchSize: Math.min(batchSize, inputs.length),
      shuffle: true,
      verbose: 0
    });
    
    xs.dispose();
    ys.dispose();
    
    const losses = history.history.loss;
    return {
      initialError: losses[0],
      finalError: losses[losses.length - 1],
      errorHistory: losses,
      iterations,
      timeMs: Date.now() - start
    };
  }

  toJSON() {
    return tf.tidy(() => {
      const result = {
        type: 'NeuralNetwork',
        sizes: [this.inputSize, ...this.hiddenLayers, this.outputSize],
        layers: [],
        options: {
          inputSize: this.inputSize,
          outputSize: this.outputSize,
          hiddenLayers: this.hiddenLayers,
          activation: this.activation,
          binaryThresh: 0.5
        },
        trainOpts: {
          activation: this.activation,
          iterations: 30,
          errorThresh: 0.005,
          learningRate: 0.1
        }
      };

      for (let i = 0; i < this.model.layers.length; i++) {
        const [kernel, bias] = this.model.layers[i].getWeights();
        const kArr = kernel.arraySync();
        const bArr = bias.arraySync();
        
        // brain.js format: weights[target][source]
        const transposed = [];
        for (let t = 0; t < kernel.shape[1]; t++) {
          transposed.push([]);
          for (let s = 0; s < kernel.shape[0]; s++) {
            transposed[t].push(kArr[s][t]);
          }
        }
        
        result.layers.push({ weights: transposed, biases: bArr });
      }
      
      return result;
    });
  }

  fromJSON(json) {
    tf.tidy(() => {
      this.inputSize = json.options?.inputSize || json.sizes?.[0] || 401;
      this.hiddenLayers = json.options?.hiddenLayers || (json.sizes ? json.sizes.slice(1, -1) : [200]);
      this.outputSize = json.options?.outputSize || json.sizes?.[json.sizes.length - 1] || 400;
      this.activation = json.options?.activation || 'sigmoid';

      this.model = tf.sequential();
      this.model.add(tf.layers.dense({
        inputShape: [this.inputSize],
        units: this.hiddenLayers[0],
        activation: this.activation
      }));
      
      for (let i = 1; i < this.hiddenLayers.length; i++) {
        this.model.add(tf.layers.dense({
          units: this.hiddenLayers[i],
          activation: this.activation
        }));
      }
      
      this.model.add(tf.layers.dense({
        units: this.outputSize,
        activation: this.activation
      }));
      
      this.model.compile({ optimizer: tf.train.sgd(0.1), loss: 'meanSquaredError' });

      for (let i = 0; i < this.model.layers.length; i++) {
        // TF.js saved format: layers[0] is first dense layer
        // Brain.js format: layers[0] is input layer (no weights), layers[1] is first dense layer
        const brainLayer = json.layers[i] || json.layers[i + 1];
        if (!brainLayer?.weights) continue;
        
        const src = brainLayer.weights[0]?.length || 0;
        const tgt = brainLayer.weights.length || 0;
        
        const k = [];
        for (let s = 0; s < src; s++) {
          k.push([]);
          for (let t = 0; t < tgt; t++) {
            k[s].push(brainLayer.weights[t][s]);
          }
        }
        
        this.model.layers[i].setWeights([
          tf.tensor2d(k, [src, tgt]),
          tf.tensor1d(brainLayer.biases)
        ]);
      }
    });
  }
}

// ============================================================
// TF.js Convolutional Neural Network (Phase 2)
// ============================================================

/**
 * Conv network with dual heads: Policy (move probabilities) + Value (position score)
 * 
 * Input: 10×10×18 canonical one-hot tensor
 * Backbone: 3 Conv2D layers for spatial pattern recognition
 * Policy head: 400 outputs (one per possible move)
 * Value head: 1 output (position evaluation, -1 to +1)
 * 
 * This architecture matches the design principles from AlphaGo Zero:
 * - Conv layers learn local spatial patterns (enclosures, territory boundaries)
 * - Policy head: "What should I do?"
 * - Value head: "How good is my position?"
 * - Canonical encoding makes pattern detection trivial for conv filters
 */
class TFConvNetwork {
  constructor(config = {}) {
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.channels = config.channels || 18; // canonical encoding channels
    this.numFilters = config.numFilters || [32, 64, 64];
    this.policySize = this.rows * this.cols * 4; // 400 possible moves
    this.model = null;
    this._buildModel();
  }

  _buildModel() {
    // Input: [batch, 10, 10, 18]
    const input = tf.input({ shape: [this.rows, this.cols, this.channels] });
    
    // Backbone: 3 Conv2D layers with batch normalization and ReLU
    let x = input;
    for (let i = 0; i < this.numFilters.length; i++) {
      x = tf.layers.conv2d({
        filters: this.numFilters[i],
        kernelSize: 3,
        padding: 'same',
        kernelInitializer: 'glorotUniform',
        biasInitializer: 'zeros'
      }).apply(x);
      x = tf.layers.batchNormalization().apply(x);
      x = tf.layers.reLU().apply(x);
    }
    
    // Policy head: "What move should I make?"
    // Conv → Flatten → Dense → Softmax-like (sigmoid for [0,1] range matching training data)
    const policyConv = tf.layers.conv2d({
      filters: 2,
      kernelSize: 1,
      padding: 'same',
      kernelInitializer: 'glorotUniform'
    }).apply(x);
    const policyFlat = tf.layers.flatten().apply(policyConv);
    const policyDense = tf.layers.dense({
      units: this.policySize,
      activation: 'sigmoid',
      kernelInitializer: 'glorotUniform'
    }).apply(policyFlat);
    
    // Value head: "How good is my position?"
    // Conv → Flatten → Dense → Dense → tanh (-1 to +1)
    const valueConv = tf.layers.conv2d({
      filters: 1,
      kernelSize: 1,
      padding: 'same',
      kernelInitializer: 'glorotUniform'
    }).apply(x);
    const valueFlat = tf.layers.flatten().apply(valueConv);
    const valueDense1 = tf.layers.dense({
      units: 64,
      activation: 'relu',
      kernelInitializer: 'glorotUniform'
    }).apply(valueFlat);
    const valueDense2 = tf.layers.dense({
      units: 1,
      activation: 'tanh',
      kernelInitializer: 'glorotUniform'
    }).apply(valueDense1);
    
    this.model = tf.model({
      inputs: input,
      outputs: [policyDense, valueDense2]
    });
    
    // Compile with separate losses for each head
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: ['meanSquaredError', 'meanSquaredError'],
      lossWeights: [1.0, 1.0] // Equal weights for policy and value (AlphaGo Zero style)
    });
  }

  /**
   * Run a forward pass — returns { policy, value }
   */
  run(input1d) {
    return tf.tidy(() => {
      // Reshape flat 1800 array to [1, 10, 10, 18]
      const inputTensor = tf.tensor4d(new Float32Array(input1d), [1, this.rows, this.cols, this.channels]);
      const [policyOut, valueOut] = this.model.predict(inputTensor);
      return {
        policy: Array.from(policyOut.dataSync()),
        value: valueOut.dataSync()[0]
      };
    });
  }

  /**
   * Batch prediction for MCTS optimization
   * Evaluates multiple positions in a single forward pass
   * 
   * @param {number[][]} inputs - Array of canonical state vectors (1800 each)
   * @returns {Object} { policies: number[][], values: number[] }
   */
  async batchPredict(inputs) {
    // Filter out empty inputs
    const validInputs = inputs.filter(arr => arr && arr.length > 0);
    
    if (validInputs.length === 0) {
      // Return empty arrays with proper structure
      return { policies: [], values: [] };
    }
    
    // Track mapping from original indices to valid indices
    const validIndices = [];
    inputs.forEach((inp, i) => {
      if (inp && inp.length > 0) validIndices.push(i);
    });
    
    return tf.tidy(() => {
      // Build batch tensor from valid inputs only
      const batchData = new Float32Array(validInputs.length * this.rows * this.cols * this.channels);
      for (let i = 0; i < validInputs.length; i++) {
        batchData.set(new Float32Array(validInputs[i]), i * this.rows * this.cols * this.channels);
      }
      const batchTensor = tf.tensor4d(batchData, [validInputs.length, this.rows, this.cols, this.channels]);
      
      // Batch prediction
      const [policyOut, valueOut] = this.model.predict(batchTensor);
      
      // Extract results - map back to original indices
      const policies = new Array(inputs.length).fill(null);
      const values = new Array(inputs.length).fill(0);
      const policyData = policyOut.dataSync();
      const valueData = valueOut.dataSync();
      
      for (let i = 0; i < validInputs.length; i++) {
        const origIdx = validIndices[i];
        policies[origIdx] = Array.from(policyData.slice(i * 400, (i + 1) * 400));
        values[origIdx] = valueData[i];
      }
      
      return { policies, values };
    });
  }

  /**
   * Train on a batch with gradient isolation (only update chosen move in policy)
   * or full distribution training when MCTS visit distributions are provided.
   * Also trains value head toward the game outcome.
   * 
   * @param {number[][]} inputs - Canonical state vectors (1800 each)
   * @param {number[][]} policyTargets - Full policy target vectors (400 each)
   * @param {number[]} chosenIndices - Which move was chosen per sample
   * @param {number[]} valueTargets - Game outcome per sample (-1, 0, +1)
   * @param {number} iterations - Epochs
   * @param {number} learningRate - LR
   * @param {number} batchSize - Mini-batch size
   * @param {boolean} isolateChosenOnly - If true, mask unchosen moves; if false (MCTS), train full visit distribution
   */
  async trainBatchWithIsolation(inputs, policyTargets, chosenIndices, valueTargets, iterations = 10, learningRate = 0.001, batchSize = 256, isolateChosenOnly = true) {
    // Use Adam optimizer (cap legacy high momentum LR >= 0.01 down to 0.001 for Adam stability)
    const adamLr = learningRate >= 0.01 ? 0.001 : learningRate;
    if (!this._adamOptimizer || this._adamLr !== adamLr) {
      this._adamOptimizer = tf.train.adam(adamLr);
      this._adamLr = adamLr;
    }
    const optimizer = this._adamOptimizer;
    const start = Date.now();
    const lossHistory = [];
    const actualBatchSize = Math.min(batchSize, inputs.length);
    
    // Pre-convert inputs to Float32Arrays for efficient tensor creation
    const float32Inputs = inputs.map(inp => new Float32Array(inp));
    const inputSize = this.rows * this.cols * this.channels;
    
    for (let epoch = 0; epoch < iterations; epoch++) {
      // Shuffle
      const indices = Array.from({length: inputs.length}, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      let epochLoss = 0;
      let batchCount = 0;
      
      for (let b = 0; b < inputs.length; b += actualBatchSize) {
        const batchIndices = indices.slice(b, b + actualBatchSize);
        
        const grads = optimizer.computeGradients(() => {
          // Build batch tensor from flat Float32Arrays
          const batchData = new Float32Array(batchIndices.length * inputSize);
          for (let s = 0; s < batchIndices.length; s++) {
            batchData.set(float32Inputs[batchIndices[s]], s * inputSize);
          }
          const batchXs = tf.tensor4d(batchData, [batchIndices.length, this.rows, this.cols, this.channels]);
          
          const [policyPred, valuePred] = this.model.predict(batchXs);
          const outputSize = this.policySize;
          
          let policyLoss;
          let policyTargetTensor;
          
          if (isolateChosenOnly) {
            const policyData = policyPred.dataSync();
            // Build isolated policy targets: copy predictions, override chosen moves only
            const policyTargetData = new Float32Array(batchIndices.length * outputSize);
            for (let s = 0; s < batchIndices.length; s++) {
              const sampleIdx = batchIndices[s];
              // Copy predictions (zero gradient for unchosen moves)
              for (let o = 0; o < outputSize; o++) {
                policyTargetData[s * outputSize + o] = policyData[s * outputSize + o];
              }
              // Override chosen move with target
              policyTargetData[s * outputSize + chosenIndices[sampleIdx]] = policyTargets[sampleIdx][chosenIndices[sampleIdx]];
            }
            policyTargetTensor = tf.tensor2d(policyTargetData, [batchIndices.length, outputSize]);
            // Sum over the 400 action dimensions so the single active move gradient is not diluted 400x
            policyLoss = policyPred.sub(policyTargetTensor).square().sum(-1).mean();
          } else {
            // MCTS full visit distribution: train via categorical cross-entropy over logSoftmax of sigmoid logits
            const policyTargetData = new Float32Array(batchIndices.length * outputSize);
            for (let s = 0; s < batchIndices.length; s++) {
              const sampleIdx = batchIndices[s];
              const targetVec = policyTargets[sampleIdx];
              for (let o = 0; o < outputSize; o++) {
                policyTargetData[s * outputSize + o] = targetVec[o];
              }
            }
            policyTargetTensor = tf.tensor2d(policyTargetData, [batchIndices.length, outputSize]);
            const eps = 1e-7;
            const pClip = policyPred.clipByValue(eps, 1 - eps);
            const logits = pClip.div(tf.scalar(1).sub(pClip)).log();
            const logProbs = tf.logSoftmax(logits, -1);
            policyLoss = policyTargetTensor.mul(logProbs).neg().sum(-1).mean();
          }
          
          // Value targets
          const valueTargetTensor = tf.tensor2d(
            batchIndices.map(i => [valueTargets[i]]),
            [batchIndices.length, 1]
          );
          
          const valueLoss = valuePred.sub(valueTargetTensor).square().mean();
          
          const totalLoss = policyLoss.add(valueLoss); // Equal weight for policy and value (AlphaGo Zero style)
          
          batchXs.dispose();
          policyTargetTensor.dispose();
          valueTargetTensor.dispose();
          policyPred.dispose();
          valuePred.dispose();
          
          return totalLoss;
        });
        
        optimizer.applyGradients(grads.grads);
        
        const lossVal = grads.value.dataSync()[0];
        epochLoss += lossVal;
        batchCount++;
        
        for (const tensor of Object.values(grads.grads)) {
          if (tensor) tensor.dispose();
        }
        grads.value.dispose();
      }
      
      lossHistory.push(epochLoss / batchCount);
    }
    
    return {
      initialError: lossHistory[0],
      finalError: lossHistory[lossHistory.length - 1],
      errorHistory: lossHistory,
      iterations,
      timeMs: Date.now() - start
    };
  }

  /**
   * Standard batch training (no gradient isolation, no value head)
   * Used for backward compatibility / ablation studies
   */
  async trainBatch(inputs, policyTargets, iterations = 10, learningRate = 0.01, batchSize = 256) {
    const inputSize = this.rows * this.cols * this.channels;
    const batchData = new Float32Array(inputs.length * inputSize);
    for (let i = 0; i < inputs.length; i++) {
      batchData.set(new Float32Array(inputs[i]), i * inputSize);
    }
    const xs = tf.tensor4d(batchData, [inputs.length, this.rows, this.cols, this.channels]);
    const ys = tf.tensor2d(policyTargets, [policyTargets.length, this.policySize]);
    
    this.model.compile({
      optimizer: tf.train.momentum(learningRate, 0.9),
      loss: ['meanSquaredError', 'meanSquaredError'],
      lossWeights: [1.0, 0.5]
    });
    
    const start = Date.now();
    const history = await this.model.fit(xs, ys, {
      epochs: iterations,
      batchSize: Math.min(batchSize, inputs.length),
      shuffle: true,
      verbose: 0
    });
    
    xs.dispose();
    ys.dispose();
    
    const losses = history.history.loss;
    return {
      initialError: losses[0],
      finalError: losses[losses.length - 1],
      errorHistory: losses,
      iterations,
      timeMs: Date.now() - start
    };
  }

  toJSON() {
    return tf.tidy(() => {
      const result = {
        type: 'ConvNetwork',
        architecture: {
          rows: this.rows,
          cols: this.cols,
          channels: this.channels,
          numFilters: this.numFilters,
          policySize: this.policySize
        },
        weights: []
      };
      
      for (const layer of this.model.layers) {
        const weights = layer.getWeights();
        if (weights.length > 0) {
          result.weights.push({
            name: layer.name,
            data: weights.map(w => Array.from(w.dataSync()))
          });
        }
      }
      
      return result;
    });
  }

  fromJSON(json) {
    tf.tidy(() => {
      this.rows = json.architecture?.rows || 10;
      this.cols = json.architecture?.cols || 10;
      this.channels = json.architecture?.channels || 18;
      this.numFilters = json.architecture?.numFilters || [32, 64, 64];
      this.policySize = json.architecture?.policySize || 400;
      
      this._buildModel();
      
      if (json.weights) {
        let weightIdx = 0;
        for (const layer of this.model.layers) {
          const layerWeights = layer.getWeights();
          if (layerWeights.length > 0 && weightIdx < json.weights.length) {
            const saved = json.weights[weightIdx];
            const restored = saved.data.map((arr, i) => {
              const original = layerWeights[i];
              return tf.tensor(arr, original.shape);
            });
            try {
              layer.setWeights(restored);
            } catch (e) {
              // Shape mismatch — skip this layer
            }
            restored.forEach(t => t.dispose());
            weightIdx++;
          }
        }
      }
    });
  }
}

// ============================================================
// Helper Functions
// ============================================================

/**
 * FIXED PERSPECTIVE state representation (matching original brain.js)
 * 
 * AI always sees: my pieces = 1, enemy pieces = -1
 * Independent of which player number AI is assigned
 * 
 * @param {GridManager} grid - The game state
 * @param {number} aiPlayerNumber - Which player the AI is playing as (1 or 2)
 * @returns {number[]} - 401-element state vector
 */
function getStateFixed(grid, aiPlayerNumber) {
  const state = [];
  const { rows, cols } = grid;
  const opponent = aiPlayerNumber === 1 ? 2 : 1;
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid.grid[r][c];
      const t1 = cell.triangles[0], t2 = cell.triangles[1];
      
      // Player value: AI's pieces = 1, opponent = -1, empty = 0
      const pv = t => {
        if (!t) return 0;
        return t.player === aiPlayerNumber ? 1 : -1;
      };
      
      // Rotation: normalized 0-0.75 or -1 for empty
      const rv = r => r === undefined || r === -1 ? -1 : r / 4;
      
      state.push(pv(t1), rv(t1?.rotation ?? -1), pv(t2), rv(t2?.rotation ?? -1));
    }
  }
  // Final element: always 1 (from AI's perspective)
  state.push(1);
  return state;
}

/**
 * CANONICAL 1-TO-1 SLOT ENCODING
 * 
 * Every possible placement is a discrete binary fact: present or absent.
 * No implicit ordering between rotations. Empty is first-class.
 * Triantica is Markovian — only current board state matters.
 * 
 * Per cell, per diagonal group (╱ vs ╲):
 *   Channel 0: empty
 *   Channel 1-4: my piece, rotation 0°/90°/180°/270°
 *   Channel 5-8: opponent, rotation 0°/90°/180°/270°
 * 
 * 9 channels × 2 groups = 18 channels per cell
 * Input tensor: 10×10×18 = 1,800 values
 * 
 * @param {GridManager} grid - The game state
 * @param {number} aiPlayerNumber - Which player the AI is playing as (1 or 2)
 * @returns {number[]} - 1,800-element flat array (reshape to 10×10×18 for conv)
 */
function getStateCanonical(grid, aiPlayerNumber) {
  if (typeof grid.getCanonicalState === 'function') {
    return grid.getCanonicalState(aiPlayerNumber);
  }
  const { rows, cols } = grid;
  const channels = rows * cols * 18;
  const state = new Float32Array(channels);
  
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid.grid[r][c];
      const cellOffset = (r * cols + c) * 18;
      
      // Process each non-overlapping slot (0 = rotations 0,1; 1 = rotations 2,3)
      for (let slot = 0; slot < 2; slot++) {
        const slotOffset = cellOffset + slot * 9;
        const tri = cell.triangles.find(t => (slot === 0 ? t.rotation < 2 : t.rotation >= 2));
        
        if (!tri) {
          state[slotOffset] = 1;
        } else {
          const ownerOffset = tri.player === aiPlayerNumber ? 0 : 4;
          state[slotOffset + 1 + ownerOffset + tri.rotation] = 1;
        }
      }
    }
  }
  
  return Array.from(state);
}

// Keep alias for compatibility
const getStateRelative = getStateFixed;

function isValidMove(grid, x, y, rot) {
  // Use GridManager's native method
  return grid.isValidMove(x, y, rot);
}

function getRandomMove(grid) {
  const valid = [];
  const { rows, cols } = grid;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      for (let r = 0; r < 4; r++) {
        if (isValidMove(grid, x, y, r)) valid.push({ x, y, rotation: r });
      }
    }
  }
  return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
}

function getAIMove(network, grid, exploration, aiPlayerNumber = 1, useConv = false) {
  if (Math.random() < exploration) return getRandomMove(grid);
  
  let policy;
  if (useConv) {
    const state = getStateCanonical(grid, aiPlayerNumber);
    const result = network.run(state);
    policy = result.policy;
  } else {
    const state = getStateFixed(grid, aiPlayerNumber);
    policy = network.run(state);
  }
  
  const moves = [];
  const { rows, cols } = grid;
  
  for (let i = 0; i < policy.length; i++) {
    const rot = i % 4;
    const idx = Math.floor(i / 4);
    const x = idx % cols;
    const y = Math.floor(idx / cols);
    moves.push({ x, y, rotation: rot, score: policy[i] });
  }
  
  moves.sort((a, b) => b.score - a.score);
  
  for (const m of moves) {
    if (isValidMove(grid, m.x, m.y, m.rotation)) return m;
  }
  
  return getRandomMove(grid);
}

/**
 * MCTS-enhanced move selection
 * Uses Monte Carlo Tree Search with neural network evaluation
 * Returns both move and improved policy for training
 */
async function getMCTSMove(network, grid, simulations, aiPlayerNumber = 1, batchSize = 8) {
  const mcts = new MCTS(network, {
    c: 1.5,
    alpha: 0.03,
    epsilon: 0.25,
    batchSize: batchSize,
    temperature: 1.0,
    encodeState: (g, p) => getStateCanonical(g, p)
  });
  
  const result = await mcts.searchBatched(grid, simulations);
  return {
    move: result.move,
    policy: result.policy,  // Improved policy from visit counts
    value: result.value    // Position evaluation
  };
}

function computeMedian(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Diversity tracking functions
function computeEntropy(counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let entropy = 0;
  for (const c of counts) {
    if (c > 0) {
      const p = c / total;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

function computeOpeningDiversity(firstMoves) {
  if (firstMoves.length === 0) return 0;
  const counts = {};
  for (const move of firstMoves) {
    counts[move] = (counts[move] || 0) + 1;
  }
  return computeEntropy(Object.values(counts));
}

function computeBoardDiversity(finalBoards) {
  if (finalBoards.length === 0) return 0;
  const numCells = finalBoards[0].length;
  let totalEntropy = 0;
  
  for (let i = 0; i < numCells; i++) {
    const stateCounts = {};
    for (const board of finalBoards) {
      const state = `${board[i].t1},${board[i].t2}`;
      stateCounts[state] = (stateCounts[state] || 0) + 1;
    }
    totalEntropy += computeEntropy(Object.values(stateCounts));
  }
  return totalEntropy / numCells;
}

// ============================================================
// Immediate Reward Calculation
// ============================================================

function calculateImmediateRewardVector(grid, player) {
  const opponent = player === 1 ? 2 : 1;
  const { rows, cols } = grid;
  const outputVector = new Array(rows * cols * 4).fill(0.01);
  
  // CRITICAL: Match original AIPlayer.js factors
  const claimRewardFactor = 0.3;
  const connectedClusterRewardFactor = 0.1;
  // NO largestLeadFactor in immediate reward (was added incorrectly)
  
  // Dynamic range calculation (matching original)
  const maxPossibleClaimBonus = 1.5; // Estimate: max ~2 triangles claimed
  const maxPossibleClusterSize = rows * cols * 2;
  const maxPossibleClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;
  const minImmediateReward = 0;
  const maxImmediateReward = maxPossibleClaimBonus + maxPossibleClusterBonus;
  const rewardRange = maxImmediateReward - minImmediateReward;
  
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      for (let rotation = 0; rotation < 4; rotation++) {
        const moveIndex = (y * cols + x) * 4 + rotation;
        
        if (!isValidMove(grid, x, y, rotation)) continue;
        
        const tempManager = grid.clone();
        tempManager.currentPlayer = player;
        
        const simSuccess = tempManager.placeTriangle(x, y, rotation, player);
        
        if (simSuccess) {
          const simClaimed = tempManager.claimEnclosedTriangles(player);
          const hypoClaimReward = simClaimed * claimRewardFactor;
          
          const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);
          const hypoClusterReward = simClusterSize * connectedClusterRewardFactor;
          
          // NO lead delta in immediate reward (removed - not in original)
          
          const hypotheticalReward = hypoClaimReward + hypoClusterReward;
          
          let scaledReward = rewardRange !== 0
            ? (hypotheticalReward - minImmediateReward) / rewardRange
            : 0.5;
          
          outputVector[moveIndex] = Math.max(0.0, Math.min(1.0, scaledReward));
        }
      }
    }
  }
  
  return outputVector;
}

// ============================================================
// Main Training Loop
// ============================================================

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name, def) => {
    const i = args.indexOf('--' + name);
    return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
  };
  
  const cycles = parseInt(getArg('cycles', '10'));
  const games = parseInt(getArg('games', '100'));
  const evalGames = parseInt(getArg('eval-games', '100'));
  const iterations = parseInt(getArg('iterations', '10'));
  const learningRate = parseFloat(getArg('learning-rate', '0.05'));
  const exploration = parseFloat(getArg('exploration', '0.3'));
  const selfPlayMode = getArg('self-play-mode', 'network');
  const opponentMix = getArg('opponent-mix', 'network');
  const networkRatio = parseFloat(getArg('network-ratio', '1.0'));
  const randomRatio = parseFloat(getArg('random-ratio', '0.0'));
  const heuristicRatio = parseFloat(getArg('heuristic-ratio', '0.0'));
  const startFrom = getArg('start-from', null);
  const output = getArg('output', 'data/training.json');
  const enableClusterReward = getArg('enable-cluster-reward', 'false') === 'true';
  const audit = getArg('audit', 'false') === 'true';
  const auditRate = parseFloat(getArg('audit-rate', '0.05'));
  const useGradientIsolation = getArg('gradient-isolation', 'true') === 'true';
  const batchSize = parseInt(getArg('batch-size', '256'));
  const useReplayBuffer = getArg('replay-buffer', 'true') === 'true';
  const replayBufferSize = parseInt(getArg('replay-buffer-size', '500000'));
  const replayBatchSize = parseInt(getArg('replay-batch-size', '256'));
  const replayRatio = parseFloat(getArg('replay-ratio', '2.0')); // How many replay batches per cycle vs new data batches
  const architecture = getArg('architecture', 'dense'); // 'dense' or 'conv'
  const mctsSims = parseInt(getArg('mcts-sims', '0')); // MCTS simulations per move (0 = disabled)
  const mctsBatchSize = parseInt(getArg('mcts-batch-size', '8')); // Batch size for parallel MCTS evaluation
  
  console.log('🧪  TF.js Training Pipeline');
  console.log('='.repeat(60));
  console.log(`  Node version:    ${process.version}`);
  console.log(`  TF.js version:   ${tf.version.tfjs}`);
  console.log(`  Backend:         ${tf.getBackend()}`);
  console.log(`  Platform:        ${process.platform} (${process.arch})`);
  console.log('='.repeat(60));
  console.log(`  Cycles:          ${cycles}`);
  console.log(`  Games/cycle:     ${games}`);
  console.log(`  Self-play mode:  ${selfPlayMode}`);
  console.log(`  Opponent mix:    ${opponentMix}`);
  if (opponentMix === 'mixed') {
    const total = networkRatio + randomRatio + heuristicRatio;
    console.log(`  Network ratio:   ${(networkRatio / total * 100).toFixed(0)}%`);
    console.log(`  Random ratio:     ${(randomRatio / total * 100).toFixed(0)}%`);
    console.log(`  Heuristic ratio:  ${(heuristicRatio / total * 100).toFixed(0)}%`);
  }
  console.log(`  Eval games:      ${evalGames}`);
  console.log(`  Train iters:     ${iterations}`);
  console.log(`  Learning rate:   ${learningRate}`);
  console.log(`  Batch size:      ${batchSize}`);
  console.log(`  Gradient iso:    ${useGradientIsolation}`);
  console.log(`  Exploration:      ${exploration}`);
  console.log(`  Cluster reward:   ${enableClusterReward}`);
  console.log(`  Replay buffer:    ${useReplayBuffer ? 'ENABLED' : 'disabled'}`);
  if (useReplayBuffer) {
    console.log(`  Replay size:      ${replayBufferSize.toLocaleString()}`);
    console.log(`  Replay batch:     ${replayBatchSize}`);
    console.log(`  Replay ratio:     ${replayRatio}x`);
  }
  console.log(`  Architecture:     ${architecture}`);
  if (mctsSims > 0) {
    console.log(`  MCTS sims/move:  ${mctsSims}`);
    console.log(`  MCTS batch size: ${mctsBatchSize}`);
  }
  console.log(`  Audit mode:       ${audit ? 'ENABLED' + (audit ? ' (' + (auditRate * 100).toFixed(0) + '% sample rate)' : '') : 'disabled'}`);
  console.log(`  Start from:      ${startFrom || 'fresh'}`);
  console.log(`  Output:          ${output}`);
  console.log('='.repeat(60));
  
  const grid = new GridManager(10, 10);
  const useConv = architecture === 'conv';
  let network;
  
  if (useConv) {
    network = new TFConvNetwork({
      rows: 10,
      cols: 10,
      channels: 18,
      numFilters: [32, 64, 64]
    });
  } else {
    network = new TFNeuralNetwork({
      inputSize: 401,
      hiddenLayers: [200],
      outputSize: 400,
      activation: 'sigmoid'
    });
  }
  
  if (startFrom) {
    const data = JSON.parse(fs.readFileSync(startFrom, 'utf-8'));
    network.fromJSON(data);
    console.log(`📂 Loaded network from ${startFrom}`);
  }
  
  // Sanity check
  if (useConv) {
    const testState = new Array(1800).fill(0);
    const testOut = network.run(testState);
    console.log(`🧠 Conv network ready (policy range: ${Math.min(...testOut.policy).toFixed(3)} - ${Math.max(...testOut.policy).toFixed(3)}, value: ${testOut.value.toFixed(3)})`);
  } else {
    const testState = new Array(401).fill(0);
    const testOut = network.run(testState);
    console.log(`🧠 Dense network ready (output range: ${Math.min(...testOut).toFixed(3)} - ${Math.max(...testOut).toFixed(3)})`);
  }
  
  const log = {
    experimentId: path.basename(output, '.json').replace('_training', ''),
    startedAt: new Date().toISOString(),
    options: {
      cycles,
      games,
      evalGames,
      iterations,
      learningRate,
      exploration,
      selfPlayMode,
      opponentMix,
      networkRatio,
      randomRatio,
      heuristicRatio,
      startFrom,
      enableClusterReward,
      iterations,
      learningRate,
      batchSize,
      gradientIsolation: useGradientIsolation,
      architecture: architecture,
      replayBuffer: useReplayBuffer,
      replayBufferSize,
      replayBatchSize,
      replayRatio,
      audit,
      auditRate,
      framework: 'tensorflow.js',
      backend: tf.getBackend()
    },
    cycles: []
  };
  
  // Initialize auditor if enabled
  const auditor = new GameAuditor({
    enabled: audit,
    sampleRate: auditRate,
    verbose: audit
  });

  if (audit) {
    console.log(`🔍  Game auditing ENABLED (${(auditRate * 100).toFixed(0)}% sample rate)`);
  }

  // Initialize replay buffer if enabled
  const replayBuffer = useReplayBuffer ? new ReplayBuffer({
    maxSize: replayBufferSize,
    alpha: 0.6,
    beta: 0.4,
    betaGrowth: 0.001
  }) : null;

  if (useReplayBuffer) {
    console.log(`🗄️  Replay buffer initialized (max ${replayBufferSize.toLocaleString()} samples)`);
  }
  
  // Training reward constants
  // CRITICAL: Match original AIPlayer.js range calculation
  const finalRewardWeight = 1.0;
  const minImmediateReward = 0.01; // Min output from immediate calculation
  const maxImmediateReward = 0.99; // Max output from immediate calculation
  const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight);
  const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight);
  const combinedRewardRange = maxCombinedReward - minCombinedReward;
  
  for (let cycle = 1; cycle <= cycles; cycle++) {
    console.log(`\n🔄 Cycle ${cycle}/${cycles}`);
    const cycleStart = Date.now();
    
    // Self-play with opponent tracking
    const allInputs = [], allOutputs = [], allChosenIndices = [], allValueTargets = [];
    const winsByOpponent = {};
    const opponentTypeCounts = {};
    const scoresByOpponent = {};
    const firstMoves = [];  // Track opening moves for diversity
    const finalBoards = [];  // Track final board states for diversity
    
    let p1Wins = 0, p2Wins = 0, draws = 0;
    
    // Self-play loop (sync version for non-MCTS mode)
    for (let g = 0; g < games; g++) {
      // MCTS mode uses async version below
      if (mctsSims > 0) continue;
      grid.reset();
      let curP = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      const hist = [];
      
      // FIXED PERSPECTIVE: AI alternates which player it plays as
      // This ensures balanced training for both starting and responding positions
      const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;
      
      // Determine opponent type for this game
      let opponentType = 'network';
      if (opponentMix === 'random') {
        opponentType = 'random';
      } else if (opponentMix === 'heuristic') {
        opponentType = 'heuristic';
      } else if (opponentMix === 'mixed') {
        const total = networkRatio + randomRatio + heuristicRatio;
        const r = Math.random() * total;
        if (r < networkRatio) {
          opponentType = 'network';
        } else if (r < networkRatio + randomRatio) {
          opponentType = 'random';
        } else {
          opponentType = 'heuristic';
        }
      }
      
      opponentTypeCounts[opponentType] = (opponentTypeCounts[opponentType] || 0) + 1;
      if (!winsByOpponent[opponentType]) {
        winsByOpponent[opponentType] = { 1: 0, 2: 0, draws: 0, aiWins: 0, aiLosses: 0, aiDraws: 0, aiAsP1Wins: 0, aiAsP1Games: 0, aiAsP2Wins: 0, aiAsP2Games: 0 };
        scoresByOpponent[opponentType] = { p1: [], p2: [], ai: [], opponent: [] };
      }
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = curP;
        
        const state = useConv ? getStateCanonical(grid, aiPlayerNumber) : getStateFixed(grid, aiPlayerNumber);
        const immediateRewardVector = calculateImmediateRewardVector(grid, curP);
        // Don't calculate rawImmediateReward here - we'll use chosenMoveReward after move selection
        
        // Determine if current player is AI
        const isAIP = curP === aiPlayerNumber;
        
        let move;
        if (isAIP) {
          // AI's turn - use network
          move = getAIMove(network, grid, exploration, aiPlayerNumber, useConv);
        } else {
          // Opponent's turn
          if (opponentType === 'network') {
            // Self-play: opponent also uses network (but from THEIR perspective)
            move = getAIMove(network, grid, exploration, curP, useConv);
          } else if (opponentType === 'heuristic') {
            // Create heuristic with correct player number for this turn
            const heuristicOpp = new HeuristicPlayer(curP, { explorationFactor: 0 });
            move = heuristicOpp.getMove(grid);
          } else {
            move = getRandomMove(grid);
          }
        }
        
        if (!move) break;
        
        // Track first move for opening diversity
        if (m === 0) {
          firstMoves.push(`${move.x},${move.y},r${move.rotation}`);
        }
        
        const execRes = grid.executeMove(move.x, move.y, move.rotation, curP);
        if (!execRes.success) break;
        
        const moveIdx = (move.y * grid.cols + move.x) * 4 + move.rotation;
        
        // Store the immediate reward value at the CHOSEN move index
        // CRITICAL: Use vector[moveIdx], NOT average of all values
        const chosenMoveReward = immediateRewardVector[moveIdx];
        
        hist.push({
          state,
          moveIdx,
          player: curP,
          aiPlayerNumber,  // Track which player AI was playing as
          chosenMoveReward,  // Use this instead of rawImmediateReward
          immediateRewardVector: [...immediateRewardVector],
          move,
          opponentType
        });
        
        curP = execRes.nextPlayer;
        m++;
      }
      
      // Determine winner
      let winner = 0;
      if (grid.isGridFull()) {
        const s1 = grid.getLargestContiguous(1);
        const s2 = grid.getLargestContiguous(2);
        winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
      }
      
      if (winner === 1) p1Wins++;
      else if (winner === 2) p2Wins++;
      else draws++;
      
      winsByOpponent[opponentType][winner === 0 ? 'draws' : (winner === 1 ? 1 : 2)]++;
      if (aiPlayerNumber === 1) {
        winsByOpponent[opponentType].aiAsP1Games = (winsByOpponent[opponentType].aiAsP1Games || 0) + 1;
        if (winner === 1) { winsByOpponent[opponentType].aiWins = (winsByOpponent[opponentType].aiWins || 0) + 1; winsByOpponent[opponentType].aiAsP1Wins = (winsByOpponent[opponentType].aiAsP1Wins || 0) + 1; }
        else if (winner === 2) { winsByOpponent[opponentType].aiLosses = (winsByOpponent[opponentType].aiLosses || 0) + 1; }
        else { winsByOpponent[opponentType].aiDraws = (winsByOpponent[opponentType].aiDraws || 0) + 1; }
      } else {
        winsByOpponent[opponentType].aiAsP2Games = (winsByOpponent[opponentType].aiAsP2Games || 0) + 1;
        if (winner === 2) { winsByOpponent[opponentType].aiWins = (winsByOpponent[opponentType].aiWins || 0) + 1; winsByOpponent[opponentType].aiAsP2Wins = (winsByOpponent[opponentType].aiAsP2Wins || 0) + 1; }
        else if (winner === 1) { winsByOpponent[opponentType].aiLosses = (winsByOpponent[opponentType].aiLosses || 0) + 1; }
        else { winsByOpponent[opponentType].aiDraws = (winsByOpponent[opponentType].aiDraws || 0) + 1; }
      }
      scoresByOpponent[opponentType].p1.push(grid.getLargestContiguous(1));
      scoresByOpponent[opponentType].p2.push(grid.getLargestContiguous(2));
      // Also capture scores from AI's perspective (for role-alternation experiments)
      const aiFinalScore = grid.getLargestContiguous(aiPlayerNumber);
      const oppPlayerNumber = aiPlayerNumber === 1 ? 2 : 1;
      const oppFinalScore = grid.getLargestContiguous(oppPlayerNumber);
      scoresByOpponent[opponentType].ai.push(aiFinalScore);
      scoresByOpponent[opponentType].opponent.push(oppFinalScore);
      
      // Track final board state for diversity
      finalBoards.push(grid.grid.flat().map(cell => ({
        t1: cell.triangles[0]?.player ?? 0,
        t2: cell.triangles[1]?.player ?? 0
      })));
      
      // Audit game for bugs
      if (auditor.shouldAudit() && hist.length > 0) {
        auditor.auditSelfPlayGame(hist, grid, opponentType);
      }
      
      // Build training data (reverse iteration)
      // CRITICAL: Only train on AI's moves, not opponent's!
      // Training on opponent moves with negative reward teaches AI to lose.
      //
      // GRADIENT ISOLATION: We only push the chosen move index per sample.
      // The trainBatchWithIsolation method will set unchosen move targets
      // to the current prediction (zero gradient), matching brain.js
      // online training behavior.
      const totalMoves = hist.length;
      for (let t = totalMoves - 1; t >= 0; t--) {
        const step = hist[t];
        
        // Skip opponent moves - only train on AI's decisions
        if (step.player !== aiPlayerNumber) continue;
        
        const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
        
        // Start with immediate reward vector
        const finalOutputVector = [...step.immediateRewardVector];
        
        // Combined reward for chosen move: immediate + final outcome
        // NO discount factor — train.js gives full win/loss credit to ALL moves
        const combinedReward = step.chosenMoveReward + rewardForPlayer * finalRewardWeight;
        let scaledCombined = combinedRewardRange !== 0
          ? (combinedReward - minCombinedReward) / combinedRewardRange
          : 0.5;
        scaledCombined = Math.max(0.0, Math.min(1.0, scaledCombined));
        
        // Override chosen move with combined signal
        finalOutputVector[step.moveIdx] = scaledCombined;
        
        allInputs.push(step.state);
        allOutputs.push(finalOutputVector);
        allChosenIndices.push(step.moveIdx);
        allValueTargets.push(rewardForPlayer); // Value head target: game outcome from AI's perspective
      }
      
      if ((g + 1) % 10 === 0) process.stdout.write(`${g + 1}/${games} games done\r`);
    }
    
    // MCTS self-play (when enabled)
    if (mctsSims > 0) {
      console.log(`🌳 MCTS self-play with ${mctsSims} simulations/move...`);
      const mctsStart = Date.now();
      
      for (let g = 0; g < games; g++) {
        grid.reset();
        let curP = 1;
        let m = 0;
        const maxM = grid.rows * grid.cols * 3;
        const hist = [];
        
        // AI alternates which player it plays as
        const aiPlayerNumber = (g % 2 === 0) ? 1 : 2;
        let opponentType = 'network';
        if (opponentMix === 'random') {
          opponentType = 'random';
        } else if (opponentMix === 'heuristic') {
          opponentType = 'heuristic';
        } else if (opponentMix === 'mixed') {
          const total = networkRatio + randomRatio + heuristicRatio;
          const r = Math.random() * total;
          if (r < networkRatio) {
            opponentType = 'network';
          } else if (r < networkRatio + randomRatio) {
            opponentType = 'random';
          } else {
            opponentType = 'heuristic';
          }
        }
        
        // Initialize tracking for this opponent type (Fix for observation #5)
        opponentTypeCounts[opponentType] = (opponentTypeCounts[opponentType] || 0) + 1;
        if (!winsByOpponent[opponentType]) {
          winsByOpponent[opponentType] = { 1: 0, 2: 0, draws: 0, aiWins: 0, aiLosses: 0, aiDraws: 0, aiAsP1Wins: 0, aiAsP1Games: 0, aiAsP2Wins: 0, aiAsP2Games: 0 };
          scoresByOpponent[opponentType] = { p1: [], p2: [], ai: [], opponent: [] };
        }
        
        while (!grid.isGridFull() && m < maxM) {
          grid.currentPlayer = curP;
          
          // State encoded from current player's perspective (AlphaGo Zero style)
          // Network learns: "my pieces = +1, opponent = -1"
          const state = getStateCanonical(grid, curP);
          const isAI = curP === aiPlayerNumber;
          
          let move, mctsPolicy;
          
          if (isAI) {
            // AI's turn - use MCTS
            const result = await getMCTSMove(network, grid, mctsSims, aiPlayerNumber, mctsBatchSize);
            move = result.move;
            mctsPolicy = result.policy;
          } else {
            // Opponent's turn
            if (opponentType === 'network') {
              const result = await getMCTSMove(network, grid, mctsSims, curP, mctsBatchSize);
              move = result.move;
              mctsPolicy = result.policy;
            } else if (opponentType === 'heuristic') {
              const heuristicOpp = new HeuristicPlayer(curP, { explorationFactor: 0 });
              move = heuristicOpp.getMove(grid);
              mctsPolicy = null;
            } else {
              move = getRandomMove(grid);
              mctsPolicy = null;
            }
          }
          
          if (!move) break;
          
          // Track first move
          if (m === 0) {
            firstMoves.push(`${move.x},${move.y},r${move.rotation}`);
          }
          
          // Store MCTS result for training
          const moveIdx = (move.y * grid.cols + move.x) * 4 + move.rotation;
          hist.push({
            state,
            moveIdx,
            player: curP,
            aiPlayerNumber,
            mctsPolicy,  // Store MCTS policy directly
            move
          });
          
          // Make move using consolidated executeMove
          const execRes = grid.executeMove(move.x, move.y, move.rotation, curP);
          if (!execRes.success) break;
          curP = execRes.nextPlayer;
          
          m++;
        }
        
        // Determine winner
        let winner = 0;
        if (grid.isGridFull()) {
          const s1 = grid.getLargestContiguous(1);
          const s2 = grid.getLargestContiguous(2);
          winner = s1 > s2 ? 1 : (s2 > s1 ? 2 : 0);
        }
        
        if (winner === 1) p1Wins++;
        else if (winner === 2) p2Wins++;
        else draws++;
        
        // Track wins by opponent type (Fix for observation #5)
        winsByOpponent[opponentType][winner === 0 ? 'draws' : winner]++;
        if (aiPlayerNumber === 1) {
          winsByOpponent[opponentType].aiAsP1Games = (winsByOpponent[opponentType].aiAsP1Games || 0) + 1;
          if (winner === 1) { winsByOpponent[opponentType].aiWins = (winsByOpponent[opponentType].aiWins || 0) + 1; winsByOpponent[opponentType].aiAsP1Wins = (winsByOpponent[opponentType].aiAsP1Wins || 0) + 1; }
          else if (winner === 2) { winsByOpponent[opponentType].aiLosses = (winsByOpponent[opponentType].aiLosses || 0) + 1; }
          else { winsByOpponent[opponentType].aiDraws = (winsByOpponent[opponentType].aiDraws || 0) + 1; }
        } else {
          winsByOpponent[opponentType].aiAsP2Games = (winsByOpponent[opponentType].aiAsP2Games || 0) + 1;
          if (winner === 2) { winsByOpponent[opponentType].aiWins = (winsByOpponent[opponentType].aiWins || 0) + 1; winsByOpponent[opponentType].aiAsP2Wins = (winsByOpponent[opponentType].aiAsP2Wins || 0) + 1; }
          else if (winner === 1) { winsByOpponent[opponentType].aiLosses = (winsByOpponent[opponentType].aiLosses || 0) + 1; }
          else { winsByOpponent[opponentType].aiDraws = (winsByOpponent[opponentType].aiDraws || 0) + 1; }
        }
        scoresByOpponent[opponentType].p1.push(grid.getLargestContiguous(1));
        scoresByOpponent[opponentType].p2.push(grid.getLargestContiguous(2));
        
        // Track scores from AI's perspective (Bug #4 fix)
        const aiFinalScore = grid.getLargestContiguous(aiPlayerNumber);
        const oppPlayerNumber = aiPlayerNumber === 1 ? 2 : 1;
        const oppFinalScore = grid.getLargestContiguous(oppPlayerNumber);
        scoresByOpponent[opponentType].ai.push(aiFinalScore);
        scoresByOpponent[opponentType].opponent.push(oppFinalScore);
        
        // Track final board state for diversity (Bug #3 fix)
        finalBoards.push(grid.grid.flat().map(cell => ({
          t1: cell.triangles[0]?.player ?? 0,
          t2: cell.triangles[1]?.player ?? 0
        })));
        
        // Process MCTS training data
        for (const step of hist) {
          // Only train on AI's moves to avoid dual-perspective conflict
          // (Bug #1 fix: see docs/TF-B8-V2_POSTMORTEM.md for analysis)
          if (step.player !== aiPlayerNumber || !step.mctsPolicy) continue;
          
          // Value target from the perspective of the player who made the move
          // (matches state encoding which is from mover's perspective)
          const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
          
          // For MCTS: policy target is the visit distribution, value target is outcome
          allInputs.push(step.state);
          allOutputs.push(step.mctsPolicy);  // MCTS policy as target
          allChosenIndices.push(step.moveIdx);
          allValueTargets.push(rewardForPlayer);  // Value from mover's perspective
        }
        
        if ((g + 1) % 10 === 0) process.stdout.write(`${g + 1}/${games} MCTS games done\r`);
      }
      
      console.log(`✅ MCTS self-play: ${games} games, ${allInputs.length} samples, ${Date.now() - mctsStart}ms`);
      console.log(`   Wins: P1=${p1Wins} P2=${p2Wins} Draws=${draws}`);
    }
    
    console.log(`✅ Self-play: ${games} games, ${allInputs.length} samples, ${Date.now() - cycleStart}ms`);
    console.log(`   Wins: P1=${p1Wins} P2=${p2Wins} Draws=${draws}`);

    // Add new samples to replay buffer
    if (useReplayBuffer && allInputs.length > 0) {
      replayBuffer.add(allInputs, allOutputs, allChosenIndices, allValueTargets);
      const bufStats = replayBuffer.getStats();
      console.log(`🗄️  Buffer: ${bufStats.size.toLocaleString()}/${bufStats.maxSize.toLocaleString()} samples (${(bufStats.utilization * 100).toFixed(1)}% full)`);
    }

    // Training — dispatch based on architecture
    const trainStart = Date.now();
    let trainResult;
    let trainingSamplesUsed = allInputs.length;
    const halfIters = Math.max(1, Math.floor(iterations / 2));

    /**
     * Train function that handles both dense and conv architectures
     * For conv: uses trainBatchWithIsolation with value targets
     * For dense: uses existing train methods
     */
    async function trainOnData(inputs, outputs, chosenIndices, valueTargets, iters) {
      if (useConv) {
        const isolateChosenOnly = mctsSims === 0 ? useGradientIsolation : false;
        return await network.trainBatchWithIsolation(
          inputs, outputs, chosenIndices,
          valueTargets || new Array(inputs.length).fill(0),
          iters, learningRate, batchSize, isolateChosenOnly
        );
      } else if (useGradientIsolation) {
        return await network.trainBatchWithIsolation(inputs, outputs, chosenIndices, iters, learningRate, batchSize);
      } else {
        return await network.trainBatch(inputs, outputs, iters, learningRate, batchSize);
      }
    }

    if (useReplayBuffer && replayBuffer.size >= replayBatchSize) {
      // Train on a mix of new data + replayed historical data
      const newBatchCount = Math.ceil(allInputs.length / replayBatchSize);
      const totalBatches = Math.max(1, Math.round(newBatchCount * replayRatio));

      // First: train on current cycle data (on-policy)
      if (allInputs.length > 0) {
        trainResult = await trainOnData(allInputs, allOutputs, allChosenIndices, allValueTargets, halfIters);
      }

      // Then: train on replay buffer samples (off-policy)
      let replayLossTotal = 0;
      let replayBatchCount = 0;

      for (let rb = 0; rb < totalBatches; rb++) {
        const batch = replayBuffer.sample(replayBatchSize);
        const replayValueTargets = batch.valueTargets || new Array(replayBatchSize).fill(0);
        const replayResult = await trainOnData(batch.inputs, batch.outputs, batch.chosenIndices, replayValueTargets, halfIters);
        
        replayLossTotal += replayResult.finalError;
        replayBatchCount++;
        trainingSamplesUsed += replayBatchSize;

        // Update priorities based on TD-error
        if (batch.indices) {
          const tdErrors = batch.inputs.map((inp, i) => {
            if (useConv) {
              const pred = network.run(inp);
              return Math.abs(pred.policy[batch.chosenIndices[i]] - batch.outputs[i][batch.chosenIndices[i]]);
            } else {
              const pred = network.run(inp);
              return Math.abs(pred[batch.chosenIndices[i]] - batch.outputs[i][batch.chosenIndices[i]]);
            }
          });
          replayBuffer.updatePriorities(batch.indices, tdErrors);
        }
      }

      // Merge training results for logging
      if (!trainResult) {
        trainResult = {
          initialError: replayLossTotal / replayBatchCount,
          finalError: replayLossTotal / replayBatchCount,
          errorHistory: [replayLossTotal / replayBatchCount],
          iterations: halfIters * replayBatchCount,
          timeMs: Date.now() - trainStart
        };
      } else {
        // Weighted average: preserve cycle training error, not replay buffer error (Bug #2 fix)
        const cycleSamples = allInputs.length;
        const replaySamples = trainingSamplesUsed - cycleSamples;
        const totalSamples = trainingSamplesUsed;
        const weightedFinalError = (trainResult.finalError * cycleSamples + (replayLossTotal / replayBatchCount) * replaySamples) / totalSamples;
        
        trainResult.errorHistory.push(replayLossTotal / replayBatchCount);
        trainResult.finalError = weightedFinalError;  // Weighted average instead of overwrite
        trainResult.timeMs = Date.now() - trainStart;
      }
      console.log(`📚 Training (new+replay): ${trainResult.initialError.toFixed(6)} → ${trainResult.finalError.toFixed(6)} (${iterations} epochs, ${replayBatchCount} replay batches, ${trainResult.timeMs}ms)`);
      console.log(`   Samples used: ${trainingSamplesUsed.toLocaleString()} (new: ${allInputs.length}, replay: ${trainingSamplesUsed - allInputs.length})`);
    } else {
      // No replay buffer (or buffer too small) — standard training on current cycle
      trainResult = await trainOnData(allInputs, allOutputs, allChosenIndices, allValueTargets, iterations);
      console.log(`📚 Training: ${trainResult.initialError.toFixed(6)} → ${trainResult.finalError.toFixed(6)} (${iterations} epochs, ${trainResult.timeMs}ms)`);
    }
    
    // Evaluation vs Random
    let evalWins = 0, evalLosses = 0, evalDraws = 0;
    let evalP1Wins = 0, evalP1Losses = 0, evalP1Draws = 0, evalP1Games = 0;
    let evalP2Wins = 0, evalP2Losses = 0, evalP2Draws = 0, evalP2Games = 0;
    let evalAIScores = [], evalRandScores = [];
    
    for (let g = 0; g < evalGames; g++) {
      grid.reset();
      let cur = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      // FIXED PERSPECTIVE: AI alternates which player it plays as in eval too
      const evalAIPlayerNumber = (g % 2 === 0) ? 1 : 2;
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = cur;
        const move = cur === evalAIPlayerNumber ? getAIMove(network, grid, 0, evalAIPlayerNumber, useConv) : getRandomMove(grid);
        if (!move) break;
        
        const exec = grid.executeMove(move.x, move.y, move.rotation, cur);
        if (!exec.success) break;
        cur = exec.nextPlayer;
        m++;
      }
      
      const s1 = grid.getLargestContiguous(1);
      const s2 = grid.getLargestContiguous(2);
      // Score from AI's perspective
      if (evalAIPlayerNumber === 1) {
        evalP1Games++;
        if (s1 > s2) { evalWins++; evalP1Wins++; }
        else if (s2 > s1) { evalLosses++; evalP1Losses++; }
        else { evalDraws++; evalP1Draws++; }
        evalAIScores.push(s1);
        evalRandScores.push(s2);
      } else {
        evalP2Games++;
        if (s2 > s1) { evalWins++; evalP2Wins++; }
        else if (s1 > s2) { evalLosses++; evalP2Losses++; }
        else { evalDraws++; evalP2Draws++; }
        evalAIScores.push(s2);
        evalRandScores.push(s1);
      }
    }
    
    const evalRate = evalWins / evalGames * 100;
    const evalP1Rate = evalP1Games > 0 ? (evalP1Wins / evalP1Games * 100) : 0;
    const evalP2Rate = evalP2Games > 0 ? (evalP2Wins / evalP2Games * 100) : 0;
    console.log(`📊 Eval vs Random: ${evalRate.toFixed(1)}% (${evalWins}W / ${evalLosses}L / ${evalDraws}D) [as P1: ${evalP1Rate.toFixed(1)}%, as P2: ${evalP2Rate.toFixed(1)}%]`);
    
    // Evaluation vs Heuristic
    let heurWins = 0, heurLosses = 0, heurDraws = 0;
    let heurP1Wins = 0, heurP1Losses = 0, heurP1Draws = 0, heurP1Games = 0;
    let heurP2Wins = 0, heurP2Losses = 0, heurP2Draws = 0, heurP2Games = 0;
    let heurAIScores = [], heurOppScores = [];
    
    for (let g = 0; g < evalGames; g++) {
      grid.reset();
      let cur = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      // FIXED PERSPECTIVE: AI alternates which player it plays as
      const heurAIPlayerNumber = (g % 2 === 0) ? 1 : 2;
      const heuristicOpponentNumber = heurAIPlayerNumber === 1 ? 2 : 1;
      const heuristicEval = new HeuristicPlayer(heuristicOpponentNumber, { explorationFactor: 0 });
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = cur;
        const move = cur === heurAIPlayerNumber 
          ? getAIMove(network, grid, 0, heurAIPlayerNumber, useConv) 
          : heuristicEval.getMove(grid);
        if (!move) break;
        
        const exec = grid.executeMove(move.x, move.y, move.rotation, cur);
        if (!exec.success) break;
        cur = exec.nextPlayer;
        m++;
      }
      
      const s1 = grid.getLargestContiguous(1);
      const s2 = grid.getLargestContiguous(2);
      // Score from AI's perspective
      if (heurAIPlayerNumber === 1) {
        heurP1Games++;
        if (s1 > s2) { heurWins++; heurP1Wins++; }
        else if (s2 > s1) { heurLosses++; heurP1Losses++; }
        else { heurDraws++; heurP1Draws++; }
        heurAIScores.push(s1);
        heurOppScores.push(s2);
      } else {
        heurP2Games++;
        if (s2 > s1) { heurWins++; heurP2Wins++; }
        else if (s1 > s2) { heurLosses++; heurP2Losses++; }
        else { heurDraws++; heurP2Draws++; }
        heurAIScores.push(s2);
        heurOppScores.push(s1);
      }
    }
    
    const heurRate = heurWins / evalGames * 100;
    const heurP1Rate = heurP1Games > 0 ? (heurP1Wins / heurP1Games * 100) : 0;
    const heurP2Rate = heurP2Games > 0 ? (heurP2Wins / heurP2Games * 100) : 0;
    console.log(`📊 Eval vs Heuristic: ${heurRate.toFixed(1)}% (${heurWins}W / ${heurLosses}L / ${heurDraws}D) [as P1: ${heurP1Rate.toFixed(1)}%, as P2: ${heurP2Rate.toFixed(1)}%]`);
    
    // Compute diversity metrics
    const openingDiversity = computeOpeningDiversity(firstMoves);
    const boardDiversity = computeBoardDiversity(finalBoards);
    console.log(`🎨 Diversity: Open=${openingDiversity.toFixed(2)} Board=${boardDiversity.toFixed(2)}`);
    
    // Log cycle
    log.cycles.push({
      cycle,
      selfPlay: {
        games,
        wins: { 1: p1Wins, 2: p2Wins, draws },
        winsByOpponent,
        opponentTypeCounts,
        scoresByOpponent: Object.fromEntries(
          Object.entries(scoresByOpponent).map(([k, v]) => [
            k,
            {
              p1: { median: computeMedian(v.p1), min: Math.min(...v.p1), max: Math.max(...v.p1) },
              p2: { median: computeMedian(v.p2), min: Math.min(...v.p2), max: Math.max(...v.p2) },
              // AI perspective scores (for role-alternation experiments)
              ai: v.ai && v.ai.length > 0 ? { median: computeMedian(v.ai), min: Math.min(...v.ai), max: Math.max(...v.ai) } : null,
              opponent: v.opponent && v.opponent.length > 0 ? { median: computeMedian(v.opponent), min: Math.min(...v.opponent), max: Math.max(...v.opponent) } : null
            }
          ])
        ),
        openingDiversity,
        boardDiversity
      },
      training: {
        samples: allInputs.length,
        totalSamplesUsed: trainingSamplesUsed || allInputs.length,
        iterations,
        learningRate,
        initialError: trainResult.initialError,
        finalError: trainResult.finalError,
        errorSnapshots: trainResult.errorHistory.map((err, i) => ({ iteration: i, error: err })),
        timeMs: trainResult.timeMs,
        replayBuffer: useReplayBuffer ? replayBuffer.getStats() : null
      },
      evaluation: {
        vsRandom: {
          aiWinRate: evalRate,
          aiWins: evalWins,
          losses: evalLosses,
          draws: evalDraws,
          p1WinRate: evalP1Rate,
          p2WinRate: evalP2Rate,
          p1Wins: evalP1Wins,
          p1Losses: evalP1Losses,
          p1Draws: evalP1Draws,
          p1Games: evalP1Games,
          p2Wins: evalP2Wins,
          p2Losses: evalP2Losses,
          p2Draws: evalP2Draws,
          p2Games: evalP2Games,
          aiScores: evalAIScores,
          opponentScores: evalRandScores
        },
        vsHeuristic: {
          aiWinRate: heurRate,
          aiWins: heurWins,
          losses: heurLosses,
          draws: heurDraws,
          p1WinRate: heurP1Rate,
          p2WinRate: heurP2Rate,
          p1Wins: heurP1Wins,
          p1Losses: heurP1Losses,
          p1Draws: heurP1Draws,
          p1Games: heurP1Games,
          p2Wins: heurP2Wins,
          p2Losses: heurP2Losses,
          p2Draws: heurP2Draws,
          p2Games: heurP2Games,
          aiScores: heurAIScores,
          opponentScores: heurOppScores
        }
      }
    });
    
    // Save after every cycle for live dashboard tracking
    {
      const outputPath = path.resolve(output);
      const outputDir = path.dirname(outputPath);
      if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
      
      const networkPath = outputPath.replace('.json', '_network.json');
      fs.writeFileSync(networkPath, JSON.stringify(network.toJSON(), null, 2));
      
      log.endedAt = new Date().toISOString();
      log.finalResult = { 
        vsRandom: evalRate, 
        vsHeuristic: heurRate,
        vsRandomP1: evalP1Rate,
        vsRandomP2: evalP2Rate,
        vsHeuristicP1: heurP1Rate,
        vsHeuristicP2: heurP2Rate
      };
      fs.writeFileSync(outputPath, JSON.stringify(log, null, 2));
      
      console.log(`💾 Saved: ${path.basename(networkPath)}, ${path.basename(outputPath)}`);
    }
  }
  
  // Print audit summary if enabled
  if (audit) {
    auditor.printSummary();
  }

  console.log(`\n✅ Training complete in ${((Date.now() - new Date(log.startedAt).getTime()) / 1000 / 60).toFixed(1)} min`);
  const lastCycle = log.cycles[log.cycles.length - 1];
  console.log(`   Final vs Random: ${lastCycle.evaluation.vsRandom.aiWinRate.toFixed(1)}%`);
  console.log(`   Final vs Heuristic: ${lastCycle.evaluation.vsHeuristic.aiWinRate.toFixed(1)}%`);
}

// Wait for GridManager to load, then run main
async function waitForGridManager() {
  await importPromise;
}

// Only run main if this is the entry point (not when imported)
if (require.main === module) {
  waitForGridManager().then(() => main()).catch(e => {
    console.error('Training failed:', e);
    process.exit(1);
  });
}

// Export for MCTS and testing
module.exports = { TFConvNetwork, TFNeuralNetwork };
