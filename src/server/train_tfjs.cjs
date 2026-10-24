#!/usr/bin/env node
/**
 * TF.js Training Pipeline for Triantica AI
 * 
 * Supports:
 * - Pure network self-play
 * - Mixed opponents (network + random)
 * - Cluster reward with TD(λ)
 * - Full dashboard metrics output
 * - Game auditing for bug detection
 * 
 * Usage: node src/server/train_tfjs.cjs --cycles 10 --games 100 --output data/TF-A1_training.json
 * 
 * Requires Node v20 LTS for tfjs-node native bindings (Metal GPU on M2 Ultra)
 */

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Dynamic imports for ES modules
let GridManager, GameAuditor, HeuristicPlayer;
(async () => {
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
      optimizer: tf.train.sgd(0.1),
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

  async trainBatch(inputs, outputs, iterations = 30, learningRate = 0.1) {
    const xs = tf.tensor2d(inputs, [inputs.length, inputs[0].length]);
    const ys = tf.tensor2d(outputs, [outputs.length, outputs[0].length]);
    
    this.model.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: 'meanSquaredError'
    });
    
    const start = Date.now();
    const history = await this.model.fit(xs, ys, {
      epochs: iterations,
      batchSize: Math.min(256, inputs.length),
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
// Helper Functions
// ============================================================

function getStateRelative(grid) {
  const state = [];
  const { rows, cols } = grid;
  const curPlayer = grid.currentPlayer;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cell = grid.grid[r][c];
      const t1 = cell.triangles[0], t2 = cell.triangles[1];
      const pv = t => t ? (t.player === curPlayer ? 1 : -1) : 0;
      const rv = r => r === undefined || r === -1 ? -1 : r / 4;
      state.push(pv(t1), rv(t1?.rotation ?? -1), pv(t2), rv(t2?.rotation ?? -1));
    }
  }
  state.push(1);
  return state;
}

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

function getAIMove(network, grid, exploration) {
  if (Math.random() < exploration) return getRandomMove(grid);
  
  const state = getStateRelative(grid);
  const output = network.run(state);
  const moves = [];
  const { rows, cols } = grid;
  
  for (let i = 0; i < output.length; i++) {
    const rot = i % 4;
    const idx = Math.floor(i / 4);
    const x = idx % cols;
    const y = Math.floor(idx / cols);
    moves.push({ x, y, rotation: rot, score: output[i] });
  }
  
  moves.sort((a, b) => b.score - a.score);
  
  for (const m of moves) {
    if (isValidMove(grid, m.x, m.y, m.rotation)) return m;
  }
  
  return getRandomMove(grid);
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
          
          outputVector[moveIndex] = Math.max(0.01, Math.min(0.99, scaledReward));
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
  const iterations = parseInt(getArg('iterations', '30'));
  const learningRate = parseFloat(getArg('learning-rate', '0.1'));
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
  console.log(`  Exploration:      ${exploration}`);
  console.log(`  Cluster reward:   ${enableClusterReward}`);
  console.log(`  Audit mode:       ${audit ? 'ENABLED' + (audit ? ' (' + (auditRate * 100).toFixed(0) + '% sample rate)' : '') : 'disabled'}`);
  console.log(`  Start from:      ${startFrom || 'fresh'}`);
  console.log(`  Output:          ${output}`);
  console.log('='.repeat(60));
  
  const grid = new GridManager(10, 10);
  const network = new TFNeuralNetwork({
    inputSize: 401,
    hiddenLayers: [200],
    outputSize: 400,
    activation: 'sigmoid'
  });
  
  if (startFrom) {
    const data = JSON.parse(fs.readFileSync(startFrom, 'utf-8'));
    network.fromJSON(data);
    console.log(`📂 Loaded network from ${startFrom}`);
  }
  
  // Sanity check
  const testState = new Array(401).fill(0);
  const testOut = network.run(testState);
  console.log(`🧠 Network ready (output range: ${Math.min(...testOut).toFixed(3)} - ${Math.max(...testOut).toFixed(3)})`);
  
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
    const allInputs = [], allOutputs = [];
    const winsByOpponent = {};
    const opponentTypeCounts = {};
    const scoresByOpponent = {};
    const firstMoves = [];  // Track opening moves for diversity
    const finalBoards = [];  // Track final board states for diversity
    const heuristicP2 = new HeuristicPlayer(2, { explorationFactor: 0 });
    
    let p1Wins = 0, p2Wins = 0, draws = 0;
    
    for (let g = 0; g < games; g++) {
      grid.reset();
      let curP = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      const hist = [];
      
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
        winsByOpponent[opponentType] = { 1: 0, 2: 0, draws: 0 };
        scoresByOpponent[opponentType] = { p1: [], p2: [] };
      }
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = curP;
        
        const state = getStateRelative(grid);
        const immediateRewardVector = calculateImmediateRewardVector(grid, curP);
        // Don't calculate rawImmediateReward here - we'll use chosenMoveReward after move selection
        
        // Determine whose perspective we're playing from
        const isNetworkP = curP === 1 || (selfPlayMode === 'network' && curP === 2 && opponentType === 'network');
        
        let move;
        if (curP === 1) {
          move = getAIMove(network, grid, exploration);
        } else if (opponentType === 'network') {
          move = getAIMove(network, grid, exploration);
        } else if (opponentType === 'heuristic') {
          move = heuristicP2.getMove(grid);
        } else {
          move = getRandomMove(grid);
        }
        
        if (!move) break;
        
        // Track first move for opening diversity
        if (m === 0) {
          firstMoves.push(`${move.x},${move.y},r${move.rotation}`);
        }
        
        grid.selectedRotation = move.rotation;
        if (!grid.placeTriangle(move.x, move.y)) break;
        
        const claimed = grid.claimEnclosedTriangles();
        const moveIdx = (move.y * grid.cols + move.x) * 4 + move.rotation;
        
        // Store the immediate reward value at the CHOSEN move index
        // CRITICAL: Use vector[moveIdx], NOT average of all values
        const chosenMoveReward = immediateRewardVector[moveIdx];
        
        hist.push({
          state,
          moveIdx,
          player: curP,
          chosenMoveReward,  // Use this instead of rawImmediateReward
          immediateRewardVector: [...immediateRewardVector],
          move,
          opponentType
        });
        
        // Extra turn logic
        grid.extraTurns += Math.floor((1 + claimed) / 2);
        if (grid.extraTurns > 0) grid.extraTurns--;
        else curP = curP === 1 ? 2 : 1;
        
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
      scoresByOpponent[opponentType].p1.push(grid.getLargestContiguous(1));
      scoresByOpponent[opponentType].p2.push(grid.getLargestContiguous(2));
      
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
      const totalMoves = hist.length;
      for (let t = totalMoves - 1; t >= 0; t--) {
        const step = hist[t];
        const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
        
        // Start with immediate reward vector
        const finalOutputVector = [...step.immediateRewardVector];
        
        // Combined reward for chosen move: immediate + discounted final reward
        // Linear discount: later moves get stronger final signal
        const discountFactor = (t + 1) / totalMoves;
        const combinedReward = step.chosenMoveReward + rewardForPlayer * finalRewardWeight * discountFactor;
        let scaledCombined = combinedRewardRange !== 0
          ? (combinedReward - minCombinedReward) / combinedRewardRange
          : 0.5;
        scaledCombined = Math.max(0.01, Math.min(0.99, scaledCombined));
        
        // Override chosen move with combined signal
        finalOutputVector[step.moveIdx] = scaledCombined;
        
        allInputs.push(step.state);
        allOutputs.push(finalOutputVector);
      }
      
      if ((g + 1) % 10 === 0) process.stdout.write(`${g + 1}/${games} games done\r`);
    }
    
    console.log(`✅ Self-play: ${games} games, ${allInputs.length} samples, ${Date.now() - cycleStart}ms`);
    console.log(`   Wins: P1=${p1Wins} P2=${p2Wins} Draws=${draws}`);
    
    // Training
    const trainStart = Date.now();
    const trainResult = await network.trainBatch(allInputs, allOutputs, iterations, learningRate);
    console.log(`📚 Training: ${trainResult.initialError.toFixed(6)} → ${trainResult.finalError.toFixed(6)} (${iterations} epochs, ${trainResult.timeMs}ms)`);
    
    // Evaluation vs Random
    let evalWins = 0, evalLosses = 0, evalDraws = 0;
    let evalAIScores = [], evalRandScores = [];
    
    for (let g = 0; g < evalGames; g++) {
      grid.reset();
      let cur = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = cur;
        const move = cur === 1 ? getAIMove(network, grid, 0) : getRandomMove(grid);
        if (!move) break;
        
        grid.selectedRotation = move.rotation;
        if (!grid.placeTriangle(move.x, move.y)) break;
        
        const claimed = grid.claimEnclosedTriangles();
        grid.extraTurns += Math.floor((1 + claimed) / 2);
        if (grid.extraTurns > 0) grid.extraTurns--;
        else cur = cur === 1 ? 2 : 1;
        m++;
      }
      
      const s1 = grid.getLargestContiguous(1);
      const s2 = grid.getLargestContiguous(2);
      if (s1 > s2) evalWins++; else if (s2 > s1) evalLosses++; else evalDraws++;
      evalAIScores.push(s1);
      evalRandScores.push(s2);
    }
    
    const evalRate = evalWins / evalGames * 100;
    console.log(`📊 Eval vs Random: ${evalRate.toFixed(1)}% (${evalWins}W / ${evalLosses}L / ${evalDraws}D)`);
    
    // Evaluation vs Heuristic
    let heurWins = 0, heurLosses = 0, heurDraws = 0;
    let heurAIScores = [], heurOppScores = [];
    const heuristicEvalP2 = new HeuristicPlayer(2, { explorationFactor: 0 });
    
    for (let g = 0; g < evalGames; g++) {
      grid.reset();
      let cur = 1, m = 0;
      const maxM = grid.rows * grid.cols * 3;
      
      while (!grid.isGridFull() && m < maxM) {
        grid.currentPlayer = cur;
        const move = cur === 1 ? getAIMove(network, grid, 0) : heuristicEvalP2.getMove(grid);
        if (!move) break;
        
        grid.selectedRotation = move.rotation;
        if (!grid.placeTriangle(move.x, move.y)) break;
        
        const claimed = grid.claimEnclosedTriangles();
        grid.extraTurns += Math.floor((1 + claimed) / 2);
        if (grid.extraTurns > 0) grid.extraTurns--;
        else cur = cur === 1 ? 2 : 1;
        m++;
      }
      
      const s1 = grid.getLargestContiguous(1);
      const s2 = grid.getLargestContiguous(2);
      if (s1 > s2) heurWins++; else if (s2 > s1) heurLosses++; else heurDraws++;
      heurAIScores.push(s1);
      heurOppScores.push(s2);
    }
    
    const heurRate = heurWins / evalGames * 100;
    console.log(`📊 Eval vs Heuristic: ${heurRate.toFixed(1)}% (${heurWins}W / ${heurLosses}L / ${heurDraws}D)`);
    
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
              p2: { median: computeMedian(v.p2), min: Math.min(...v.p2), max: Math.max(...v.p2) }
            }
          ])
        ),
        openingDiversity,
        boardDiversity
      },
      training: {
        samples: allInputs.length,
        iterations,
        learningRate,
        initialError: trainResult.initialError,
        finalError: trainResult.finalError,
        errorSnapshots: trainResult.errorHistory.map((err, i) => ({ iteration: i, error: err })),
        timeMs: trainResult.timeMs
      },
      evaluation: {
        vsRandom: {
          aiWinRate: evalRate,
          aiWins: evalWins,
          losses: evalLosses,
          draws: evalDraws,
          aiScores: evalAIScores,
          opponentScores: evalRandScores
        },
        vsHeuristic: {
          aiWinRate: heurRate,
          aiWins: heurWins,
          losses: heurLosses,
          draws: heurDraws,
          aiScores: heurAIScores,
          opponentScores: heurOppScores
        }
      }
    });
    
    // Save intermediate
    if (cycle % 5 === 0 || cycle === cycles) {
      const outputPath = path.resolve(output);
      const outputDir = path.dirname(outputPath);
      if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
      
      const networkPath = outputPath.replace('.json', '_network.json');
      fs.writeFileSync(networkPath, JSON.stringify(network.toJSON(), null, 2));
      
      log.endedAt = new Date().toISOString();
      log.finalResult = { 
        vsRandom: evalRate, 
        vsHeuristic: heurRate 
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
  while (!GridManager) {
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

waitForGridManager().then(() => main()).catch(e => {
  console.error('Training failed:', e);
  process.exit(1);
});