/**
 * AIPlayer - Neural network player implementing PlayerInterface
 * 
 * Refactored from original AIPlayer.js to work with the extracted
 * GridManager and PlayerInterface architecture.
 * 
 * Works in both browser (brain.js global) and Node.js (brain.js import).
 */

import { PlayerInterface, PlayerType } from './PlayerInterface.js';

// brain.js reference — resolved lazily at runtime
let brainRef = null;

async function resolveBrain() {
  if (brainRef) return brainRef;
  // Try dynamic import (ESM-safe, works in Node.js and modern browsers)
  try {
    const mod = await import('brain.js');
    // brain.js ESM export: mod.default is the main namespace, mod also has named exports
    // Need to handle: mod.default.NeuralNetwork, mod.NeuralNetwork, or mod.default.default
    const candidates = [mod.default, mod, mod.default?.default];
    for (const candidate of candidates) {
      if (candidate && typeof candidate.NeuralNetwork === 'function') {
        brainRef = candidate;
        return brainRef;
      }
    }
  } catch (e) {}
  // Try Node.js CJS via createRequire
  try {
    const { createRequire } = await import('node:module');
    const req = createRequire(import.meta.url);
    const cjsBrain = req('brain.js');
    if (cjsBrain && typeof cjsBrain.NeuralNetwork === 'function') {
      brainRef = cjsBrain;
      return brainRef;
    }
  } catch (e) {}
  // Try browser global
  try {
    if (typeof brain !== 'undefined' && typeof brain.NeuralNetwork === 'function') {
      brainRef = brain;
      return brain;
    }
  } catch (e) {}
  return null;
}

export class AIPlayer extends PlayerInterface {
  /**
   * @param {number} playerNumber - 1 or 2
   * @param {object} config
   * @param {string} config.difficulty - 'easy' | 'medium' | 'hard'
   * @param {number} config.explorationFactor - Probability of random move (default 0.2)
   * @param {number} config.immediacyFactor - Probability of immediate-reward move vs network (default 0.9)
   * @param {string} config.explorationMode - 'perMove' | 'perGame' (default 'perMove')
   * @param {boolean} config.enableClusterReward - Enable TD(λ) cluster-contribution reward (default: false)
   * @param {number} config.clusterRewardFactor - Per-triangle cluster bonus (default: 0.1)
   * @param {number} config.clusterLambda - Eligibility trace decay (default: 0.9)
   * @param {number} config.clusterGamma - Discount factor (default: 0.99)
   * @param {object} config.networkJSON - Pre-trained network state (optional)
   * @param {Function} config.brainFactory - () => brain.js reference (for testing/DI)
   */
  constructor(playerNumber, config = {}) {
    super(playerNumber, config);
    this.type = PlayerType.AI;
    this.difficulty = config.difficulty || 'medium';
    this.explorationFactor = config.explorationFactor ?? 0.2;
    this.immediacyFactor = config.immediacyFactor ?? 0.9;
    this.explorationMode = config.explorationMode || 'perMove'; // 'perMove' | 'perGame'
    this.selfPlayMode = config.selfPlayMode || 'random'; // 'random' | 'network'
    this.gamePlayMode = 'network'; // 'random' | 'network' — set per game when explorationMode='perGame'
    this.isNetworkLoaded = false;
    this.trainingData = [];
    this._networkJSON = config.networkJSON || null;
    this._brainFactory = config.brainFactory || null;
    // TD(λ) cluster reward configuration
    this.enableClusterReward = config.enableClusterReward ?? false;
    this.clusterRewardFactor = config.clusterRewardFactor ?? 0.1;
    this.clusterLambda = config.clusterLambda ?? 0.9;
    this.clusterGamma = config.clusterGamma ?? 0.99;
  }

  /**
   * Get brain.js reference — lazy init
   */
  async _getBrain() {
    if (this._brainFactory) return this._brainFactory();
    return resolveBrain();
  }

  /**
   * Initialize or load network
   */
  async initialize() {
    super.initialize();
    const brain = await this._getBrain();
    
    if (!brain && !this._networkJSON) {
      console.warn('AIPlayer: brain.js not available, will play random moves');
      this.network = null;
      return;
    }

    if (brain) {
      this._createNetwork(brain);
    }

    // Load pre-trained weights
    if (this._networkJSON && this.network) {
      this.network.fromJSON(this._networkJSON);
      this.isNetworkLoaded = true;
    }
  }

  _createNetwork(brain) {
    // Default grid dimensions
    const rows = 10, cols = 10;
    const inputSize = rows * cols * 4 + 1; // 401
    const outputSize = rows * cols * 4;     // 400

    // Debug: verify brain object
    if (!brain || typeof brain.NeuralNetwork !== 'function') {
      console.error('AIPlayer: Invalid brain.js reference:', typeof brain, typeof brain?.NeuralNetwork);
      this.network = null;
      return;
    }

    let hiddenLayers;
    switch (this.difficulty) {
      case 'easy':
        hiddenLayers = [Math.floor(inputSize / 4)];
        break;
      case 'hard':
        hiddenLayers = [Math.floor(inputSize / 1.5), Math.floor(inputSize / 2.5)];
        break;
      case 'medium':
      default:
        hiddenLayers = [Math.floor(inputSize / 2)]; // [200]
        break;
    }

    this.network = new brain.NeuralNetwork({
      inputSize,
      hiddenLayers,
      outputSize,
      activation: 'sigmoid',
    });

    // Initialize weights with random data (brain.js needs this before run())
    const dummyInput = new Array(inputSize).fill(0).map(() => Math.random());
    const dummyOutput = new Array(outputSize).fill(0.5);
    this.network.train([{ input: dummyInput, output: dummyOutput }], {
      iterations: 1,
      errorThresh: 1.0,
    });

    this.isNetworkLoaded = false;
  }

  /**
   * Set play mode for this game (used in 'perGame' exploration mode).
   * Call at the start of each game.
   * @param {'random' | 'network'} mode
   */
  setGamePlayMode(mode) {
    this.gamePlayMode = mode;
  }

  /**
   * Determine play mode for a new game (per-game exploration).
   * @returns {'random' | 'network'}
   */
  _determineGamePlayMode() {
    return Math.random() < this.explorationFactor ? 'random' : 'network';
  }

  /**
   * Get move from AI — implements PlayerInterface
   * @param {GridManager} gridManager
   * @returns {Promise<{x, y, rotation}>}
   */
  async getMove(gridManager) {
    if (!this.network || !this.isNetworkLoaded) {
      // Fallback to random if no network
      return this._getRandomValidMove(gridManager);
    }

    // Per-game exploration: mode is determined once at game start
    if (this.explorationMode === 'perGame') {
      if (this.gamePlayMode === 'random') {
        return this._getRandomValidMove(gridManager);
      }
      // Network mode — use network prediction
      return this._getNetworkMove(gridManager);
    }

    // Per-move exploration: roll dice each turn (original behavior)
    if (Math.random() < this.explorationFactor) {
      return this._getRandomValidMove(gridManager);
    }

    return this._getNetworkMove(gridManager);
  }

  /**
   * Get move using network prediction (exploit)
   */
  async _getNetworkMove(gridManager) {
    // Always use relative state for consistency with training
    const state = this._getStateRelativeToCurrentPlayer(gridManager);
    const output = this.network.run(state);

    // Map output to moves, sorted by score
    const moves = [];
    for (let i = 0; i < output.length; i++) {
      const score = output[i];
      const rotation = i % 4;
      const cellIndex = Math.floor(i / 4);
      const x = cellIndex % gridManager.cols;
      const y = Math.floor(cellIndex / gridManager.cols);
      moves.push({ x, y, rotation, score });
    }
    moves.sort((a, b) => b.score - a.score);

    // Return highest-scoring valid move
    for (const move of moves) {
      if (gridManager.isValidMove(move.x, move.y, move.rotation)) {
        return move;
      }
    }

    // No valid move from network — fallback
    return this._getRandomValidMove(gridManager);
  }

  /**
   * Random valid move using GridManager.getValidMoves()
   */
  _getRandomValidMove(gridManager) {
    const validMoves = gridManager.getValidMoves();
    if (validMoves.length === 0) return null;
    return validMoves[Math.floor(Math.random() * validMoves.length)];
  }

  /**
   * Build state representation from GridManager
   * Same format as original AIPlayer._getStateRepresentation
   * Note: For self-play mode='network', use _getStateRelativeToCurrentPlayer instead.
   */
  _getStateRepresentation(gridManager) {
    const state = [];
    for (let y = 0; y < gridManager.rows; y++) {
      for (let x = 0; x < gridManager.cols; x++) {
        const cell = gridManager.grid[y][x];
        const t1 = cell.triangles[0];
        const t2 = cell.triangles[1];

        const normalizePlayer = (p) => {
          if (!p) return 0;
          return p === this.playerNumber ? 1 : -1;
        };

        const normalizeRotation = (r) => {
          return r === undefined || r === -1 ? -1 : r / 4.0;
        };

        state.push(normalizePlayer(t1?.player));
        state.push(normalizeRotation(t1?.rotation ?? -1));
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation ?? -1));
      }
    }
    state.push(gridManager.currentPlayer === this.playerNumber ? 1 : -1);
    return state;
  }

  /**
   * Build state representation relative to current player.
   * Used for 'network' self-play mode where both players share the same policy.
   * Current player's triangles → +1, opponent's → -1.
   */
  _getStateRelativeToCurrentPlayer(gridManager) {
    const state = [];
    const currentPlayer = gridManager.currentPlayer;
    for (let y = 0; y < gridManager.rows; y++) {
      for (let x = 0; x < gridManager.cols; x++) {
        const cell = gridManager.grid[y][x];
        const t1 = cell.triangles[0];
        const t2 = cell.triangles[1];

        const normalizePlayer = (p) => {
          if (!p) return 0;
          return p === currentPlayer ? 1 : -1;
        };

        const normalizeRotation = (r) => {
          return r === undefined || r === -1 ? -1 : r / 4.0;
        };

        state.push(normalizePlayer(t1?.player));
        state.push(normalizeRotation(t1?.rotation ?? -1));
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation ?? -1));
      }
    }
    // Current player is always self in this representation
    state.push(1);
    return state;
  }

  // === Training Methods ===

  /**
   * Calculate immediate reward vector for all possible moves
   * Uses GridManager.clone() for safe simulation
   * 
   * Reward components (Exp 10):
   * 1. Enclosure claims: +0.3 per claimed triangle
   * 2. Connected cluster growth: +0.1 per triangle in cluster
   * 3. Largest-cluster lead: +0.2 per triangle lead (NEW - directly optimizes win condition)
   * 
   * @param {GridManager} gridManager - The game state
   * @param {number} [forPlayer] - Player to compute rewards for (defaults to this.playerNumber)
   */
  calculateImmediateRewardVector(gridManager, forPlayer = null) {
    const player = forPlayer ?? this.playerNumber;
    const opponent = player === 1 ? 2 : 1;
    const { rows, cols } = gridManager;
    const outputVector = new Array(rows * cols * 4).fill(0.01);

    // Reward factors
    const claimRewardFactor = 0.3;           // Enclosure claims
    const connectedClusterRewardFactor = 0.1; // Cluster growth
    const largestLeadFactor = 0.2;            // NEW: Largest cluster lead (win condition)

    // Get current largest clusters (before move) for baseline
    const myCurrentLargest = gridManager.getLargestContiguous(player);
    const oppCurrentLargest = gridManager.getLargestContiguous(opponent);

    // REALISTIC reward bounds (not theoretical maximums)
    // Based on observed game data, not unreachable edge cases
    // Claim: 0-5 triangles typical, max 5 → max bonus 1.5
    // Cluster: 1-100 triangles typical, max ~100 → max bonus 10.0
    // Lead delta: -20 to +20 typical per move → max bonus ±4.0
    // Total realistic max immediate: ~15.5 (not 61.5!)
    const maxClaimBonus = 1.5;
    const maxClusterBonus = 10.0;  // Realistic: 100 triangles × 0.1
    const maxLeadBonus = 4.0;       // Realistic: 20 lead delta × 0.2
    const maxImmediateReward = maxClaimBonus + maxClusterBonus + maxLeadBonus;
    const rewardRange = maxImmediateReward;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        for (let rotation = 0; rotation < 4; rotation++) {
          const moveIndex = (y * cols + x) * 4 + rotation;

          if (!gridManager.isValidMove(x, y, rotation)) continue;

          // Simulate on a clone
          const tempManager = gridManager.clone();
          tempManager.currentPlayer = player;
          
          const simSuccess = tempManager.placeTriangle(x, y, rotation, player);

          if (simSuccess) {
            // 1. Enclosure claim reward
            const simClaimed = tempManager.claimEnclosedTriangles(player);
            const hypoClaimReward = simClaimed * claimRewardFactor;

            // 2. Connected cluster reward (cluster this move joins/grows)
            const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);
            const hypoClusterReward = simClusterSize * connectedClusterRewardFactor;

            // 3. NEW: Largest-cluster lead reward (directly optimizes win condition)
            const myNewLargest = tempManager.getLargestContiguous(player);
            const oppNewLargest = tempManager.getLargestContiguous(opponent);
            const myLargestGain = myNewLargest - myCurrentLargest;
            const oppLargestGain = oppNewLargest - oppCurrentLargest;
            const leadDelta = (myNewLargest - oppNewLargest) - (myCurrentLargest - oppCurrentLargest);
            const hypoLeadReward = leadDelta * largestLeadFactor;

            // Combined reward
            const hypotheticalReward = hypoClaimReward + hypoClusterReward + hypoLeadReward;

            let scaledReward = rewardRange !== 0
              ? hypotheticalReward / rewardRange
              : 0.01;
            outputVector[moveIndex] = Math.max(0.01, Math.min(0.99, scaledReward));
          }
        }
      }
    }

    return outputVector;
  }

  /**
   * Add training sample from a game step
   * 
   * Reward scaling for Exp 10 (FIXED):
   * Uses REALISTIC bounds, not theoretical maximums.
   * - Claim bonus: max 1.5
   * - Cluster bonus: max 10.0 (realistic: 100 triangles × 0.1)
   * - Lead bonus: max 4.0 (realistic: ±20 lead delta × 0.2)
   * - Win/loss: ±1.0
   * 
   * Total realistic max: ~16.5 (not 62.5!)
   */
  addTrainingData(state, move, reward, gridManager) {
    const outputSize = gridManager.rows * gridManager.cols * 4;
    const output = new Array(outputSize).fill(0.01);
    const moveIndex = (move.y * gridManager.cols + move.x) * 4 + move.rotation;

    if (moveIndex < 0 || moveIndex >= outputSize) return;

    // REALISTIC reward bounds (matching immediate reward calculation)
    const maxClaimBonus = 1.5;
    const maxClusterBonus = 10.0;  // Realistic: 100 triangles × 0.1
    const maxLeadBonus = 4.0;       // Realistic: ±20 lead delta × 0.2
    const minReward = -5.0;        // Realistic worst case: large negative lead
    const maxReward = 1.0 + maxClaimBonus + maxClusterBonus + maxLeadBonus;  // ~16.5
    const range = maxReward - minReward;  // ~21.5

    let targetOutput = range !== 0
      ? (reward - minReward) / range
      : 0.5;
    targetOutput = Math.max(0.01, Math.min(0.99, targetOutput));

    output[moveIndex] = targetOutput;
    this.trainingData.push({ input: state, output });
  }

  /**
   * Train the network on accumulated training data
   * @returns {{error: number, iterations: number, initialError: number} | null}
   */
  trainNetwork(iterations = 500, learningRate = 0.1) {
    if (!this.network || this.trainingData.length === 0) return null;

    // Compute initial error (forward pass only, no training)
    let initialError = 0;
    for (const sample of this.trainingData) {
      const output = this.network.run(sample.input);
      let sampleError = 0;
      for (let i = 0; i < output.length; i++) {
        const delta = output[i] - sample.output[i];
        sampleError += delta * delta;
      }
      initialError += sampleError / output.length;
    }
    initialError /= this.trainingData.length;

    // Track error at intervals to measure convergence rate
    const errorSnapshots = [{ iteration: 0, error: initialError }];

    const result = this.network.train(this.trainingData, {
      iterations,
      log: false,
      learningRate,
      errorThresh: 0.00093,
      callbackPeriod: Math.max(1, Math.floor(iterations / 10)),
      callback: (status) => {
        errorSnapshots.push({ iteration: status.iterations, error: status.error });
      },
    });

    // Add final error if not already captured
    if (result && errorSnapshots[errorSnapshots.length - 1].iteration !== result.iterations) {
      errorSnapshots.push({ iteration: result.iterations, error: result.error });
    }

    if (result && result.error < 1) {
      this.isNetworkLoaded = true;
    }

    // Calculate error reduction rate: what fraction of total error drop happens in first 50% of iterations?
    let errorReductionHalf = null;
    if (errorSnapshots.length >= 3) {
      const halfIter = Math.floor(iterations / 2);
      const atHalf = errorSnapshots.find(s => s.iteration >= halfIter) || errorSnapshots[errorSnapshots.length - 1];
      const totalDrop = initialError - (result?.error ?? initialError);
      if (totalDrop > 0) {
        const halfDrop = initialError - atHalf.error;
        errorReductionHalf = halfDrop / totalDrop;
      }
    }

    return {
      ...result,
      initialError,
      errorSnapshots,
      errorReductionHalf,
    };
  }

  /**
   * Run a self-play game for training
   * Compatible with GameSession's game loop but also usable standalone
   */
  runSelfPlayGame(gridManager, gradientIsolation = false) {
    const gameHistory = [];
    const opponentPlayer = this.playerNumber === 1 ? 2 : 1;
    let currentPlayerNum = 1;
    let movesCount = 0;
    let firstMove = null;
    const maxMoves = gridManager.rows * gridManager.cols * 2 * 1.5;

    // Per-game exploration: determine once at game start
    // In 'perGame' mode, either the whole game is network-play or random-play
    // This gives coherent game trajectories instead of per-move noise
    const isRandomGame = this.explorationMode === 'perGame'
      ? Math.random() < this.explorationFactor
      : false;

    while (!gridManager.isGridFull() && movesCount < maxMoves) {
      const currentIsSelf = currentPlayerNum === this.playerNumber;
      // For 'network' self-play, state is always relative to current player
      // Set current player BEFORE computing state (so state representation sees correct player)
      gridManager.currentPlayer = currentPlayerNum;
      
      const state = this.selfPlayMode === 'network'
        ? this._getStateRelativeToCurrentPlayer(gridManager)
        : this._getStateRepresentation(gridManager);

      // Calculate immediate rewards from current player's perspective for 'network' mode
      const immediateRewardVector = this.selfPlayMode === 'network'
        ? this.calculateImmediateRewardVector(gridManager, currentPlayerNum)
        : this.calculateImmediateRewardVector(gridManager);

      // Choose move based on self-play mode
      let move;
      
      // Opening diversity injection: 50% random for first 5 moves
      const isOpeningPhase = movesCount < 5;
      const openingRandomFactor = isOpeningPhase ? 0.50 : 0;
      
      // Per-game random: entire game plays random (coherent exploration)
      if (isRandomGame) {
        move = this._getRandomValidMove(gridManager);
      } else if (!currentIsSelf && this.selfPlayMode === 'random') {
        // Mode: random opponent — opponent plays randomly (no network influence)
        move = this._getRandomValidMove(gridManager);
      } else if (this.isNetworkLoaded && movesCount > 0) {
        // Both self and network-mode opponent use the same policy
        // Network is conditioned on currentPlayer via input, so it predicts good moves for whoever is moving
        const rand = Math.random();
        
        if (rand < openingRandomFactor) {
          // Opening diversity: random move to explore different openings
          move = this._getRandomValidMove(gridManager);
        } else if (rand < 0.10 + openingRandomFactor) {
          // Random exploration — discover new moves
          move = this._getRandomValidMove(gridManager);
        } else if (rand < 0.20 + openingRandomFactor) {
          // Greedy immediate reward — structured exploration
          move = this._predictMoveFromRewards(immediateRewardVector, gridManager);
        } else {
          // Network prediction — exploitation
          move = this._getBestNetworkMove(gridManager);
        }
      } else {
        move = this._getRandomValidMove(gridManager);
      }

      if (!move) break;

      const actualMoveIndex = (move.y * gridManager.cols + move.x) * 4 + move.rotation;

      // Execute move
      const success = gridManager.placeTriangle(move.x, move.y, move.rotation, currentPlayerNum);
      if (!success) break;
      
      // Track first move for diversity metric
      if (movesCount === 0) {
        firstMove = { x: move.x, y: move.y, rotation: move.rotation };
      }

      const claimed = gridManager.claimEnclosedTriangles(currentPlayerNum);

      gameHistory.push({
        state,
        move,
        player: currentPlayerNum,
        claimed,
        immediateRewardVector,
        // Store raw immediate reward for TD(λ) combination with win/cluster rewards
        rawImmediateReward: immediateRewardVector[actualMoveIndex] * 15.5, // De-normalize
        actualMoveIndex,
      });

      // Handle extra turns
      gridManager.extraTurns += Math.floor((1 + claimed) / 2);
      if (gridManager.extraTurns > 0) {
        gridManager.extraTurns--;
      } else {
        currentPlayerNum = currentPlayerNum === 1 ? 2 : 1;
      }

      movesCount++;
    }

    // Determine winner
    let winner = 0;
    let p1Score = 0, p2Score = 0;
    if (gridManager.isGridFull()) {
      p1Score = gridManager.getLargestContiguous(1);
      p2Score = gridManager.getLargestContiguous(2);
      if (p1Score > p2Score) winner = 1;
      else if (p2Score > p1Score) winner = 2;
    }

    // === TD(λ) Cluster-Contribution Reward ===
    // After game ends, identify triangles in each player's largest cluster
    // Only moves that placed triangles IN the final largest cluster get
    // the cluster contribution bonus, with TD(λ) decay going backwards.
    // Win/loss reward still propagates equally (λ=1) to all moves.
    //
    // CONFIGURABLE: Set config.enableClusterReward = true to enable.
    // Default: disabled (Exp 12+). The cluster reward added noise that
    // suppressed learning (see Exp 11 results).

    const gamma = this.clusterGamma;    // Discount factor
    const lambda = this.clusterLambda;    // Eligibility trace decay for cluster reward
    const clusterRewardFactor = this.clusterRewardFactor;  // Per-triangle reward in the largest cluster

    // Get the actual triangles in each player's largest cluster
    const p1LargestCluster = gridManager.getLargestContiguousCluster(1);
    const p2LargestCluster = gridManager.getLargestContiguousCluster(2);

    // Build lookup sets for fast membership testing
    const p1ClusterSet = this.enableClusterReward ? new Set(p1LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`)) : null;
    const p2ClusterSet = this.enableClusterReward ? new Set(p2LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`)) : null;

    // Total cluster bonus per player (proportional to cluster size)
    const p1ClusterBonus = p1LargestCluster.length * clusterRewardFactor;
    const p2ClusterBonus = p2LargestCluster.length * clusterRewardFactor;

    // Process trajectory backwards for training data
    const finalRewardWeight = 1.0;
    const totalMoves = gameHistory.length;

    // Updated normalization bounds
    // Without cluster reward: immediate (0-15.5) + win/loss (±1.0) = range ~17.5
    // With cluster reward: immediate (0-15.5) + win/loss (±1.0) + cluster (0-10.0) = range ~27.5
    const minImmediateReward = 0;
    const maxImmediateReward = 15.5;  // Realistic max
    const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight) - (this.enableClusterReward ? 0 : 0);
    const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight) + (this.enableClusterReward ? 10.0 : 0);
    const combinedRewardRange = maxCombinedReward - minCombinedReward;

    // TD(λ) cluster contribution reward
    // Process trajectory backwards from game end. Each contributing move
    // gets credit based on its distance from the end of the game,
    // with exponential decay: credit = clusterBonus × (γλ)^stepsFromEnd
    //
    // Example (P1, going backwards):
    //   Move 149 (P1, IN cluster): 1 step from end → bonus × (0.891)^1 = bonus × 0.891
    //   Move 148 (P2 move):        2 steps from end → no P1 cluster reward
    //   Move 140 (P1, IN cluster): 10 steps from end → bonus × (0.891)^10 = bonus × 0.30
    //   Move 100 (P1, IN cluster): 50 steps from end → bonus × (0.891)^50 = bonus × 0.004
    //
    // Result: moves closer to game end get much more credit for the cluster.
    // Opening moves that started the cluster still get *some* credit,
    // but endgame moves that sealed it get the lion's share.
    let stepsFromEnd = 0;

    for (let t = totalMoves - 1; t >= 0; t--) {
      const step = gameHistory[t];
      stepsFromEnd = totalMoves - 1 - t;
      
      // Skip opponent moves only in 'random' mode
      if (this.selfPlayMode === 'random' && step.player !== this.playerNumber) continue;

      // Determine reward from this move's player's perspective
      let rewardForPlayer;
      if (this.selfPlayMode === 'network') {
        rewardForPlayer = winner === step.player ? 1.0
          : (winner === 0 ? 0.0 : -1.0);
      } else {
        rewardForPlayer = winner === this.playerNumber ? 1.0
          : (winner === 0 ? 0.0 : -1.0);
      }

      const finalOutputVector = [...step.immediateRewardVector];
      const immediateReward = step.rawImmediateReward;  // Use raw (de-normalized) value

      // 1. Win/loss reward: λ=1, equal credit for all moves
      const winReward = rewardForPlayer * finalRewardWeight;

      // 2. Cluster contribution reward: TD(λ=0.9)
      //    Only moves whose triangles are IN the final largest cluster get this.
      //    Credit decays based on distance from game end.
      //    DISABLED by default (enableClusterReward=false).
      let clusterReward = 0;
      if (this.enableClusterReward) {
        const moveKey = `${step.move.x},${step.move.y},${step.move.rotation}`;

        const isP1Contributor = step.player === 1 && p1ClusterSet.has(moveKey);
        const isP2Contributor = step.player === 2 && p2ClusterSet.has(moveKey);

        if (isP1Contributor || isP2Contributor) {
          // This move contributed to the winning cluster
          // Credit decays exponentially with distance from game end
          const eligibility = Math.pow(gamma * lambda, stepsFromEnd);
          const clusterBonus = isP1Contributor ? p1ClusterBonus : p2ClusterBonus;
          clusterReward = clusterBonus * eligibility;
        }
      }

      // Combined: immediate + win/loss (λ=1) + cluster contribution (TD λ=0.9)
      const combinedReward = immediateReward + winReward + clusterReward;

      let scaledCombined = combinedRewardRange !== 0
        ? (combinedReward - minCombinedReward) / combinedRewardRange
        : 0.5;
      scaledCombined = Math.max(0.01, Math.min(0.99, scaledCombined));

      if (gradientIsolation && this.network) {
        // Exp 13a: Self-anchored target masking
        // Get current network prediction for unchosen moves
        const currentPrediction = this.network.run(step.state);
        
        // Build target: anchor unchosen moves to current prediction (zero gradient)
        // Only the chosen move gets the learning signal
        for (let i = 0; i < 400; i++) {
          if (i === step.actualMoveIndex) {
            finalOutputVector[i] = scaledCombined; // chosen move gets reward
          } else {
            finalOutputVector[i] = currentPrediction[i] || 0.5; // anchor to prediction (no gradient)
          }
        }
        
        // Suppress illegal moves to minimum baseline
        // Check legality based on state at time of move
        for (let y = 0; y < 10; y++) {
          for (let x = 0; x < 10; x++) {
            const cellIndex = y * 10 + x;
            const cellValue = step.state[cellIndex];
            if (cellValue !== 0) {
              // Cell is occupied, all 4 rotations illegal
              for (let r = 0; r < 4; r++) {
                const moveIndex = (y * 10 + x) * 4 + r;
                finalOutputVector[moveIndex] = 0.01;
              }
            }
          }
        }
      } else {
        // Original behavior: all moves get targets (immediate reward heuristic)
        finalOutputVector[step.actualMoveIndex] = scaledCombined;
      }
      
      this.trainingData.push({ input: step.state, output: finalOutputVector });
    }

    return { winner, p1Score, p2Score, moves: movesCount, firstMove, trainingSamplesAdded: totalMoves };
  }

  /**
   * Pick the move with highest immediate reward
   */
  _predictMoveFromRewards(rewardVector, gridManager) {
    let bestIndex = -1;
    let maxReward = -Infinity;
    for (let i = 0; i < rewardVector.length; i++) {
      if (rewardVector[i] > 0.01 && rewardVector[i] > maxReward) {
        maxReward = rewardVector[i];
        bestIndex = i;
      }
    }
    if (bestIndex === -1) return this._getRandomValidMove(gridManager);

    const rotation = bestIndex % 4;
    const cellIndex = Math.floor(bestIndex / 4);
    const x = cellIndex % gridManager.cols;
    const y = Math.floor(cellIndex / gridManager.cols);
    return { x, y, rotation };
  }

  /**
   * Get best move from network prediction
   */
  _getBestNetworkMove(gridManager) {
    // Use state representation consistent with training data
    const state = this.selfPlayMode === 'network'
      ? this._getStateRelativeToCurrentPlayer(gridManager)
      : this._getStateRepresentation(gridManager);
    const output = this.network.run(state);

    const moves = [];
    for (let i = 0; i < output.length; i++) {
      const rotation = i % 4;
      const cellIndex = Math.floor(i / 4);
      const x = cellIndex % gridManager.cols;
      const y = Math.floor(cellIndex / gridManager.cols);
      moves.push({ x, y, rotation, score: output[i] });
    }
    moves.sort((a, b) => b.score - a.score);

    for (const move of moves) {
      if (gridManager.isValidMove(move.x, move.y, move.rotation)) {
        return move;
      }
    }
    return this._getRandomValidMove(gridManager);
  }

  // === Persistence ===

  exportNetwork() {
    return this.network ? this.network.toJSON() : null;
  }

  async importNetworkFromJSON(jsonData) {
    if (!this.network) {
      const brain = await this._getBrain();
      if (brain) this._createNetwork(brain);
      if (!this.network) return false;
    }
    try {
      this.network.fromJSON(jsonData);
      this.isNetworkLoaded = true;
      return true;
    } catch (e) {
      console.error('Failed to import network:', e);
      return false;
    }
  }

  async importNetworkFromFile(filePath = './trainedAI_network.json') {
    try {
      const { readFileSync } = await import('fs');
      const data = readFileSync(filePath, 'utf-8');
      return this.importNetworkFromJSON(JSON.parse(data));
    } catch (e) {
      // Browser fallback — try fetch
      try {
        const response = await fetch(filePath, { cache: 'no-store' });
        if (response.ok) {
          const data = await response.json();
          return this.importNetworkFromJSON(data);
        }
      } catch (fetchErr) {
        console.warn('Could not load network from file or fetch:', fetchErr.message);
      }
      return false;
    }
  }

  // Override
  isReady() {
    return this.network !== null;
  }

  getTrainingData() {
    return this.trainingData;
  }

  /**
   * Replace training data (used by experience replay buffer).
   */
  setTrainingData(data) {
    this.trainingData = data;
  }

  reset() {
    super.reset();
    this.trainingData = [];
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.AIPlayer = AIPlayer;
}
