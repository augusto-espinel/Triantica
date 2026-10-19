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
   * @param {object} config.networkJSON - Pre-trained network state (optional)
   * @param {Function} config.brainFactory - () => brain.js reference (for testing/DI)
   */
  constructor(playerNumber, config = {}) {
    super(playerNumber, config);
    this.type = PlayerType.AI;
    this.difficulty = config.difficulty || 'medium';
    this.explorationFactor = config.explorationFactor ?? 0.2;
    this.immediacyFactor = config.immediacyFactor ?? 0.9;
    this.selfPlayMode = config.selfPlayMode || 'random'; // 'random' | 'network'
    this.isNetworkLoaded = false;
    this.trainingData = [];
    this._networkJSON = config.networkJSON || null;
    this._brainFactory = config.brainFactory || null;
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

    this.isNetworkLoaded = false;
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

    // Epsilon-greedy: explore vs exploit
    if (Math.random() < this.explorationFactor) {
      return this._getRandomValidMove(gridManager);
    }

    // Exploit: use network prediction
    // Use state representation consistent with training data
    const state = this.selfPlayMode === 'network'
      ? this._getStateRelativeToCurrentPlayer(gridManager)
      : this._getStateRepresentation(gridManager);
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
   * @param {GridManager} gridManager - The game state
   * @param {number} [forPlayer] - Player to compute rewards for (defaults to this.playerNumber)
   */
  calculateImmediateRewardVector(gridManager, forPlayer = null) {
    const player = forPlayer ?? this.playerNumber;
    const { rows, cols } = gridManager;
    const outputVector = new Array(rows * cols * 4).fill(0.01);

    const claimRewardFactor = 0.3;
    const connectedClusterRewardFactor = 0.1;
    const maxPossibleClaimBonus = 1.5;
    const maxPossibleClusterSize = rows * cols * 2;
    const maxPossibleClusterBonus = maxPossibleClusterSize * connectedClusterRewardFactor;
    const maxImmediateReward = maxPossibleClaimBonus + maxPossibleClusterBonus;
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
            const simClaimed = tempManager.claimEnclosedTriangles(player);
            const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);

            const hypoClaimReward = simClaimed * claimRewardFactor;
            const hypoClusterReward = simClusterSize * connectedClusterRewardFactor;
            const hypotheticalReward = hypoClaimReward + hypoClusterReward;

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
   */
  addTrainingData(state, move, reward, gridManager) {
    const outputSize = gridManager.rows * gridManager.cols * 4;
    const output = new Array(outputSize).fill(0.01);
    const moveIndex = (move.y * gridManager.cols + move.x) * 4 + move.rotation;

    if (moveIndex < 0 || moveIndex >= outputSize) return;

    // Scale reward to [0.01, 0.99]
    const maxClaimBonus = 1.5;
    const maxClusterBonus = gridManager.rows * gridManager.cols * 2 * 0.1;
    const minReward = -1.0;
    const maxReward = 1.0 + maxClaimBonus + maxClusterBonus;
    const range = maxReward - minReward;

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
  runSelfPlayGame(gridManager) {
    const gameHistory = [];
    const opponentPlayer = this.playerNumber === 1 ? 2 : 1;
    let currentPlayerNum = 1;
    let movesCount = 0;
    const maxMoves = gridManager.rows * gridManager.cols * 2 * 1.5;

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
      if (!currentIsSelf && this.selfPlayMode === 'random') {
        // Mode: random opponent — opponent plays randomly (no network influence)
        move = this._getRandomValidMove(gridManager);
      } else if (this.isNetworkLoaded && movesCount > 0) {
        // Both self and network-mode opponent use the same policy
        // Network is conditioned on currentPlayer via input, so it predicts good moves for whoever is moving
        const rand = Math.random();
        if (rand < 0.10) {
          // Random exploration — discover new moves
          move = this._getRandomValidMove(gridManager);
        } else if (rand < 0.20) {
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

      const claimed = gridManager.claimEnclosedTriangles(currentPlayerNum);

      gameHistory.push({
        state,
        move,
        player: currentPlayerNum,
        claimed,
        immediateRewardVector,
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

    // Process trajectory backwards for training data
    const finalRewardWeight = 1.0;
    const totalMoves = gameHistory.length;

    const minImmediateReward = 0.01;
    const maxImmediateReward = 0.99;
    const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight);
    const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight);
    const combinedRewardRange = maxCombinedReward - minCombinedReward;

    // For 'network' mode, train on both players' moves with proper reward perspective
    for (let t = totalMoves - 1; t >= 0; t--) {
      const step = gameHistory[t];
      
      // Skip opponent moves only in 'random' mode
      if (this.selfPlayMode === 'random' && step.player !== this.playerNumber) continue;

      // Determine reward from this move's player's perspective
      let rewardForPlayer;
      if (this.selfPlayMode === 'network') {
        // In network mode, reward is from the moving player's perspective
        rewardForPlayer = winner === step.player ? 1.0
          : (winner === 0 ? 0.0 : -1.0);
      } else {
        // In random mode, reward is always from self's perspective
        rewardForPlayer = winner === this.playerNumber ? 1.0
          : (winner === 0 ? 0.0 : -1.0);
      }

      const finalOutputVector = [...step.immediateRewardVector];
      const discountFactor = 1.0;
      const scaledFinalReward = rewardForPlayer * discountFactor * finalRewardWeight;

      const immediateReward = step.immediateRewardVector[step.actualMoveIndex];
      const combinedReward = immediateReward + scaledFinalReward;

      let scaledCombined = combinedRewardRange !== 0
        ? (combinedReward - minCombinedReward) / combinedRewardRange
        : 0.5;
      scaledCombined = Math.max(0.01, Math.min(0.99, scaledCombined));

      finalOutputVector[step.actualMoveIndex] = scaledCombined;
      this.trainingData.push({ input: step.state, output: finalOutputVector });
    }

    gridManager.reset();
    return { winner, p1Score, p2Score, moves: movesCount, trainingSamplesAdded: totalMoves };
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
