/**
 * Automated training pipeline for Triantica AI
 *
 * Runs self-play games → trains network → evaluates vs random → repeats
 * Designed for M2 Ultra: uses single-threaded brain.js training
 * (brain.js GPU not available in Node, but M2 is fast enough for CPU)
 *
 * Usage:
 *   node src/server/train.js [options]
 *
 * Options:
 *   --cycles N         Number of train+eval cycles (default: 10)
 *   --games N          Self-play games per cycle (default: 50)
 *   --eval-games N     Evaluation games vs random (default: 50)
 *   --iterations N     Training iterations per cycle (default: 200)
 *   --learning-rate F  Learning rate (default: 0.1)
 *   --exploration F    Exploration factor (default: 0.3)
 *   --self-play-mode M  Self-play mode: 'random' or 'network' (default: random)
 *   --start-from FILE  Load pre-trained network (default: trainedAI_network.json, use 'fresh' for random init)
 *   --output FILE      Save trained network (default: trainedAI_network_v2.json)
 *   --audit            Enable game auditing (random sampling for bug detection)
 *   --audit-rate F     Audit sample rate 0.0-1.0 (default: 0.05 = 5% of games)
 */

/**
 * Compute Shannon entropy of a distribution
 * @param {number[]} counts - Array of counts for each category
 * @returns {number} Entropy in bits
 */
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

/**
 * Compute median of an array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Median value
 */
function computeMedian(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Compute variance of an array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Variance
 */
function computeVariance(arr) {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
}

/**
 * Compute min of an array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Minimum value
 */
function computeMin(arr) {
  if (arr.length === 0) return 0;
  return Math.min(...arr);
}

/**
 * Compute max of an array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Maximum value
 */
function computeMax(arr) {
  if (arr.length === 0) return 0;
  return Math.max(...arr);
}

/**
 * Compute opening move diversity (entropy of first-move distribution)
 * @param {string[]} firstMoves - Array of "x,y,rN" strings
 * @returns {number} Opening diversity (0 = same move every game)
 */
function computeOpeningDiversity(firstMoves) {
  if (firstMoves.length === 0) return 0;
  const counts = {};
  for (const move of firstMoves) {
    counts[move] = (counts[move] || 0) + 1;
  }
  return computeEntropy(Object.values(counts));
}

/**
 * Compute final board diversity (mean cell entropy across all positions)
 * @param {Object[]} finalBoards - Array of boardState arrays
 * @returns {number} Board diversity (0 = identical boards)
 */
function computeBoardDiversity(finalBoards) {
  if (finalBoards.length === 0) return 0;
  const numCells = finalBoards[0].length;
  let totalEntropy = 0;

  for (let i = 0; i < numCells; i++) {
    // Count ownership states for this cell across all games
    // State: "t1_p,t2_p" (e.g., "1,2" means t1 owned by P1, t2 owned by P2)
    const stateCounts = {};
    for (const board of finalBoards) {
      const state = `${board[i].t1},${board[i].t2}`;
      stateCounts[state] = (stateCounts[state] || 0) + 1;
    }
    totalEntropy += computeEntropy(Object.values(stateCounts));
  }

  return totalEntropy / numCells;
}

/**
 * Automated training pipeline for Triantica AI
 *
 * Runs self-play games → trains network → evaluates vs random → repeats
 * Designed for M2 Ultra: uses single-threaded brain.js training
 * (brain.js GPU not available in Node, but M2 is fast enough for CPU)
 *
 * Usage:
 *   node src/server/train.js [options]
 *
 * Options:
 *   --cycles N         Number of train+eval cycles (default: 10)
 *   --games N          Self-play games per cycle (default: 50)
 *   --eval-games N     Evaluation games vs random (default: 50)
 *   --iterations N     Training iterations per cycle (default: 200)
 *   --learning-rate F  Learning rate (default: 0.1)
 *   --exploration F    Exploration factor (default: 0.3)
 *   --self-play-mode M  Self-play mode: 'random' or 'network' (default: random)
 *   --start-from FILE  Load pre-trained network (default: trainedAI_network.json, use 'fresh' for random init)
 *   --output FILE      Save trained network (default: trainedAI_network_v2.json)
 *   --save-every N     Save checkpoint every N cycles (default: 5)
 *   --log FILE         Log progress to file (default: experiments/logs/training-TIMESTAMP.log)
 */

import { GridManager } from '../core/GridManager.js';
import { GameSession } from '../core/GameSession.js';
import { RandomPlayer } from '../players/RandomPlayer.js';
import { GameAuditor } from './GameAuditor.js';
import { AIPlayer } from '../players/AIPlayer.js';
import { GameLogger } from '../data/GameLogger.js';
import { ReplayBuffer } from '../data/ReplayBuffer.js';
import { HeuristicPlayer } from '../players/HeuristicPlayer.js';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Parse CLI args
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    cycles: 10,
    games: 50,
    evalGames: 50,
    iterations: 200,
    learningRate: 0.1,
    exploration: 0.3,
    selfPlayMode: 'network', // 'random' (vs random) | 'network' (proper self-play)
    startFrom: 'trainedAI_network.json',
    output: 'trainedAI_network_v2.json',
    saveEvery: 5,
    log: null,
    replayRatio: 0.5, // fraction of training data from replay buffer
    replaySize: 50000, // max replay buffer size
    explorationMode: 'perMove', // 'perMove' | 'perGame'
    opponentMix: 'network', // 'random' | 'network' | 'heuristic' | 'mixed'
    heuristicRatio: 0.4, // fraction of games vs heuristic (when opponentMix='mixed')
    randomRatio: 0.2, // fraction of games vs random (when opponentMix='mixed')
    enableClusterReward: false, // enable TD(λ) cluster-contribution reward
    clusterRewardFactor: 0.1, // per-triangle cluster bonus
    clusterLambda: 0.9, // eligibility trace decay
    clusterGamma: 0.99, // discount factor
    gradientIsolation: false, // Exp 13a: only train on chosen move, anchor unchosen to current prediction
    trainOnOpponentMoves: false, // Exp 12s: learn from opponent's winning moves
    audit: false, // Enable game auditing for bug detection
    auditRate: 0.05, // Fraction of games to audit (default 5%)
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith('--')) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const val = args[++i];
      if (key === 'learningRate' || key === 'exploration' || key === 'replayRatio' || key === 'heuristicRatio' || key === 'randomRatio' || key === 'clusterRewardFactor' || key === 'clusterLambda' || key === 'clusterGamma' || key === 'auditRate') {
        opts[key] = parseFloat(val);
      } else if (key === 'startFrom' || key === 'output' || key === 'selfPlayMode' || key === 'log' || key === 'explorationMode' || key === 'opponentMix') {
        opts[key] = val;
      } else if (key === 'enableClusterReward' || key === 'gradientIsolation' || key === 'trainOnOpponentMoves' || key === 'audit') {
        opts[key] = val === 'true';
      } else if (key === 'replaySize' || key === 'trainOnOpponentMoves') {
        // trainOnOpponentMoves is boolean, handled above
        opts[key] = parseInt(val, 10);
      } else {
        opts[key] = parseInt(val, 10);
      }
    }
  }
  return opts;
}

/**
 * Run a game: AI (P1) vs an opponent player (P2)
 * When trainOnOpponentMoves is true, collects training data from both perspectives.
 * @param {AIPlayer} aiPlayer - The learning AI (always P1)
 * @param {object} opponent - Opponent player with getMove(gridManager) method (P2)
 * @param {GridManager} gridManager - The game state
 * @param {boolean} gradientIsolation - Enable gradient isolation
 * @param {boolean} trainOnOpponent - Also train on opponent's moves from their perspective
 * @returns {Promise<{ winner, p1Score, p2Score, moves, firstMove, trainingSamplesAdded }>}
 */
async function _runGameVsOpponent(aiPlayer, opponent, gridManager, gradientIsolation = false, trainOnOpponent = false) {
  const player = 1; // AI is always P1
  gridManager.currentPlayer = 1;
  gridManager.reset();
  
  const gameHistory = [];      // Full move log for auditing (ALL moves)
  const trainingHistory = [];   // Training data (AI moves + optionally opponent moves)
  let movesCount = 0;
  let firstMove = null;
  
  while (!gridManager.isGridFull()) {
    const currentPlayer = gridManager.currentPlayer;
    
    let move;
    if (currentPlayer === player) {
      // AI's turn
      const state = aiPlayer.selfPlayMode === 'network'
        ? aiPlayer._getStateRelativeToCurrentPlayer(gridManager)
        : aiPlayer._getStateRepresentation(gridManager);
      const immediateRewardVector = aiPlayer.calculateImmediateRewardVector(gridManager, player);
      const rawImmediateReward = immediateRewardVector.reduce((sum, v) => sum + v, 0) / immediateRewardVector.length;
      
      move = await aiPlayer.getMove(gridManager);
      if (!move) break;
      
      const actualMoveIndex = (move.y * gridManager.cols + move.x) * 4 + move.rotation;
      
      // Record for training
      trainingHistory.push({
        player,
        move,
        state,
        immediateRewardVector,
        rawImmediateReward,
        actualMoveIndex,
        claimed: 0, // Will be updated after placement
      });
      
      if (!firstMove) firstMove = move;
    } else {
      // Opponent's turn
      move = opponent.getMove(gridManager);
      if (!move) {
        console.error('Opponent returned null/undefined move');
        break;
      }
      if (typeof move.x !== 'number' || typeof move.y !== 'number' || typeof move.rotation !== 'number') {
        console.error('Opponent returned invalid move:', JSON.stringify(move));
        break;
      }
      
      // Exp 12s: also collect training data from opponent's perspective
      if (trainOnOpponent) {
        const state = aiPlayer._getStateRelativeToCurrentPlayer(gridManager);
        const immediateRewardVector = aiPlayer.calculateImmediateRewardVector(gridManager, 2);
        const rawImmediateReward = immediateRewardVector.reduce((sum, v) => sum + v, 0) / immediateRewardVector.length;
        const actualMoveIndex = (move.y * gridManager.cols + move.x) * 4 + move.rotation;
        
        trainingHistory.push({
          player: 2,
          move,
          state,
          immediateRewardVector,
          rawImmediateReward,
          actualMoveIndex,
          claimed: 0, // Will be updated after placement
        });
      }
    }
    
    // Record ALL moves for audit replay
    gameHistory.push({
      player: currentPlayer,
      move,
      claimed: 0, // Will be updated after placement
    });
    
    let success = gridManager.placeTriangle(move.x, move.y, move.rotation, currentPlayer);
    
    // FIX: Retry with random valid move if first attempt failed
    if (!success) {
      console.warn(`train.js: Move invalid, retrying with random valid move`);
      const retryMoves = gridManager.getValidMoves();
      if (retryMoves.length > 0) {
        move = retryMoves[Math.floor(Math.random() * retryMoves.length)];
        success = gridManager.placeTriangle(move.x, move.y, move.rotation, currentPlayer);
        if (success) {
          // Update gameHistory with the retry move
          gameHistory[gameHistory.length - 1].move = move;
          // Update trainingHistory if present
          if (trainingHistory.length > 0 && trainingHistory[trainingHistory.length - 1].player === currentPlayer) {
            trainingHistory[trainingHistory.length - 1].move = move;
          }
        }
      }
    }
    
    if (!success) {
      console.error(`Failed to place triangle even after retry at (${move.x}, ${move.y}, r${move.rotation}) for player ${currentPlayer}`);
    }
    
    let claimed = 0;
    if (success) {
      claimed = gridManager.claimEnclosedTriangles(currentPlayer);
      // Handle extra turns (same logic as AIPlayer.runSelfPlayGame)
      gridManager.extraTurns += Math.floor((1 + claimed) / 2);
    }
    
    // Update claimed in history entries
    const lastGameEntry = gameHistory[gameHistory.length - 1];
    lastGameEntry.claimed = claimed;
    
    const lastTrainingEntry = trainingHistory[trainingHistory.length - 1];
    if (lastTrainingEntry && lastTrainingEntry.player === currentPlayer) {
      lastTrainingEntry.claimed = claimed;
    }
    
    movesCount++;
    
    // Switch to other player for next turn (respecting extra turns)
    if (gridManager.extraTurns > 0) {
      gridManager.extraTurns--;
      // Same player continues
    } else {
      gridManager.currentPlayer = currentPlayer === 1 ? 2 : 1;
    }
  }
  
  // Determine winner
  let winner = 0;
  let p1Score = 0;
  let p2Score = 0;
  if (gridManager.isGridFull()) {
    p1Score = gridManager.getLargestContiguous(1);
    p2Score = gridManager.getLargestContiguous(2);
    if (p1Score > p2Score) winner = 1;
    else if (p2Score > p1Score) winner = 2;
  }
  
  // Generate training data from AI's moves only
  const finalRewardWeight = 1.0;
  const totalMoves = trainingHistory.length;
  
  // TD(λ) cluster reward setup - both players when trainOnOpponent
  const p1LargestCluster = gridManager.getLargestContiguousCluster(1);
  const p2LargestCluster = gridManager.getLargestContiguousCluster(2);
  const p1ClusterSet = aiPlayer.enableClusterReward
    ? new Set(p1LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`))
    : null;
  const p2ClusterSet = aiPlayer.enableClusterReward
    ? new Set(p2LargestCluster.map(t => `${t.x},${t.y},${t.rotation}`))
    : null;
  const p1ClusterBonus = p1LargestCluster.length * aiPlayer.clusterRewardFactor;
  const p2ClusterBonus = p2LargestCluster.length * aiPlayer.clusterRewardFactor;
  
  // Normalization
  const minImmediateReward = -4.0;
  const maxImmediateReward = 15.5;
  const minCombinedReward = minImmediateReward - (1.0 * finalRewardWeight) - (aiPlayer.enableClusterReward ? 0 : 0);
  const maxCombinedReward = maxImmediateReward + (1.0 * finalRewardWeight) + (aiPlayer.enableClusterReward ? 10.0 : 0);
  const combinedRewardRange = maxCombinedReward - minCombinedReward;
  
  let trainingSamplesAdded = 0;
  
  for (let t = totalMoves - 1; t >= 0; t--) {
    const step = trainingHistory[t];
    const stepsFromEnd = totalMoves - 1 - t;
    
    const rewardForPlayer = winner === step.player ? 1.0 : (winner === 0 ? 0.0 : -1.0);
    const finalOutputVector = [...step.immediateRewardVector];
    const immediateReward = step.rawImmediateReward;
    
    // Win/loss reward
    const winReward = rewardForPlayer * finalRewardWeight;
    
    // Cluster contribution reward (if enabled)
    let clusterReward = 0;
    if (aiPlayer.enableClusterReward) {
      const moveKey = `${step.move.x},${step.move.y},${step.move.rotation}`;
      const clusterSet = step.player === 1 ? p1ClusterSet : p2ClusterSet;
      const clusterBonus = step.player === 1 ? p1ClusterBonus : p2ClusterBonus;
      if (clusterSet && clusterSet.has(moveKey)) {
        const eligibility = Math.pow(aiPlayer.clusterGamma * aiPlayer.clusterLambda, stepsFromEnd);
        clusterReward = clusterBonus * eligibility;
      }
    }
    
    const combinedReward = immediateReward + winReward + clusterReward;
    let scaledCombined = combinedRewardRange !== 0
      ? (combinedReward - minCombinedReward) / combinedRewardRange
      : 0.5;
    scaledCombined = Math.max(0.01, Math.min(0.99, scaledCombined));
    
    if (gradientIsolation && aiPlayer.network) {
      // Exp 13a: Self-anchored target masking
      // Get current network prediction for unchosen moves
      const currentPrediction = aiPlayer.network.run(step.state);
      
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
    
    aiPlayer.trainingData.push({ input: step.state, output: finalOutputVector });
    trainingSamplesAdded++;
  }
  
  return { winner, p1Score, p2Score, moves: movesCount, firstMove, trainingSamplesAdded, gameHistory };
}

/**
 * Run a game: AI (P1) vs Random (P2)
 * Random opponent picks any valid move uniformly at random.
 * @param {AIPlayer} aiPlayer - The learning AI (always P1)
 * @param {GridManager} gridManager - The game state
 * @returns {Promise<{ winner, p1Score, p2Score, moves, firstMove, trainingSamplesAdded }>}
 */
async function _runGameVsRandom(aiPlayer, gridManager) {
  // Create a lightweight random opponent
  const randomOpponent = {
    playerNumber: 2,
    getMove(gridManager) {
      const { rows, cols } = gridManager;
      const validMoves = [];
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          for (let rotation = 0; rotation < 4; rotation++) {
            if (gridManager.isValidMove(x, y, rotation)) {
              validMoves.push({ x, y, rotation });
            }
          }
        }
      }
      return validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
    }
  };
  return _runGameVsOpponent(aiPlayer, randomOpponent, gridManager, false);
}

/**
 * Run a game: AI (P1) vs Random (P2) with gradient isolation
 * @param {AIPlayer} aiPlayer - The learning AI (always P1)
 * @param {GridManager} gridManager - The game state
 * @param {boolean} gradientIsolation - Enable gradient isolation
 * @returns {Promise<{ winner, p1Score, p2Score, moves, firstMove, trainingSamplesAdded }>}
 */
async function _runGameVsRandomWithIsolation(aiPlayer, gridManager, gradientIsolation) {
  // Create a lightweight random opponent
  // Note: trainOnOpponent=false for random - not worth learning from random moves
  const randomOpponent = {
    playerNumber: 2,
    getMove(gridManager) {
      const { rows, cols } = gridManager;
      const validMoves = [];
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          for (let rotation = 0; rotation < 4; rotation++) {
            if (gridManager.isValidMove(x, y, rotation)) {
              validMoves.push({ x, y, rotation });
            }
          }
        }
      }
      return validMoves.length > 0 ? validMoves[Math.floor(Math.random() * validMoves.length)] : null;
    }
  };
  return _runGameVsOpponent(aiPlayer, randomOpponent, gridManager, gradientIsolation, false);
}

async function main() {
  const opts = parseArgs();
  const projectRoot = join(__dirname, '..', '..');

  console.log('\n🏋️  Triantica AI Training Pipeline');
  console.log('='.repeat(50));
  console.log(`  Cycles:         ${opts.cycles}`);
  console.log(`  Games/cycle:    ${opts.games}`);
  console.log(`  Eval games:     ${opts.evalGames}`);
  console.log(`  Train iters:    ${opts.iterations}`);
  console.log(`  Learning rate:  ${opts.learningRate}`);
  console.log(`  Exploration:    ${opts.exploration}`);
  console.log(`  Expl. mode:     ${opts.explorationMode}`);
  console.log(`  Self-play mode: ${opts.selfPlayMode}`);
  console.log(`  Opponent mix:   ${opts.opponentMix}`);
  if (opts.opponentMix === 'mixed') {
    console.log(`  Heuristic ratio: ${opts.heuristicRatio}`);
    console.log(`  Random ratio:   ${opts.randomRatio}`);
    console.log(`  Network ratio:  ${(1 - opts.heuristicRatio - opts.randomRatio).toFixed(2)}`);
  }
  console.log(`  Cluster reward: ${opts.enableClusterReward ? 'ON (TD(λ))' : 'OFF'}`);
  console.log(`  Gradient isolation: ${opts.gradientIsolation ? 'ON (Exp 13a)' : 'OFF'}`);
  console.log(`  Replay ratio:   ${opts.replayRatio}`);
  console.log(`  Replay buffer:   ${opts.replaySize}`);
  console.log(`  Start from:     ${opts.startFrom}`);
  console.log(`  Output:         ${opts.output}`);
  if (opts.audit) {
    console.log(`  Audit mode:     ENABLED (rate=${(opts.auditRate * 100).toFixed(0)}%)`);
  }
  console.log('='.repeat(50) + '\n');

  // Set up log file
  const logsDir = join(projectRoot, 'experiments', 'logs');
  mkdirSync(logsDir, { recursive: true });
  const logPath = opts.log || join(logsDir, `training-${Date.now()}.log`);

  const logStream = {
    write: (msg) => {
      writeFileSync(logPath, msg, { flag: 'a' });
      console.log(msg.trim());
    }
  };

  logStream.write(`\n🏋️  Triantica AI Training Pipeline\n`);
  logStream.write(`Started: ${new Date().toISOString()}\n`);
  logStream.write(`Log: ${logPath}\n\n`);

  // Load or create AI player
  let networkJSON = null;
  const networkPath = join(projectRoot, opts.startFrom);

  if (existsSync(networkPath)) {
    try {
      networkJSON = JSON.parse(readFileSync(networkPath, 'utf-8'));
      console.log(`📂 Loaded pre-trained network from ${opts.startFrom}`);
    } catch (e) {
      console.log(`⚠️  Failed to load ${opts.startFrom}: ${e.message}`);
    }
  }

  const aiPlayer = new AIPlayer(1, {
    difficulty: 'medium',
    explorationFactor: opts.exploration,
    explorationMode: opts.explorationMode || 'perMove',
    selfPlayMode: opts.selfPlayMode,
    enableClusterReward: opts.enableClusterReward,
    clusterRewardFactor: opts.clusterRewardFactor,
    clusterLambda: opts.clusterLambda,
    clusterGamma: opts.clusterGamma,
    networkJSON,
  });

  await aiPlayer.initialize();
  console.log(`🧠 AI initialized (network loaded: ${aiPlayer.isNetworkLoaded})\n`);

  // Training log
  const trainingLog = {
    startedAt: new Date().toISOString(),
    options: {
      cycles: opts.cycles,
      games: opts.games,
      evalGames: opts.evalGames,
      iterations: opts.iterations,
      learningRate: opts.learningRate,
      exploration: opts.exploration,
      selfPlayMode: opts.selfPlayMode,
      opponentMix: opts.opponentMix,
      heuristicRatio: opts.heuristicRatio,
      randomRatio: opts.randomRatio,
      startFrom: opts.startFrom === 'fresh' || opts.startFrom === 'none' ? null : opts.startFrom,
      output: opts.output,
      saveEvery: opts.saveEvery,
    },
    cycles: [],
  };

  const dataDir = join(projectRoot, 'docs', 'data');
  mkdirSync(dataDir, { recursive: true });

  // Initialize game auditor if enabled
  const auditor = new GameAuditor({
    enabled: opts.audit,
    sampleRate: opts.auditRate,
    verbose: opts.audit
  });
  if (opts.audit) {
    logStream.write(`  🔍 Game auditing: ENABLED (rate=${(opts.auditRate * 100).toFixed(0)}%)`);
  }

  // Experience replay buffer
  const replayBuffer = new ReplayBuffer(opts.replaySize);
  logStream.write(`  🗃️  Replay buffer: maxSize=${opts.replaySize}, replayRatio=${opts.replayRatio}`);

  // Run training cycles
  for (let cycle = 1; cycle <= opts.cycles; cycle++) {
    const cycleStart = Date.now();
    logStream.write(`\n🔄 Cycle ${cycle}/${opts.cycles}`);
    logStream.write('-'.repeat(40));

    // Phase 1: Self-play games to generate training data
    // Supports mixed opponents: network, heuristic, random
    const opponentMix = opts.opponentMix;
    let heuristicPlayer = null;
    if (opponentMix === 'heuristic' || opponentMix === 'mixed') {
      heuristicPlayer = new HeuristicPlayer(2, { explorationFactor: 0.05 });
    }

    logStream.write(`  🎮 Running ${opts.games} self-play games (${opponentMix} mode)...`);
    const selfPlayStart = Date.now();

    aiPlayer.reset();
    const gridManager = new GridManager();

    let selfPlayWins = { 1: 0, 2: 0, draws: 0 };
    let selfPlayMoves = 0;
    let selfPlayScores = { 1: [], 2: [] }; // Array of all scores for median/variance
    let samplesGenerated = 0;
    let firstMoves = [];
    let finalBoards = []; // Array of { cell: { player, rotation } }
    let opponentTypeCounts = { network: 0, heuristic: 0, random: 0 };
    let winsByOpponent = { network: { 1: 0, 2: 0, draws: 0 }, heuristic: { 1: 0, 2: 0, draws: 0 }, random: { 1: 0, 2: 0, draws: 0 } };
    let scoresByOpponent = { network: { 1: [], 2: [] }, heuristic: { 1: [], 2: [] }, random: { 1: [], 2: [] } };

    for (let g = 0; g < opts.games; g++) {
      // Determine opponent type for this game
      let opponentType = 'network'; // default
      if (opponentMix === 'random') {
        opponentType = 'random';
      } else if (opponentMix === 'heuristic') {
        opponentType = 'heuristic';
      } else if (opponentMix === 'mixed') {
        const r = Math.random();
        if (r < opts.heuristicRatio) {
          opponentType = 'heuristic';
        } else if (r < opts.heuristicRatio + opts.randomRatio) {
          opponentType = 'random';
        } else {
          opponentType = 'network';
        }
      }
      opponentTypeCounts[opponentType]++;

      gridManager.reset();
      let result;

      if (opponentType === 'network') {
        // Standard self-play: both sides use network
        result = aiPlayer.runSelfPlayGame(gridManager, opts.gradientIsolation);
      } else if (opponentType === 'heuristic') {
        // AI (P1) vs Heuristic (P2) - learn from heuristic's winning moves if enabled
        result = await _runGameVsOpponent(aiPlayer, heuristicPlayer, gridManager, opts.gradientIsolation, opts.trainOnOpponentMoves);
      } else {
        // AI (P1) vs Random (P2) - don't learn from random moves
        result = await _runGameVsRandomWithIsolation(aiPlayer, gridManager, opts.gradientIsolation);
      }

      if (result.winner === 0) { selfPlayWins.draws++; winsByOpponent[opponentType].draws++; }
      else { selfPlayWins[result.winner]++; winsByOpponent[opponentType][result.winner]++; }
      selfPlayMoves += result.moves;
      selfPlayScores[1].push(result.p1Score || 0);
      selfPlayScores[2].push(result.p2Score || 0);
      scoresByOpponent[opponentType][1].push(result.p1Score || 0);
      scoresByOpponent[opponentType][2].push(result.p2Score || 0);
      samplesGenerated += result.trainingSamplesAdded || result.moves;

      // Audit this game if sampled
      if (auditor.shouldAudit() && result.gameHistory) {
        auditor.auditSelfPlayGame(result.gameHistory, gridManager, opponentType);
      }

      // Track first move for diversity
      if (result.firstMove) {
        firstMoves.push(`${result.firstMove.x},${result.firstMove.y},r${result.firstMove.rotation}`);
      }

      // Track final board state for diversity
      const boardState = [];
      for (let y = 0; y < gridManager.rows; y++) {
        for (let x = 0; x < gridManager.cols; x++) {
          const cell = gridManager.grid[y][x];
          const t1 = cell.triangles[0];
          const t2 = cell.triangles[1];
          boardState.push({
            t1: t1?.player || 0,
            t2: t2?.player || 0
          });
        }
      }
      finalBoards.push(boardState);

      if ((g + 1) % 10 === 0) {
        logStream.write(`    ${g + 1}/${opts.games} games done`);
      }
    }

    const selfPlayDuration = Date.now() - selfPlayStart;
    const openingDiversity = computeOpeningDiversity(firstMoves);
    const boardDiversity = computeBoardDiversity(finalBoards);

    // Compute score metrics for self-play
    const p1Median = computeMedian(selfPlayScores[1]);
    const p2Median = computeMedian(selfPlayScores[2]);
    const p1Variance = computeVariance(selfPlayScores[1]);
    const p2Variance = computeVariance(selfPlayScores[2]);
    const p1Min = computeMin(selfPlayScores[1]);
    const p2Min = computeMin(selfPlayScores[2]);
    const p1Max = computeMax(selfPlayScores[1]);
    const p2Max = computeMax(selfPlayScores[2]);

    logStream.write(`  ✅ Self-play complete: ${opts.games} games, ${samplesGenerated} samples, ${selfPlayDuration}ms`);
    if (opponentMix !== 'network' && opponentMix !== 'random') {
      logStream.write(`     Opponent breakdown: Network=${opponentTypeCounts.network}, Heuristic=${opponentTypeCounts.heuristic}, Random=${opponentTypeCounts.random}`);
      // Per-opponent win rates
      for (const [opp, counts] of Object.entries(winsByOpponent)) {
        const total = counts[1] + counts[2] + counts.draws;
        if (total > 0) {
          const p1Rate = ((counts[1] / total) * 100).toFixed(0);
          const p2Rate = ((counts[2] / total) * 100).toFixed(0);
          logStream.write(`     ${opp}: ${total} games → P1=${counts[1]} (${p1Rate}%) P2=${counts[2]} (${p2Rate}%) D=${counts.draws}`);
        }
      }
    }

    // Phase 2: Train network with experience replay
    const newTrainingData = aiPlayer.getTrainingData();

    // Add new samples to replay buffer
    replayBuffer.add(newTrainingData);

    // Mix new data with replay buffer samples
    let trainingData;
    const replaySamples = replayBuffer.sample(Math.floor(newTrainingData.length * opts.replayRatio));
    trainingData = [...replaySamples, ...newTrainingData];
    // Shuffle to mix old and new
    for (let i = trainingData.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [trainingData[i], trainingData[j]] = [trainingData[j], trainingData[i]];
    }

    logStream.write(`  📚 Training network (${opts.iterations} iterations, lr=${opts.learningRate}, ${trainingData.length} samples [${replaySamples.length} replay + ${newTrainingData.length} new])...`);
    const trainStart = Date.now();

    // Set the mixed training data and train
    aiPlayer.setTrainingData(trainingData);
    const trainResult = aiPlayer.trainNetwork(opts.iterations, opts.learningRate);

    const trainDuration = Date.now() - trainStart;

    if (trainResult) {
      const errorDelta = trainResult.initialError - trainResult.error;
      const convRate = trainResult.errorReductionHalf;
      const convNote = convRate !== null ? ` | ${(convRate * 100).toFixed(0)}% error drop by half iters` : '';
      logStream.write(`  ✅ Training complete: error=${trainResult.error?.toFixed(6) ?? 'N/A'} (was ${trainResult.initialError?.toFixed(6) ?? 'N/A'}, Δ=${errorDelta?.toFixed(6) ?? 'N/A'})${convNote}, ${trainDuration}ms`);

      // Log error convergence curve
      if (trainResult.errorSnapshots && trainResult.errorSnapshots.length > 1) {
        const curve = trainResult.errorSnapshots.map(s => `${s.iteration}:${s.error.toFixed(4)}`).join(' → ');
        logStream.write(`  📉 Error curve: ${curve}`);
      }
    } else {
      logStream.write(`  ⚠️  No training data available`);
    }

    // Phase 3: Evaluate vs Random
    logStream.write(`  📊 Evaluating vs Random (${opts.evalGames} games)...`);
    const evalStart = Date.now();

    const evalAI = new AIPlayer(1, {
      difficulty: 'medium',
      explorationFactor: 0, // Pure network play for evaluation
      explorationMode: 'perMove', // Per-move mode (no random games)
      networkJSON: aiPlayer.exportNetwork(),
    });
    await evalAI.initialize();

    const session = new GameSession();
    session.setPlayers(evalAI, new RandomPlayer(2));
    const evalResults = await session.runGames(opts.evalGames);

    // Audit sampled eval games
    if (opts.audit) {
      for (const gameResult of evalResults.games) {
        if (auditor.shouldAudit() && gameResult.moveHistory && gameResult.moveHistory.length > 0) {
          // Convert GameSession moveHistory to auditor format
          const auditHistory = gameResult.moveHistory.map(m => ({
            player: m.player,
            move: { x: m.x, y: m.y, rotation: m.rotation },
            claimed: m.claims,
          }));
          auditor.auditSelfPlayGame(auditHistory, new GridManager(10, 10), 'eval-vs-random');
        }
      }
    }

    const evalDuration = Date.now() - evalStart;
    const aiWinRate = evalResults.wins.ai / opts.evalGames;

    // Compute eval score metrics
    const evalP1Median = computeMedian(evalResults.scores?.[1] || [0]);
    const evalP2Median = computeMedian(evalResults.scores?.[2] || [0]);
    const evalP1Variance = computeVariance(evalResults.scores?.[1] || [0]);
    const evalP2Variance = computeVariance(evalResults.scores?.[2] || [0]);
    const evalGap = evalP1Median - evalP2Median;

    logStream.write(`  📈 AI Win Rate: ${(aiWinRate * 100).toFixed(1)}% (${evalResults.wins.ai}W / ${evalResults.wins.random}L / ${evalResults.wins.draws}D)`);
    logStream.write(`     Scores: AI med=${evalP1Median.toFixed(0)} (σ2=${evalP1Variance.toFixed(0)}), Random med=${evalP2Median.toFixed(0)} (σ2=${evalP2Variance.toFixed(0)}), gap=+${evalGap.toFixed(0)}`);
    logStream.write(`     Avg moves: ${evalResults.avgMoves.toFixed(1)}, Duration: ${evalDuration}ms`);

    // Record cycle
    const cycleData = {
      cycle,
      selfPlay: {
        games: opts.games,
        wins: selfPlayWins,
        winsByOpponent,
        opponentTypeCounts,
        avgMoves: selfPlayMoves / opts.games,
        scores: {
          p1: { median: p1Median, variance: p1Variance, min: p1Min, max: p1Max },
          p2: { median: p2Median, variance: p2Variance, min: p2Min, max: p2Max },
        },
        scoresByOpponent: {
          network: {
            p1: { median: computeMedian(scoresByOpponent.network[1]), max: Math.max(...(scoresByOpponent.network[1].length ? scoresByOpponent.network[1] : [0])) },
            p2: { median: computeMedian(scoresByOpponent.network[2]), max: Math.max(...(scoresByOpponent.network[2].length ? scoresByOpponent.network[2] : [0])) },
          },
          heuristic: {
            p1: { median: computeMedian(scoresByOpponent.heuristic[1]), max: Math.max(...(scoresByOpponent.heuristic[1].length ? scoresByOpponent.heuristic[1] : [0])) },
            p2: { median: computeMedian(scoresByOpponent.heuristic[2]), max: Math.max(...(scoresByOpponent.heuristic[2].length ? scoresByOpponent.heuristic[2] : [0])) },
          },
          random: {
            p1: { median: computeMedian(scoresByOpponent.random[1]), max: Math.max(...(scoresByOpponent.random[1].length ? scoresByOpponent.random[1] : [0])) },
            p2: { median: computeMedian(scoresByOpponent.random[2]), max: Math.max(...(scoresByOpponent.random[2].length ? scoresByOpponent.random[2] : [0])) },
          },
        },
        openingDiversity,
        boardDiversity,
        samplesGenerated,
        durationMs: selfPlayDuration,
      },
      training: {
        iterations: opts.iterations,
        learningRate: opts.learningRate,
        samplesUsed: trainingData.length,
        replaySamplesUsed: replaySamples.length,
        newSamplesUsed: newTrainingData.length,
        replayBufferSize: replayBuffer.size,
        initialError: trainResult?.initialError ?? null,
        finalError: trainResult?.error ?? null,
        errorSnapshots: trainResult?.errorSnapshots ?? [],
        errorReductionHalf: trainResult?.errorReductionHalf ?? null,
        durationMs: trainDuration,
      },
      evaluation: {
        games: opts.evalGames,
        opponent: 'random',
        aiWins: evalResults.wins.ai,
        opponentWins: evalResults.wins.random,
        draws: evalResults.wins.draws,
        scores: {
          ai: { median: evalP1Median, variance: evalP1Variance },
          opponent: { median: evalP2Median, variance: evalP2Variance },
          gap: evalGap,
        },
        avgScores: evalResults.avgScores,
        aiWinRate,
        avgMoves: evalResults.avgMoves,
        durationMs: evalDuration,
      },
      totalDurationMs: Date.now() - cycleStart,
    };

    trainingLog.cycles.push(cycleData);

    // Per-cycle summary line for easy log monitoring
    const cycleDuration = ((Date.now() - cycleStart) / 1000).toFixed(0);
    const p1ScoreStr = `med=${p1Median.toFixed(0)} (σ2=${p1Variance.toFixed(0)}, ${p1Min}-${p1Max})`;
    const p2ScoreStr = `med=${p2Median.toFixed(0)} (σ2=${p2Variance.toFixed(0)}, ${p2Min}-${p2Max})`;
    const oppMixStr = (opts.opponentMix !== 'network' && opts.opponentMix !== 'random')
      ? ` (H=${opponentTypeCounts.heuristic},R=${opponentTypeCounts.random})`
      : '';
    const oppWinStr = (opts.opponentMix !== 'network' && opts.opponentMix !== 'random')
      ? ` | Net: ${winsByOpponent.network[1]}/${winsByOpponent.network[2]}/${winsByOpponent.network.draws} Heur: ${winsByOpponent.heuristic[1]}/${winsByOpponent.heuristic[2]}/${winsByOpponent.heuristic.draws} Rand: ${winsByOpponent.random[1]}/${winsByOpponent.random[2]}/${winsByOpponent.random.draws}`
      : '';
    logStream.write(`\n  📋 Cycle ${cycle} Summary | Self-play: P1=${selfPlayWins[1]} P2=${selfPlayWins[2]} D=${selfPlayWins.draws}${oppMixStr} | P1 ${p1ScoreStr} | P2 ${p2ScoreStr} | Train: ${trainResult?.initialError?.toFixed(6) ?? 'N/A'} → ${trainResult?.error?.toFixed(6) ?? 'N/A'} (${replaySamples.length} replay + ${newTrainingData.length} new) | Eval: ${(aiWinRate * 100).toFixed(1)}% (AI med=${evalP1Median.toFixed(0)}, Rand med=${evalP2Median.toFixed(0)}, gap=+${evalGap.toFixed(0)}) | Diversity: Open=${openingDiversity.toFixed(2)} Board=${boardDiversity.toFixed(2)} | Buf: ${replayBuffer.size} | ${cycleDuration}s${oppWinStr}\n`);

    // Save checkpoint
    if (cycle % opts.saveEvery === 0 || cycle === opts.cycles) {
      const checkpointPath = join(dataDir, `checkpoint-cycle-${cycle}.json`);
      const networkData = aiPlayer.exportNetwork();
      if (networkData) {
        writeFileSync(checkpointPath, JSON.stringify(networkData, null, 2));
        logStream.write(`  💾 Checkpoint saved: ${checkpointPath}`);
      }
    }

    // Early stopping if AI is doing well
    if (aiWinRate >= 0.95) {
      logStream.write(`\n🎯 AI reached ${(aiWinRate * 100).toFixed(1)}% win rate - training target met!`);
      break;
    }
  }

  // Save final network
  const finalNetwork = aiPlayer.exportNetwork();
  if (finalNetwork) {
    const outputPath = join(projectRoot, opts.output);
    writeFileSync(outputPath, JSON.stringify(finalNetwork));
    logStream.write(`\n💾 Final network saved to ${opts.output}`);
  }

  // Save training log JSON
  trainingLog.completedAt = new Date().toISOString();
  const metricsLogPath = join(dataDir, `training-${Date.now()}.json`);
  writeFileSync(metricsLogPath, JSON.stringify(trainingLog, null, 2));
  logStream.write(`📋 Training metrics saved to ${metricsLogPath}`);

  // Summary
  logStream.write('\n' + '='.repeat(50));
  logStream.write('📊 Training Summary');
  logStream.write('='.repeat(50));

  if (trainingLog.cycles.length > 1) {
    const first = trainingLog.cycles[0].evaluation.aiWinRate;
    const last = trainingLog.cycles[trainingLog.cycles.length - 1].evaluation.aiWinRate;
    logStream.write(`  Starting win rate: ${(first * 100).toFixed(1)}%`);
    logStream.write(`  Final win rate:    ${(last * 100).toFixed(1)}%`);
    logStream.write(`  Improvement:      ${((last - first) * 100).toFixed(1)}%`);
  }

  const totalDuration = trainingLog.cycles.reduce((sum, c) => sum + c.totalDurationMs, 0);
  logStream.write(`  Total time:        ${(totalDuration / 1000).toFixed(1)}s`);
  logStream.write(`  Cycles completed:  ${trainingLog.cycles.length}`);
  logStream.write('='.repeat(50) + '\n');

  // Print audit summary if enabled
  if (opts.audit) {
    auditor.printSummary();
  }
}

main().catch(e => {
  console.error('Training failed:', e);
  process.exit(1);
});
