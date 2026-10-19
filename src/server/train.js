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
 *   --start-from FILE  Load pre-trained network (default: trainedAI_network.json)
 *   --output FILE      Save trained network (default: trainedAI_network_v2.json)
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
 *   --start-from FILE  Load pre-trained network (default: trainedAI_network.json)
 *   --output FILE      Save trained network (default: trainedAI_network_v2.json)
 *   --save-every N     Save checkpoint every N cycles (default: 5)
 *   --log FILE         Log progress to file (default: experiments/logs/training-TIMESTAMP.log)
 */

import { GridManager } from '../core/GridManager.js';
import { GameSession } from '../core/GameSession.js';
import { RandomPlayer } from '../players/RandomPlayer.js';
import { AIPlayer } from '../players/AIPlayer.js';
import { GameLogger } from '../data/GameLogger.js';
import { ReplayBuffer } from '../data/ReplayBuffer.js';
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
    selfPlayMode: 'random',
    startFrom: 'trainedAI_network.json',
    output: 'trainedAI_network_v2.json',
    saveEvery: 5,
    log: null,
    replayRatio: 0.5, // fraction of training data from replay buffer
    replaySize: 50000, // max replay buffer size
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith('--')) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const val = args[++i];
      if (key === 'learningRate' || key === 'exploration' || key === 'replayRatio') {
        opts[key] = parseFloat(val);
      } else if (key === 'startFrom' || key === 'output' || key === 'selfPlayMode' || key === 'log') {
        opts[key] = val;
      } else if (key === 'replaySize') {
        opts[key] = parseInt(val, 10);
      } else {
        opts[key] = parseInt(val, 10);
      }
    }
  }
  return opts;
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
  console.log(`  Self-play mode: ${opts.selfPlayMode}`);
  console.log(`  Replay ratio:   ${opts.replayRatio}`);
  console.log(`  Replay buffer:   ${opts.replaySize}`);
  console.log(`  Start from:     ${opts.startFrom}`);
  console.log(`  Output:         ${opts.output}`);
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
    selfPlayMode: opts.selfPlayMode,
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
      startFrom: opts.startFrom,
      output: opts.output,
      saveEvery: opts.saveEvery,
    },
    cycles: [],
  };

  const dataDir = join(projectRoot, 'experiments', 'data');
  mkdirSync(dataDir, { recursive: true });

  // Experience replay buffer
  const replayBuffer = new ReplayBuffer(opts.replaySize);
  logStream.write(`  🗃️  Replay buffer: maxSize=${opts.replaySize}, replayRatio=${opts.replayRatio}`);

  // Run training cycles
  for (let cycle = 1; cycle <= opts.cycles; cycle++) {
    const cycleStart = Date.now();
    logStream.write(`\n🔄 Cycle ${cycle}/${opts.cycles}`);
    logStream.write('-'.repeat(40));

    // Phase 1: Self-play games to generate training data
    logStream.write(`  🎮 Running ${opts.games} self-play games...`);
    const selfPlayStart = Date.now();
    
    aiPlayer.reset();
    const gridManager = new GridManager();
    
    let selfPlayWins = { 1: 0, 2: 0, draws: 0 };
    let selfPlayMoves = 0;
    let selfPlayScores = { 1: [], 2: [] }; // Array of all scores for median/variance
    let samplesGenerated = 0;
    let firstMoves = [];
    let finalBoards = []; // Array of { cell: { player, rotation } }

    for (let g = 0; g < opts.games; g++) {
      gridManager.reset();
      const result = aiPlayer.runSelfPlayGame(gridManager);
      
      if (result.winner === 0) selfPlayWins.draws++;
      else selfPlayWins[result.winner]++;
      selfPlayMoves += result.moves;
      selfPlayScores[1].push(result.p1Score || 0);
      selfPlayScores[2].push(result.p2Score || 0);
      samplesGenerated += result.trainingSamplesAdded;
      
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
      explorationFactor: 0.05, // Lower exploration for evaluation
      networkJSON: aiPlayer.exportNetwork(),
    });
    await evalAI.initialize();
    
    const session = new GameSession();
    session.setPlayers(evalAI, new RandomPlayer(2));
    const evalResults = await session.runGames(opts.evalGames);
    
    const evalDuration = Date.now() - evalStart;
    const aiWinRate = evalResults.wins[1] / opts.evalGames;
    
    // Compute eval score metrics
    const evalP1Median = computeMedian(evalResults.scores?.[1] || [0]);
    const evalP2Median = computeMedian(evalResults.scores?.[2] || [0]);
    const evalP1Variance = computeVariance(evalResults.scores?.[1] || [0]);
    const evalP2Variance = computeVariance(evalResults.scores?.[2] || [0]);
    const evalGap = evalP1Median - evalP2Median;
    
    logStream.write(`  📈 AI Win Rate: ${(aiWinRate * 100).toFixed(1)}% (${evalResults.wins[1]}W / ${evalResults.wins[2]}L / ${evalResults.wins.draws}D)`);
    logStream.write(`     Scores: AI med=${evalP1Median.toFixed(0)} (σ²=${evalP1Variance.toFixed(0)}), Random med=${evalP2Median.toFixed(0)} (σ²=${evalP2Variance.toFixed(0)}), gap=+${evalGap.toFixed(0)}`);
    logStream.write(`     Avg moves: ${evalResults.avgMoves.toFixed(1)}, Duration: ${evalDuration}ms`);

    // Record cycle
    const cycleData = {
      cycle,
      selfPlay: {
        games: opts.games,
        wins: selfPlayWins,
        avgMoves: selfPlayMoves / opts.games,
        scores: {
          p1: { median: p1Median, variance: p1Variance, min: p1Min, max: p1Max },
          p2: { median: p2Median, variance: p2Variance, min: p2Min, max: p2Max },
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
        aiWins: evalResults.wins[1],
        opponentWins: evalResults.wins[2],
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
    const p1ScoreStr = `med=${p1Median.toFixed(0)} (σ²=${p1Variance.toFixed(0)}, ${p1Min}-${p1Max})`;
    const p2ScoreStr = `med=${p2Median.toFixed(0)} (σ²=${p2Variance.toFixed(0)}, ${p2Min}-${p2Max})`;
    logStream.write(`\n  📋 Cycle ${cycle} Summary | Self-play: P1=${selfPlayWins[1]} P2=${selfPlayWins[2]} D=${selfPlayWins.draws} | P1 ${p1ScoreStr} | P2 ${p2ScoreStr} | Train: ${trainResult?.initialError?.toFixed(6) ?? 'N/A'} → ${trainResult?.error?.toFixed(6) ?? 'N/A'} (${replaySamples.length} replay + ${newTrainingData.length} new) | Eval: ${(aiWinRate * 100).toFixed(1)}% (AI med=${evalP1Median.toFixed(0)}, Rand med=${evalP2Median.toFixed(0)}, gap=+${evalGap.toFixed(0)}) | Diversity: Open=${openingDiversity.toFixed(2)} Board=${boardDiversity.toFixed(2)} | Buf: ${replayBuffer.size} | ${cycleDuration}s\n`);

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
    if (aiWinRate >= 0.80) {
      logStream.write(`\n🎯 AI reached ${(aiWinRate * 100).toFixed(1)}% win rate — training target met!`);
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
}

main().catch(e => {
  console.error('Training failed:', e);
  process.exit(1);
});
