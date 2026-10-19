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
    let selfPlayScores = { 1: 0, 2: 0 };
    let samplesGenerated = 0;

    for (let g = 0; g < opts.games; g++) {
      gridManager.reset();
      const result = aiPlayer.runSelfPlayGame(gridManager);
      
      if (result.winner === 0) selfPlayWins.draws++;
      else selfPlayWins[result.winner]++;
      selfPlayMoves += result.moves;
      selfPlayScores[1] += result.p1Score || 0;
      selfPlayScores[2] += result.p2Score || 0;
      samplesGenerated += result.trainingSamplesAdded;

      if ((g + 1) % 10 === 0) {
        logStream.write(`    ${g + 1}/${opts.games} games done`);
      }
    }

    const selfPlayDuration = Date.now() - selfPlayStart;
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
    
    logStream.write(`  📈 AI Win Rate: ${(aiWinRate * 100).toFixed(1)}% (${evalResults.wins[1]}W / ${evalResults.wins[2]}L / ${evalResults.wins.draws}D)`);
    logStream.write(`     Avg scores: AI=${(evalResults.avgScores?.[1] ?? 0).toFixed(1)} Random=${(evalResults.avgScores?.[2] ?? 0).toFixed(1)}`);
    logStream.write(`     Avg moves: ${evalResults.avgMoves.toFixed(1)}, Duration: ${evalDuration}ms`);

    // Record cycle
    const cycleData = {
      cycle,
      selfPlay: {
        games: opts.games,
        wins: selfPlayWins,
        avgMoves: selfPlayMoves / opts.games,
        avgScores: { 1: selfPlayScores[1] / opts.games, 2: selfPlayScores[2] / opts.games },
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
    const avgSelfScores = { 1: (selfPlayScores[1] / opts.games).toFixed(1), 2: (selfPlayScores[2] / opts.games).toFixed(1) };
    logStream.write(`\n  📋 Cycle ${cycle} Summary | Self-play: P1=${selfPlayWins[1]} P2=${selfPlayWins[2]} D=${selfPlayWins.draws} (scores: ${avgSelfScores[1]}-${avgSelfScores[2]}) | Train: ${trainResult?.initialError?.toFixed(6) ?? 'N/A'} → ${trainResult?.error?.toFixed(6) ?? 'N/A'} (${replaySamples.length} replay + ${newTrainingData.length} new) | Eval: ${(aiWinRate * 100).toFixed(1)}% (scores: ${(evalResults.avgScores?.[1] ?? 0).toFixed(1)}-${(evalResults.avgScores?.[2] ?? 0).toFixed(1)}) | Buf: ${replayBuffer.size} | ${cycleDuration}s\n`);

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
