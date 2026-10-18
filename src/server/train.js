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
 *   --start-from FILE  Load pre-trained network (default: trainedAI_network.json)
 *   --output FILE      Save trained network (default: trainedAI_network_v2.json)
 *   --save-every N     Save checkpoint every N cycles (default: 5)
 */

import { GridManager } from '../core/GridManager.js';
import { GameSession } from '../core/GameSession.js';
import { RandomPlayer } from '../players/RandomPlayer.js';
import { AIPlayer } from '../players/AIPlayer.js';
import { GameLogger } from '../data/GameLogger.js';
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
    startFrom: 'trainedAI_network.json',
    output: 'trainedAI_network_v2.json',
    saveEvery: 5,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith('--')) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const val = args[++i];
      if (key === 'learningRate' || key === 'exploration') {
        opts[key] = parseFloat(val);
      } else if (key === 'startFrom' || key === 'output') {
        opts[key] = val;
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
  console.log(`  Start from:     ${opts.startFrom}`);
  console.log(`  Output:         ${opts.output}`);
  console.log('='.repeat(50) + '\n');

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
    networkJSON,
  });

  await aiPlayer.initialize();
  console.log(`🧠 AI initialized (network loaded: ${aiPlayer.isNetworkLoaded})\n`);

  // Training log
  const trainingLog = {
    startedAt: new Date().toISOString(),
    options: opts,
    cycles: [],
  };

  const dataDir = join(projectRoot, 'experiments', 'data');
  mkdirSync(dataDir, { recursive: true });

  // Run training cycles
  for (let cycle = 1; cycle <= opts.cycles; cycle++) {
    const cycleStart = Date.now();
    console.log(`\n🔄 Cycle ${cycle}/${opts.cycles}`);
    console.log('-'.repeat(40));

    // Phase 1: Self-play games to generate training data
    console.log(`  🎮 Running ${opts.games} self-play games...`);
    const selfPlayStart = Date.now();
    
    aiPlayer.reset();
    const gridManager = new GridManager();
    
    let selfPlayWins = { 1: 0, 2: 0, draws: 0 };
    let selfPlayMoves = 0;
    let samplesGenerated = 0;

    for (let g = 0; g < opts.games; g++) {
      gridManager.reset();
      const result = aiPlayer.runSelfPlayGame(gridManager);
      
      if (result.winner === 0) selfPlayWins.draws++;
      else selfPlayWins[result.winner]++;
      selfPlayMoves += result.moves;
      samplesGenerated += result.trainingSamplesAdded;

      if ((g + 1) % 10 === 0) {
        process.stdout.write(`    ${g + 1}/${opts.games} games done\r`);
      }
    }

    const selfPlayDuration = Date.now() - selfPlayStart;
    console.log(`  ✅ Self-play complete: ${opts.games} games, ${samplesGenerated} samples, ${selfPlayDuration}ms`);

    // Phase 2: Train network
    console.log(`  📚 Training network (${opts.iterations} iterations, lr=${opts.learningRate})...`);
    const trainStart = Date.now();
    
    const trainingData = aiPlayer.getTrainingData();
    const trainResult = aiPlayer.trainNetwork(opts.iterations, opts.learningRate);
    
    const trainDuration = Date.now() - trainStart;
    
    if (trainResult) {
      console.log(`  ✅ Training complete: error=${trainResult.error?.toFixed(6) ?? 'N/A'}, ${trainDuration}ms`);
    } else {
      console.log(`  ⚠️  No training data available`);
    }

    // Phase 3: Evaluate vs Random
    console.log(`  📊 Evaluating vs Random (${opts.evalGames} games)...`);
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
    
    console.log(`  📈 AI Win Rate: ${(aiWinRate * 100).toFixed(1)}% (${evalResults.wins[1]}W / ${evalResults.wins[2]}L / ${evalResults.wins.draws}D)`);
    console.log(`     Avg moves: ${evalResults.avgMoves.toFixed(1)}, Duration: ${evalDuration}ms`);

    // Record cycle
    const cycleData = {
      cycle,
      selfPlay: {
        games: opts.games,
        wins: selfPlayWins,
        avgMoves: selfPlayMoves / opts.games,
        samplesGenerated,
        durationMs: selfPlayDuration,
      },
      training: {
        iterations: opts.iterations,
        learningRate: opts.learningRate,
        samplesUsed: trainingData.length,
        error: trainResult?.error ?? null,
        durationMs: trainDuration,
      },
      evaluation: {
        games: opts.evalGames,
        aiWins: evalResults.wins[1],
        randomWins: evalResults.wins[2],
        draws: evalResults.wins.draws,
        aiWinRate,
        avgMoves: evalResults.avgMoves,
        durationMs: evalDuration,
      },
      totalDurationMs: Date.now() - cycleStart,
    };

    trainingLog.cycles.push(cycleData);

    // Save checkpoint
    if (cycle % opts.saveEvery === 0 || cycle === opts.cycles) {
      const checkpointPath = join(dataDir, `checkpoint-cycle-${cycle}.json`);
      const networkData = aiPlayer.exportNetwork();
      if (networkData) {
        writeFileSync(checkpointPath, JSON.stringify(networkData, null, 2));
        console.log(`  💾 Checkpoint saved: ${checkpointPath}`);
      }
    }

    // Early stopping if AI is doing well
    if (aiWinRate >= 0.80) {
      console.log(`\n🎯 AI reached ${(aiWinRate * 100).toFixed(1)}% win rate — training target met!`);
      break;
    }
  }

  // Save final network
  const finalNetwork = aiPlayer.exportNetwork();
  if (finalNetwork) {
    const outputPath = join(projectRoot, opts.output);
    writeFileSync(outputPath, JSON.stringify(finalNetwork));
    console.log(`\n💾 Final network saved to ${opts.output}`);
  }

  // Save training log
  trainingLog.completedAt = new Date().toISOString();
  const logPath = join(dataDir, `training-${Date.now()}.json`);
  writeFileSync(logPath, JSON.stringify(trainingLog, null, 2));
  console.log(`📋 Training log saved to ${logPath}`);

  // Summary
  console.log('\n' + '='.repeat(50));
  console.log('📊 Training Summary');
  console.log('='.repeat(50));
  
  if (trainingLog.cycles.length > 1) {
    const first = trainingLog.cycles[0].evaluation.aiWinRate;
    const last = trainingLog.cycles[trainingLog.cycles.length - 1].evaluation.aiWinRate;
    console.log(`  Starting win rate: ${(first * 100).toFixed(1)}%`);
    console.log(`  Final win rate:    ${(last * 100).toFixed(1)}%`);
    console.log(`  Improvement:      ${((last - first) * 100).toFixed(1)}%`);
  }

  const totalDuration = trainingLog.cycles.reduce((sum, c) => sum + c.totalDurationMs, 0);
  console.log(`  Total time:        ${(totalDuration / 1000).toFixed(1)}s`);
  console.log(`  Cycles completed:  ${trainingLog.cycles.length}`);
  console.log('='.repeat(50) + '\n');
}

main().catch(e => {
  console.error('Training failed:', e);
  process.exit(1);
});
