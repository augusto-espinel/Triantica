#!/usr/bin/env node
/**
 * Training Trajectory Analyzer
 * 
 * Parses training logs to analyze intra-cycle learning dynamics:
 * - Plateau speed (iterations to reach 90% of final error)
 * - Plateau level (final training error)
 * - Drop rate (error reduction per 50 iterations)
 * - Cross-cycle trends
 * 
 * Usage:
 *   node src/server/analyzeTrajectories.js [logFile]
 *   node src/server/analyzeTrajectories.js  # reads most recent log
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Parse training log to extract cycle-level data
function parseTrainingLog(logContent) {
  const cycles = [];
  const lines = logContent.split('\n');
  
  let currentCycle = null;
  let currentTrajectory = [];
  
  for (const line of lines) {
    // Cycle start
    const cycleMatch = line.match(/Cycle (\d+)\/\d+/);
    if (cycleMatch) {
      if (currentCycle !== null && currentTrajectory.length > 0) {
        cycles.push({
          cycle: currentCycle,
          trajectory: currentTrajectory
        });
      }
      currentCycle = parseInt(cycleMatch[1]);
      currentTrajectory = [];
      continue;
    }
    
    // Training iteration
    const iterMatch = line.match(/iterations: (\d+), training error: ([\d.]+)/);
    if (iterMatch && currentCycle !== null) {
      currentTrajectory.push({
        iteration: parseInt(iterMatch[1]),
        error: parseFloat(iterMatch[2])
      });
    }
    
    // Final training error
    const finalMatch = line.match(/Training complete: error=([\d.]+)/);
    if (finalMatch && currentCycle !== null) {
      // Already captured via iteration logs
    }
    
    // Win rate
    const winMatch = line.match(/AI Win Rate: ([\d.]+)%/);
    if (winMatch && currentCycle !== null) {
      if (cycles.length > 0 && cycles[cycles.length - 1].cycle === currentCycle) {
        cycles[cycles.length - 1].winRate = parseFloat(winMatch[1]);
      }
    }
  }
  
  // Don't forget last cycle
  if (currentCycle !== null && currentTrajectory.length > 0) {
    cycles.push({
      cycle: currentCycle,
      trajectory: currentTrajectory,
      winRate: null // may still be running
    });
  }
  
  return cycles;
}

// Compute trajectory statistics
function analyzeTrajectory(trajectory) {
  if (trajectory.length < 2) return null;
  
  const firstError = trajectory[0].error;
  const lastError = trajectory[trajectory.length - 1].error;
  
  // Total drop
  const totalDrop = firstError - lastError;
  
  // Drop rates (error reduction per 50 iterations)
  const dropRates = [];
  for (let i = 1; i < trajectory.length; i++) {
    const errorDrop = trajectory[i - 1].error - trajectory[i].error;
    const iterDiff = trajectory[i].iteration - trajectory[i - 1].iteration;
    dropRates.push({
      from: trajectory[i - 1].iteration,
      to: trajectory[i].iteration,
      rate: errorDrop / iterDiff * 50 // normalized to per-50-iter
    });
  }
  
  // Plateau speed: iterations to reach 90% of total drop
  const targetError = firstError - (totalDrop * 0.9);
  let plateauIter = trajectory[trajectory.length - 1].iteration;
  for (const point of trajectory) {
    if (point.error <= targetError) {
      plateauIter = point.iteration;
      break;
    }
  }
  
  // Plateau level: average of last 3 points
  const lastN = trajectory.slice(-3);
  const plateauLevel = lastN.reduce((sum, p) => sum + p.error, 0) / lastN.length;
  
  // Recent drop rate: last segment
  const recentDropRate = dropRates.length > 0 ? dropRates[dropRates.length - 1].rate : 0;
  
  return {
    firstError,
    lastError,
    totalDrop,
    plateauIter,
    plateauLevel,
    recentDropRate,
    dropRates,
    iterations: trajectory.length * 50
  };
}

// Main analysis
function analyzeCycles(cycles) {
  const results = [];
  
  for (const cycle of cycles) {
    const stats = analyzeTrajectory(cycle.trajectory);
    if (stats) {
      results.push({
        cycle: cycle.cycle,
        winRate: cycle.winRate,
        ...stats
      });
    }
  }
  
  return results;
}

// Compute cross-cycle trends
function computeTrends(results) {
  if (results.length < 2) return null;
  
  // Trends over cycles
  const plateauLevels = results.map(r => r.plateauLevel);
  const plateauIters = results.map(r => r.plateauIter);
  const winRates = results.filter(r => r.winRate !== null).map(r => r.winRate);
  
  return {
    plateauLevelTrend: plateauLevels[plateauLevels.length - 1] - plateauLevels[0],
    plateauIterTrend: plateauIters[plateauIters.length - 1] - plateauIters[0],
    winRateTrend: winRates.length >= 2 ? winRates[winRates.length - 1] - winRates[0] : null,
    avgPlateauLevel: plateauLevels.reduce((a, b) => a + b, 0) / plateauLevels.length,
    avgPlateauIter: plateauIters.reduce((a, b) => a + b, 0) / plateauIters.length
  };
}

// Print results
function printResults(results, trends) {
  console.log('\n' + '='.repeat(70));
  console.log('📊 TRAINING TRAJECTORY ANALYSIS');
  console.log('='.repeat(70));
  
  console.log('\n📈 Per-Cycle Statistics:\n');
  console.log('Cycle | WinRate | First Err | Last Err | Plateau@ | Plateau Lvl | Recent Drop');
  console.log('------|---------|-----------|----------|----------|-------------|------------');
  
  for (const r of results) {
    const winRate = r.winRate !== null ? `${r.winRate.toFixed(1)}%` : '---';
    console.log(
      `  ${r.cycle.toString().padStart(2)} | ${winRate.padStart(7)} | ` +
      `${r.firstError.toFixed(6)} | ${r.lastError.toFixed(6)} | ` +
      `${r.plateauIter.toString().padStart(8)} | ${r.plateauLevel.toFixed(6)} | ` +
      `${r.recentDropRate.toFixed(6)}`
    );
  }
  
  if (trends) {
    console.log('\n📉 Cross-Cycle Trends:\n');
    console.log(`  Plateau Level Change: ${trends.plateauLevelTrend > 0 ? '+' : ''}${trends.plateauLevelTrend.toFixed(6)}`);
    console.log(`  Plateau Speed Change:  ${trends.plateauIterTrend > 0 ? '+' : ''}${trends.plateauIterTrend} iterations`);
    if (trends.winRateTrend !== null) {
      console.log(`  Win Rate Change:       ${trends.winRateTrend > 0 ? '+' : ''}${trends.winRateTrend.toFixed(1)}%`);
    }
    console.log(`\n  Avg Plateau Level: ${trends.avgPlateauLevel.toFixed(6)}`);
    console.log(`  Avg Plateau Speed: ${trends.avgPlateauIter.toFixed(0)} iterations`);
  }
  
  console.log('\n' + '='.repeat(70));
  
  // Interpretation
  console.log('\n🧠 Interpretation:\n');
  
  if (results.length === 0) {
    console.log('  No complete cycles found yet.');
  } else {
    const lastResult = results[results.length - 1];
    
    // Plateau level interpretation
    if (lastResult.plateauLevel > 0.005) {
      console.log('  ⚠️  High plateau error (>0.005): Network may be underfitting');
      console.log('      Consider: More iterations, higher capacity, or better features');
    } else if (lastResult.plateauLevel < 0.002) {
      console.log('  ✅ Low plateau error (<0.002): Network fitting training data well');
    } else {
      console.log('  ➡️  Moderate plateau error: Network learning but not saturated');
    }
    
    // Plateau speed interpretation
    if (lastResult.plateauIter < 150) {
      console.log('  ⚠️  Quick plateau (<150 iter): Network saturating too fast');
      console.log('      Consider: Higher learning rate or more complex architecture');
    } else if (lastResult.plateauIter > 400) {
      console.log('  ➡️  Slow plateau (>400 iter): Network still learning at end');
      console.log('      Consider: More iterations per cycle');
    }
    
    // Recent drop rate interpretation
    if (lastResult.recentDropRate > 0.0001) {
      console.log('  ➡️  Still dropping: Network not fully converged within cycle');
    } else {
      console.log('  ✅ Flat at end: Network converged within cycle');
    }
    
    if (trends && results.length >= 3) {
      console.log('\n  Cross-cycle analysis:');
      if (trends.plateauLevelTrend < -0.0005) {
        console.log('  📉 Plateau error decreasing across cycles: Network improving representations');
      } else if (trends.plateauLevelTrend > 0.0005) {
        console.log('  📈 Plateau error increasing: Possible overfitting or distribution shift');
      }
    }
  }
  
  console.log('');
}

// Main
async function main() {
  const args = process.argv.slice(2);
  let logFile = args[0];
  
  if (!logFile) {
    // Find most recent log
    const logsDir = path.join(__dirname, '../../experiments');
    if (fs.existsSync(logsDir)) {
      const files = fs.readdirSync(logsDir)
        .filter(f => f.startsWith('training-run-') && f.endsWith('.log'))
        .sort()
        .reverse();
      if (files.length > 0) {
        logFile = path.join(logsDir, files[0]);
        console.log(`📂 Using most recent log: ${files[0]}`);
      }
    }
  }
  
  if (!logFile || !fs.existsSync(logFile)) {
    console.error('❌ No training log found. Specify a log file:');
    console.error('   node src/server/analyzeTrajectories.js experiments/training-run-*.log');
    process.exit(1);
  }
  
  const logContent = fs.readFileSync(logFile, 'utf-8');
  const cycles = parseTrainingLog(logContent);
  const results = analyzeCycles(cycles);
  const trends = computeTrends(results);
  
  printResults(results, trends);
}

main().catch(console.error);
