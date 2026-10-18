/**
 * Browser Console Experiment Script
 * 
 * Paste this into browser console on GitHub Pages or local server
 * to run experiments and download results.
 * 
 * Usage:
 *   await runExperiment('self-play', 10);
 *   await runExperiment('vs-random', 100);
 */

// Helper to create players
function createPlayers(type, playerNum) {
  if (type === 'random') {
    return new RandomPlayer(playerNum);
  }
  // Add more player types as needed
  return new RandomPlayer(playerNum);
}

// Main experiment runner
async function runExperiment(type = 'vs-random', count = 100) {
  console.log(`\n🎮 Starting experiment: ${type}`);
  console.log(`Games: ${count}\n`);

  const experimentId = `browser-${type}-${Date.now()}`;
  const logger = new GameLogger(experimentId);

  const session = new GameSession();
  
  // Configure players
  const p1 = createPlayers(type === 'self-play' ? 'random' : 'random', 1);
  const p2 = createPlayers('random', 2);
  session.setPlayers(p1, p2);

  // Progress callback
  let progress = 0;
  const progressStep = Math.max(1, Math.floor(count / 10));
  
  session.config.onGameEnd = (result) => {
    if (logger.games.length % progressStep === 0) {
      const pct = Math.floor((logger.games.length / count) * 100);
      console.log(`  ${pct}% (${logger.games.length}/${count})`);
    }
  };

  // Run games
  console.log('Running games...');
  const results = await session.runGames(count);

  // Log results
  results.games.forEach(game => logger.logGame(game, { 1: p1, 2: p2 }));
  logger.printSummary();

  // Download results
  logger.download(`${experimentId}.json`);

  return logger;
}

// Convenience functions
async function quickBaseline() {
  return await runExperiment('vs-random', 100);
}

async function quickSelfPlay() {
  return await runExperiment('self-play', 100);
}

// Instructions
console.log(`
╔══════════════════════════════════════════════════════════╗
║         Triantica Experiment Runner                      ║
╠══════════════════════════════════════════════════════════╣
║  Usage:                                                   ║
║    await runExperiment('vs-random', 100)                  ║
║    await runExperiment('self-play', 50)                   ║
║                                                           ║
║  Quick functions:                                         ║
║    await quickBaseline()   // 100 vs-random games        ║
║    await quickSelfPlay()    // 100 self-play games       ║
╚══════════════════════════════════════════════════════════╝
`);

// Make available globally
window.runExperiment = runExperiment;
window.quickBaseline = quickBaseline;
window.quickSelfPlay = quickSelfPlay;
