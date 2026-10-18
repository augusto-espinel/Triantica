/**
 * CLI Experiment Runner
 * 
 * Usage:
 *   node src/server/runExperiment.js --type self-play --count 100
 *   node src/server/runExperiment.js --type vs-random --count 50 --output exp-001.json
 */

import { parseArgs } from 'util';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdir, writeFile } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

import { GameSession } from '../core/GameSession.js';
import { GameLogger } from '../data/GameLogger.js';
import { RandomPlayer } from '../players/RandomPlayer.js';

const { values } = parseArgs({
  options: {
    type: { type: 'string', default: 'vs-random' },
    count: { type: 'string', default: '100' },
    output: { type: 'string' },
    help: { type: 'boolean', default: false },
  },
});

if (values.help) {
  console.log(`
Triantica Experiment Runner

Options:
  --type <type>       Experiment type: self-play, vs-random (default: vs-random)
  --count <number>    Number of games to run (default: 100)
  --output <file>     Output filename (default: auto-generated)
  --help              Show this help
`);
  process.exit(0);
}

async function runExperiment() {
  const type = values.type;
  const count = parseInt(values.count, 10);
  const experimentId = `exp-${type}-${Date.now()}`;

  console.log(`\n🎮 Triantica Experiment Runner`);
  console.log('='.repeat(50));
  console.log(`Experiment ID: ${experimentId}`);
  console.log(`Type: ${type}`);
  console.log(`Games: ${count}`);
  console.log('='.repeat(50) + '\n');

  const logger = new GameLogger(experimentId);
  const session = new GameSession();

  // Configure players
  if (type === 'self-play') {
    console.log('Setting up: Random vs Random (AI placeholder)');
    session.setPlayers(new RandomPlayer(1, { name: 'Random-1' }), 
                       new RandomPlayer(2, { name: 'Random-2' }));
  } else if (type === 'vs-random') {
    console.log('Setting up: Random vs Random');
    session.setPlayers(new RandomPlayer(1, { name: 'Random-1' }),
                       new RandomPlayer(2, { name: 'Random-2' }));
  } else {
    console.error(`Unknown experiment type: ${type}`);
    process.exit(1);
  }

  console.log(`Running ${count} games...\n`);

  // Progress tracking
  let lastProgress = 0;
  session.config.onGameEnd = (result) => {
    const progress = Math.floor((logger.games.length / count) * 100);
    if (progress >= lastProgress + 10) {
      console.log(`  Progress: ${progress}% (${logger.games.length}/${count})`);
      lastProgress = progress;
    }
  };

  // Run games
  const results = await session.runGames(count);

  // Log each game
  results.games.forEach(game => {
    logger.logGame(game, session.players);
  });

  // Print summary
  logger.printSummary();

  // Save results
  const dataDir = join(__dirname, '..', '..', 'experiments', 'data');
  await mkdir(dataDir, { recursive: true });

  const filename = values.output || `${experimentId}.json`;
  const filepath = join(dataDir, filename);
  await writeFile(filepath, logger.export(), 'utf-8');

  console.log(`\n✅ Results saved to: ${filepath}`);
  console.log(`📊 Training samples: ${logger.getSummary().totalTrainingSamples}\n`);
}

runExperiment().catch(error => {
  console.error('Experiment failed:', error);
  process.exit(1);
});
