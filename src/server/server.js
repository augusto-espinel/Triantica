/**
 * Triantica Experiment Server
 * 
 * Run locally for:
 * - Batch experiments
 * - Parallel game execution
 * - Automatic file logging
 * - Training data generation
 */

import express from 'express';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { mkdir, writeFile } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Import core modules
import { GridManager } from '../core/GridManager.js';
import { GameSession } from '../core/GameSession.js';
import { GameLogger } from '../data/GameLogger.js';
import { RandomPlayer } from '../players/RandomPlayer.js';

const app = express();
app.use(express.json());

// Ensure data directory exists
const DATA_DIR = join(__dirname, '..', '..', 'experiments', 'data');

async function ensureDataDir() {
  try {
    await mkdir(DATA_DIR, { recursive: true });
  } catch (e) {
    // Already exists
  }
}

// === API Endpoints ===

app.post('/api/experiment', async (req, res) => {
  const { type, count = 100, config = {} } = req.body;
  const experimentId = `exp-${Date.now()}`;

  console.log(`Starting experiment: ${experimentId}`);
  console.log(`  Type: ${type}`);
  console.log(`  Count: ${count}`);

  try {
    const logger = new GameLogger(experimentId);
    const session = new GameSession(config);

    // Configure players based on type
    if (type === 'self-play') {
      // AI vs AI - need to load network
      // For now, use Random vs Random as placeholder
      session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    } else if (type === 'vs-random') {
      // AI vs Random
      session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    } else {
      throw new Error(`Unknown experiment type: ${type}`);
    }

    // Run games
    const results = await session.runGames(count);

    // Log each game
    results.games.forEach(game => {
      logger.logGame(game, session.players);
    });

    // Save results
    const filename = `${experimentId}.json`;
    const filepath = join(DATA_DIR, filename);
    await writeFile(filepath, logger.export(), 'utf-8');

    // Return summary
    res.json({
      experimentId,
      status: 'complete',
      summary: logger.getSummary(),
      downloadUrl: `/experiments/data/${filename}`,
    });

  } catch (error) {
    console.error('Experiment error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/experiments', async (req, res) => {
  // List available experiments
  const { readdir } = await import('fs/promises');
  try {
    const files = await readdir(DATA_DIR);
    const experiments = files
      .filter(f => f.endsWith('.json'))
      .map(f => f.replace('.json', ''));
    res.json({ experiments });
  } catch (e) {
    res.json({ experiments: [] });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// === Start Server ===

const PORT = process.env.PORT || 8080;

ensureDataDir().then(() => {
  app.listen(PORT, () => {
    console.log(`\n🎮 Triantica Experiment Server`);
    console.log(`   Running on http://localhost:${PORT}`);
    console.log(`   Data directory: ${DATA_DIR}\n`);
  });
});
