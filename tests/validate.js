/**
 * Automated validation tests for Triantica architecture
 * Run with: node tests/validate.js
 */

import { GridManager, GRID_ROWS, GRID_COLS } from '../src/core/GridManager.js';
import { GameSession } from '../src/core/GameSession.js';
import { RandomPlayer } from '../src/players/RandomPlayer.js';
import { GameLogger, HumanGameLogger } from '../src/data/GameLogger.js';

let passed = 0;
let failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`✅ ${name}`);
    passed++;
  } catch (e) {
    console.log(`❌ ${name}`);
    console.log(`   Error: ${e.message}`);
    failed++;
  }
}

function assertEqual(actual, expected, msg = '') {
  if (actual !== expected) {
    throw new Error(`${msg} Expected ${expected}, got ${actual}`);
  }
}

async function runTests() {
  console.log('\n🧪 Triantica Architecture Validation\n');
  console.log('='.repeat(50));

  // GridManager Tests
  console.log('\n📦 GridManager Tests');

  await test('GridManager creates correct size grid', () => {
    const gm = new GridManager();
    assertEqual(gm.rows, 10, 'rows');
    assertEqual(gm.cols, 10, 'cols');
    assertEqual(gm.grid.length, 10, 'grid rows');
    assertEqual(gm.grid[0].length, 10, 'grid cols');
  });

  await test('GridManager starts empty', () => {
    const gm = new GridManager();
    assertEqual(gm.isGridFull(), false);
    assertEqual(gm.isFirstMove(), true);
    assertEqual(gm.currentPlayer, 1);
  });

  await test('GridManager can place first triangle anywhere', () => {
    const gm = new GridManager();
    const success = gm.placeTriangle(5, 5, 0, 1);
    assertEqual(success, true);
    assertEqual(gm.grid[5][5].triangles.length, 1);
    assertEqual(gm.isFirstMove(), false);
  });

  await test('GridManager rejects invalid second placement (isolated)', () => {
    const gm = new GridManager();
    gm.placeTriangle(5, 5, 0, 1);
    const success = gm.placeTriangle(0, 0, 0, 2);
    assertEqual(success, false);
  });

  await test('GridManager accepts adjacent placement', () => {
    const gm = new GridManager();
    gm.placeTriangle(5, 5, 0, 1);
    const success = gm.placeTriangle(6, 5, 0, 2);
    assertEqual(success, true);
  });

  await test('GridManager tracks clusters', () => {
    const gm = new GridManager();
    gm.placeTriangle(5, 5, 0, 1);
    gm.placeTriangle(5, 5, 2, 1);
    assertEqual(gm.getLargestContiguous(1), 2);
  });

  await test('GridManager toroidal placement wraps around row=0/col=0 boundary', () => {
    const gm = new GridManager();
    gm.placeTriangle(0, 0, 0, 1); // touches vertex (0,0) which wraps to (10,10)
    assertEqual(gm.isValidMove(9, 9, 2), true, 'Toroidal opposite corner (9,9,rot=2) should be legal');
  });

  await test('GridManager manifold graph consistency (400 face-slots, 100 vertices)', () => {
    const gm = new GridManager();
    assertEqual(gm.manifoldGraph.numFaceSlots, 400, '400 face slots');
    assertEqual(gm.manifoldGraph.numVertices, 100, '100 toroidal vertices');
    gm.placeTriangle(0, 0, 0, 1);
    assertEqual(gm.isManifoldFaceValidPlacement((9 * 10 + 9) * 4 + 2), true, 'Manifold graph wraps vertex (0,0)');
  });

  await test('GridManager getValidMoves returns array', () => {
    const gm = new GridManager();
    const moves = gm.getValidMoves();
    assertEqual(moves.length, 400, 'First move should have 400 options');
    
    gm.placeTriangle(5, 5, 0, 1);
    const moves2 = gm.getValidMoves();
    assertEqual(moves2.length < 400, true, 'Second move should have fewer options');
  });

  await test('GridManager clone works', async () => {
    const gm = new GridManager();
    gm.placeTriangle(5, 5, 0, 1);
    
    const clone = gm.clone();
    
    // Check clone has the same state
    assertEqual(clone.rows, gm.rows, 'rows');
    assertEqual(clone.grid[5][5].triangles.length, 1, 'clone has original piece');
    
    // Modify clone - placeTriangle(x=6, y=5, ...) → grid[5][6]
    const success = clone.placeTriangle(6, 5, 0, 2);
    assertEqual(success, true, 'clone placement should succeed');
    
    // Check original unchanged (grid[y=5][x=6])
    assertEqual(gm.grid[5][6].triangles.length, 0, 'original unchanged');
    
    // Check clone has new piece (grid[y=5][x=6])
    assertEqual(clone.grid[5][6].triangles.length, 1, 'clone should have new piece');
  });

  // RandomPlayer Tests
  console.log('\n🎲 RandomPlayer Tests');

  await test('RandomPlayer returns valid move', async () => {
    const gm = new GridManager();
    const player = new RandomPlayer(1);
    const move = await player.getMove(gm);
    
    assertEqual(move !== null, true, 'Should return a move');
    assertEqual(move.x >= 0 && move.x < 10, true, 'x in range');
    assertEqual(move.y >= 0 && move.y < 10, true, 'y in range');
    assertEqual(move.rotation >= 0 && move.rotation < 4, true, 'rotation in range');
  });

  await test('RandomPlayer makes legal moves', async () => {
    const gm = new GridManager();
    gm.placeTriangle(5, 5, 0, 1);
    
    const player = new RandomPlayer(2);
    for (let i = 0; i < 10; i++) {
      const move = await player.getMove(gm);
      const valid = gm.isValidMove(move.x, move.y, move.rotation);
      assertEqual(valid, true, `Move ${i} should be valid`);
    }
  });

  // GameSession Tests
  console.log('\n🎮 GameSession Tests');

  await test('GameSession initializes correctly', () => {
    const session = new GameSession();
    assertEqual(session.gridManager.rows, 10);
  });

  await test('GameSession requires players', async () => {
    const session = new GameSession();
    try {
      await session.runGame();
      throw new Error('Should have thrown');
    } catch (e) {
      assertEqual(e.message.includes('Players not set'), true);
    }
  });

  await test('GameSession runs game with RandomPlayers', async () => {
    const session = new GameSession();
    session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    
    const result = await session.runGame();
    
    assertEqual(result.moves > 0, true, 'Game should have moves');
    assertEqual(result.winner !== undefined, true, 'Should have winner');
    assertEqual(result.finalScores[1] >= 0, true, 'P1 score');
    assertEqual(result.finalScores[2] >= 0, true, 'P2 score');
  });

  await test('GameSession runs multiple games', async () => {
    const session = new GameSession();
    session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    
    const results = await session.runGames(5);
    
    assertEqual(results.games.length, 5, 'Should run 5 games');
    assertEqual(results.wins.player1 + results.wins.player2 + results.wins.draws, 5, 'All games accounted');
  });

  // GameLogger Tests
  console.log('\n📊 GameLogger Tests');

  await test('GameLogger tracks games', async () => {
    const logger = new GameLogger('test');
    const session = new GameSession();
    session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    
    const result = await session.runGame();
    logger.logGame(result, { 1: session.players[1], 2: session.players[2] });
    
    const summary = logger.getSummary();
    assertEqual(summary.gamesPlayed, 1);
  });

  await test('GameLogger exports JSON', async () => {
    const logger = new GameLogger('test');
    const session = new GameSession();
    session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    
    const result = await session.runGame();
    logger.logGame(result, { 1: session.players[1], 2: session.players[2] });
    
    const json = logger.export();
    const parsed = JSON.parse(json);
    
    assertEqual(parsed.experimentId, 'test');
    assertEqual(parsed.games.length, 1);
  });

  await test('HumanGameLogger captures pre-move canonical (1800) & legacy (401) state, FIFO rotates, and handles forfeits', () => {
    const store = new Map();
    const mockStorage = {
      getItem: (k) => store.get(k) || null,
      setItem: (k, v) => store.set(k, v),
      removeItem: (k) => store.delete(k)
    };
    const hLogger = new HumanGameLogger({ maxGames: 2, storage: mockStorage });
    const gm = new GridManager();

    // Game 1: Play 1 move on empty board, then end via forfeit
    hLogger.startGame('HvAI', 'hard', 1, gm);
    const snap = hLogger.capturePreMoveState(gm, 1);
    gm.placeTriangle(5, 5, 0, 1);
    hLogger.logMove(snap, { x: 5, y: 5, rotation: 0 }, 1, 0);
    hLogger.endGame(gm, { winnerOverride: 2, forfeitedBy: 1 });

    assertEqual(hLogger.getStatus().count, 1, '1 game logged');
    const g1 = hLogger.games[0];
    assertEqual(g1.winner, 2, 'Forfeit overrides cluster score winner');
    assertEqual(g1.forfeitedBy, 1, 'Forfeited player recorded');
    assertEqual(g1.moves[0].state.length, 1800, 'Canonical state is 10x10x18 (1800 floats)');
    assertEqual(g1.moves[0].legacyState.length, 401, 'Legacy state is 401 floats');
    assertEqual(g1.moves[0].actionIndex, (5 * 10 + 5) * 4 + 0, 'Action index matches (y*10+x)*4+rot');
    assertEqual(g1.moves[0].valueTarget, -1.0, 'Value target is -1 for losing player');

    // Verify pre-move state was captured BEFORE placement (cell 5,5 slot 0 empty channel = 1)
    const cellOffset = (5 * 10 + 5) * 18;
    assertEqual(g1.moves[0].state[cellOffset], 1, 'Pre-move cell (5,5) slot 0 was empty before move');

    // Game 2 & Game 3: Verify FIFO capacity = 2
    for (let i = 0; i < 2; i++) {
      gm.reset();
      hLogger.startGame('HvAI', 'medium', 1, gm);
      const s = hLogger.capturePreMoveState(gm, 1);
      gm.placeTriangle(2, 2, 1, 1);
      hLogger.logMove(s, { x: 2, y: 2, rotation: 1 }, 1, 0);
      hLogger.endGame(gm);
    }
    assertEqual(hLogger.getStatus().count, 2, 'FIFO buffer capped at maxGames=2');

    const exported = hLogger.exportAndFlush();
    assertEqual(exported.gameCount, 2, 'Exported 2 games');
    assertEqual(hLogger.getStatus().count, 0, 'Buffer flushed after export');
  });

  // Baseline experiment: 20 games Random vs Random
  console.log('\n🔬 Quick Experiment');

  await test('Random vs Random baseline (20 games)', async () => {
    const session = new GameSession();
    session.setPlayers(new RandomPlayer(1), new RandomPlayer(2));
    
    const results = await session.runGames(20);
    
    console.log(`   P1 wins: ${results.wins.player1}, P2 wins: ${results.wins.player2}, Draws: ${results.wins.draws}`);
    console.log(`   Avg moves: ${results.avgMoves.toFixed(1)}`);
    
    assertEqual(results.games.length, 20, 'All 20 games played');
    assertEqual(results.wins.player1 + results.wins.player2 + results.wins.draws, 20, 'Results complete');
  });

  // Summary
  console.log('\n' + '='.repeat(50));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('='.repeat(50) + '\n');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});
