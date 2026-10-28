/**
 * Integration test: AIPlayer with pre-trained network vs RandomPlayer
 * Run with: node tests/test_ai_integration.js
 */

import { GridManager } from '../src/core/GridManager.js';
import { GameSession } from '../src/core/GameSession.js';
import { RandomPlayer } from '../src/players/RandomPlayer.js';
import { AIPlayer } from '../src/players/AIPlayer.js';
import { GameLogger } from '../src/data/GameLogger.js';
import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
  console.log('\n🧠 AIPlayer Integration Test\n');
  console.log('='.repeat(50));

  // 1. Load pre-trained network
  let networkJSON = null;
  const networkPath = join(__dirname, '..', 'trainedAI_network.json');
  
  try {
    const data = readFileSync(networkPath, 'utf-8');
    networkJSON = JSON.parse(data);
    console.log(`✅ Loaded pre-trained network from ${networkPath}`);
    console.log(`   Network keys: ${Object.keys(networkJSON).join(', ')}`);
  } catch (e) {
    console.log(`⚠️  No pre-trained network found at ${networkPath}`);
    console.log('   AI will play random moves (no training yet)');
  }

  // 2. Create AI player with network
  const aiPlayer = new AIPlayer(1, {
    difficulty: 'medium',
    explorationFactor: 0.1,  // Less exploration for testing
    networkJSON: networkJSON,
  });

  await aiPlayer.initialize();
  console.log(`✅ AIPlayer initialized (network loaded: ${aiPlayer.isNetworkLoaded})`);

  // 3. Test: AI vs Random (50 games)
  console.log('\n🎮 Running AI vs Random (50 games)...');
  
  const randomPlayer = new RandomPlayer(2);
  const session = new GameSession();
  session.setPlayers(aiPlayer, randomPlayer);
  
  const results = await session.runGames(50);
  
  console.log('\n📊 AI vs Random Results:');
  console.log(`   AI (P1) wins:    ${results.wins[1]}`);
  console.log(`   Random (P2) wins: ${results.wins[2]}`);
  console.log(`   Draws:            ${results.wins.draws}`);
  console.log(`   AI win rate:       ${(results.wins[1] / 50 * 100).toFixed(1)}%`);
  console.log(`   Avg moves/game:    ${results.avgMoves.toFixed(1)}`);
  console.log(`   Avg duration:      ${results.avgDuration.toFixed(0)}ms`);

  // 4. Test: Random vs AI (swapped, 50 games)
  console.log('\n🎮 Running Random vs AI (AI as P2, 50 games)...');
  
  const aiPlayer2 = new AIPlayer(2, {
    difficulty: 'medium',
    explorationFactor: 0.1,
    networkJSON: networkJSON,
  });
  await aiPlayer2.initialize();
  
  const session2 = new GameSession();
  session2.setPlayers(new RandomPlayer(1), aiPlayer2);
  
  const results2 = await session2.runGames(50);
  
  console.log('\n📊 Random vs AI Results:');
  console.log(`   Random (P1) wins: ${results2.wins[1]}`);
  console.log(`   AI (P2) wins:    ${results2.wins[2]}`);
  console.log(`   Draws:            ${results2.wins.draws}`);
  console.log(`   AI win rate:       ${(results2.wins[2] / 50 * 100).toFixed(1)}%`);
  console.log(`   Avg moves/game:    ${results2.avgMoves.toFixed(1)}`);

  // 6. Test: AI vs AI (10 games - self-play)
  console.log('\n🎮 Running AI vs AI (10 games)...');
  
  const aiPlayer3 = new AIPlayer(1, { difficulty: 'medium', explorationFactor: 0.2, networkJSON });
  const aiPlayer4 = new AIPlayer(2, { difficulty: 'medium', explorationFactor: 0.2, networkJSON });
  await aiPlayer3.initialize();
  await aiPlayer4.initialize();
  
  const session3 = new GameSession();
  session3.setPlayers(aiPlayer3, aiPlayer4);
  const results3 = await session3.runGames(10);
  
  console.log('\n📊 AI vs AI Results:');
  console.log(`   P1 wins:  ${results3.wins[1]}`);
  console.log(`   P2 wins:  ${results3.wins[2]}`);
  console.log(`   Draws:    ${results3.wins.draws}`);
  console.log(`   Avg moves: ${results3.avgMoves.toFixed(1)}`);

  // 7. Save all results
  const allResults = {
    aiVsRandom: { p1Wins: results.wins[1], p2Wins: results.wins[2], draws: results.wins.draws, aiWinRate: results.wins[1] / 50 },
    randomVsAi: { p1Wins: results2.wins[1], p2Wins: results2.wins[2], draws: results2.wins.draws, aiWinRate: results2.wins[2] / 50 },
    aiVsAi: { p1Wins: results3.wins[1], p2Wins: results3.wins[2], draws: results3.wins.draws },
    overallAIWinRate: (results.wins[1] + results2.wins[2]) / 100,
    networkLoaded: aiPlayer.isNetworkLoaded,
  };
  
  const outDir = join(__dirname, '..', 'experiments', 'data');
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    join(outDir, 'ai-integration.json'),
    JSON.stringify(allResults, null, 2)
  );
  console.log('\n💾 All results saved to experiments/data/ai-integration.json');

  // Summary
  const totalAIWins = results.wins[1] + results2.wins[2];
  const totalGames = 100;
  console.log('\n' + '='.repeat(50));
  console.log(`📈 Overall AI Win Rate: ${(totalAIWins / totalGames * 100).toFixed(1)}% (${totalAIWins}/${totalGames})`);
  console.log('='.repeat(50) + '\n');
}

main().catch(e => {
  console.error('Integration test failed:', e);
  process.exit(1);
});
