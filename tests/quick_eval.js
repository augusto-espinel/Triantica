/**
 * Quick evaluation script to test AI against both Random and Heuristic opponents
 */

import { AIPlayer } from '../src/players/AIPlayer.js';
import { RandomPlayer } from '../src/players/RandomPlayer.js';
import { HeuristicPlayer } from '../src/players/HeuristicPlayer.js';
import { GameSession } from '../src/core/GameSession.js';
import { readFileSync } from 'fs';

async function evalAgainst(opponent, opponentName, games = 100) {
  const networkJSON = JSON.parse(readFileSync('trainedAI_network.json', 'utf-8'));
  
  const ai = new AIPlayer(1, {
    difficulty: 'medium',
    explorationFactor: 0, // Pure network
    networkJSON
  });
  await ai.initialize();
  
  const session = new GameSession();
  session.setPlayers(ai, opponent);
  
  const results = await session.runGames(games);
  const winRate = results.wins.ai / games * 100;
  
  const aiScores = results.scores && results.scores[1] ? results.scores[1] : [];
  const oppScores = results.scores && results.scores[2] ? results.scores[2] : [];
  const aiMedian = aiScores.length > 0 ? aiScores.sort((a,b) => a-b)[Math.floor(aiScores.length/2)] : 0;
  const oppMedian = oppScores.length > 0 ? oppScores.sort((a,b) => a-b)[Math.floor(oppScores.length/2)] : 0;
  
  console.log(`vs ${opponentName}: ${winRate.toFixed(1)}% (${results.wins.ai}W / ${results.wins.opponent}L / ${results.wins.draws}D)`);
  console.log(`  Scores: AI median=${aiMedian}, ${opponentName} median=${oppMedian}`);
  return winRate;
}

(async () => {
  console.log('📊 Evaluation: Pure Network + Opening Diversity\n');
  console.log('='.repeat(50));
  await evalAgainst(new RandomPlayer(2), 'Random', 100);
  console.log('');
  await evalAgainst(new HeuristicPlayer(2, { explorationFactor: 0 }), 'Heuristic', 100);
  console.log('='.repeat(50));
})();
