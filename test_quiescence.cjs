#!/usr/bin/env node
/**
 * Comprehensive verification test for Quiescence MCTS (search & searchBatched)
 */

const assert = require('assert');
const { QuiescenceMCTS, findEnclosureOpportunities, playEnclosureCascade, calculateEnclosureRisk } = require('./src/server/QuiescenceMCTS.cjs');

(async () => {
  const { GridManager } = await import('./src/core/GridManager.js');
  const { HeuristicPlayer } = await import('./src/players/HeuristicPlayer.js');

  console.log('\n🧪 Testing Quiescence MCTS (search & searchBatched)\n');
  console.log('='.repeat(60));

  // Play a few moves in a real game to reach an inflection point where enclosures exist
  const grid = new GridManager(10, 10);
  const p1 = new HeuristicPlayer(1, { explorationFactor: 0.2 });
  const p2 = new HeuristicPlayer(2, { explorationFactor: 0.2 });

  let cur = 1;
  let foundBoard = null;
  for (let step = 0; step < 60; step++) {
    grid.currentPlayer = cur;
    // Check if there is a position where P1 has an enclosing move OR a risky move
    const p1Encl = findEnclosureOpportunities(grid, cur);
    if (p1Encl.length > 0 && step > 10) {
      foundBoard = grid.clone();
      break;
    }
    const move = cur === 1 ? p1.getMove(grid) : p2.getMove(grid);
    if (!move) break;
    const res = grid.executeMove(move.x, move.y, move.rotation, cur);
    cur = res.nextPlayer;
  }

  assert(foundBoard, 'Should find a board with enclosure opportunities');
  const activePlayer = foundBoard.currentPlayer;
  const enclosures = findEnclosureOpportunities(foundBoard, activePlayer);
  console.log(`📋 Found test board (Player ${activePlayer} to move) with ${enclosures.length} enclosure opportunity(ies)`);
  console.log(`   Best enclosing move: (${enclosures[0].x}, ${enclosures[0].y}) r${enclosures[0].rotation} claims ${enclosures[0].claimed}`);

  // Test 1: Enclosing move MUST have 0 risk!
  const bestEnclMove = enclosures[0];
  const enclRisk = calculateEnclosureRisk(foundBoard, bestEnclMove, activePlayer);
  console.log(`\n📋 Test 1: Enclosing Move Risk Check`);
  console.log(`   Move (${bestEnclMove.x}, ${bestEnclMove.y}) r${bestEnclMove.rotation}: claimed=${enclRisk.claimed}, risk=${enclRisk.risk}`);
  assert.strictEqual(enclRisk.risk, 0, 'Enclosing move must have 0 risk because extra turns allow patching!');

  // Test 2: Mock flat-prior network to test QuiescenceMCTS.searchBatched vs MCTS
  const mockNetwork = {
    run: async () => ({
      policy: new Array(400).fill(0.5),
      value: 0.0
    }),
    batchPredict: async (states) => ({
      policies: states.map(() => new Array(400).fill(0.5)),
      values: states.map(() => 0.0)
    })
  };

  const qMcts = new QuiescenceMCTS(mockNetwork, {
    c: 1.5,
    alpha: 0.03,
    epsilon: 0.0, // deterministic for test
    batchSize: 8,
    temperature: 0, // pick argmax
    quiescenceMode: 'greedy',
    quiescenceDepth: 5,
    riskPenaltyWeight: 0.25,
    enclosureBonusWeight: 0.3,
    encodeState: () => new Array(1800).fill(0)
  });

  const batchedRes = await qMcts.searchBatched(foundBoard, 48);
  const enclIdx = (bestEnclMove.y * 10 + bestEnclMove.x) * 4 + bestEnclMove.rotation;

  console.log(`\n📋 Test 2: QuiescenceMCTS.searchBatched() Execution`);
  console.log(`   Selected move: (${batchedRes.move.x}, ${batchedRes.move.y}) r${batchedRes.move.rotation}`);
  console.log(`   Policy prob on enclosing move: ${(batchedRes.policy[enclIdx] * 100).toFixed(2)}%`);
  console.log(`   Root Q-value: ${batchedRes.value.toFixed(4)}`);

  // Verify the selected move actually claims an enclosure!
  const verifyGrid = foundBoard.clone();
  const execCheck = verifyGrid.executeMove(batchedRes.move.x, batchedRes.move.y, batchedRes.move.rotation, activePlayer);
  console.log(`   Selected move claimed: ${execCheck.claimed} triangle(s)`);
  assert(execCheck.claimed > 0, 'QuiescenceMCTS.searchBatched should select an enclosing move even with a flat network prior!');

  console.log('\n' + '='.repeat(60));
  console.log('✅ All QuiescenceMCTS tests passed!\n');
})();
