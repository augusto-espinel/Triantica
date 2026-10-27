const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

async function evaluateCheckpoint(networkPath, evalGames = 100) {
  const { GridManager } = await import('../src/core/GridManager.js');
  const { HeuristicPlayer } = await import('../src/players/HeuristicPlayer.js');
  const { TFConvNetwork, getStateCanonical } = require('../src/server/train_tfjs.cjs');

  const netData = JSON.parse(fs.readFileSync(networkPath, 'utf8'));
  const network = new TFConvNetwork();
  network.fromJSON(netData);

  const grid = new GridManager(10, 10);

  function getRandomMove(g) {
    const valid = [];
    for (let y = 0; y < g.rows; y++) {
      for (let x = 0; x < g.cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (g.isValidMove(x, y, r)) valid.push({ x, y, rotation: r });
        }
      }
    }
    return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
  }

  function getAIMove(g, aiPlayerNumber) {
    const state = g.getCanonicalState(aiPlayerNumber);
    const result = network.run(state);
    const policy = result.policy;
    const moves = [];
    for (let i = 0; i < policy.length; i++) {
      const rot = i % 4;
      const idx = Math.floor(i / 4);
      const x = idx % g.cols;
      const y = Math.floor(idx / g.cols);
      moves.push({ x, y, rotation: rot, score: policy[i] });
    }
    moves.sort((a, b) => b.score - a.score);
    for (const m of moves) {
      if (g.isValidMove(m.x, m.y, m.rotation)) return m;
    }
    return getRandomMove(g);
  }

  let heurWins = 0, heurLosses = 0, heurDraws = 0;
  let heurP1Wins = 0, heurP1Losses = 0, heurP1Draws = 0, heurP1Games = 0;
  let heurP2Wins = 0, heurP2Losses = 0, heurP2Draws = 0, heurP2Games = 0;
  const heurAIScores = [], heurOppScores = [];

  for (let g = 0; g < evalGames; g++) {
    grid.reset();
    let cur = 1, m = 0;
    const maxM = grid.rows * grid.cols * 3;
    const heurAIPlayerNumber = (g % 2 === 0) ? 1 : 2;
    const heuristicOpponentNumber = heurAIPlayerNumber === 1 ? 2 : 1;
    const heuristicEval = new HeuristicPlayer(heuristicOpponentNumber, { explorationFactor: 0 });

    while (!grid.isGridFull() && m < maxM) {
      grid.currentPlayer = cur;
      let move;
      if (m < 2) {
        // First 2 plies (1 opening move per player) are random
        move = getRandomMove(grid);
      } else {
        move = cur === heurAIPlayerNumber
          ? getAIMove(grid, heurAIPlayerNumber)
          : heuristicEval.getMove(grid);
      }
      if (!move) break;
      const exec = grid.executeMove(move.x, move.y, move.rotation, cur);
      if (!exec.success) break;
      cur = exec.nextPlayer;
      m++;
    }

    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    if (heurAIPlayerNumber === 1) {
      heurP1Games++;
      if (s1 > s2) { heurWins++; heurP1Wins++; }
      else if (s2 > s1) { heurLosses++; heurP1Losses++; }
      else { heurDraws++; heurP1Draws++; }
      heurAIScores.push(s1);
      heurOppScores.push(s2);
    } else {
      heurP2Games++;
      if (s2 > s1) { heurWins++; heurP2Wins++; }
      else if (s1 > s2) { heurLosses++; heurP2Losses++; }
      else { heurDraws++; heurP2Draws++; }
      heurAIScores.push(s2);
      heurOppScores.push(s1);
    }
  }

  return {
    aiWinRate: (heurWins / evalGames) * 100,
    aiWins: heurWins,
    losses: heurLosses,
    draws: heurDraws,
    p1WinRate: heurP1Games > 0 ? (heurP1Wins / heurP1Games) * 100 : 0,
    p2WinRate: heurP2Games > 0 ? (heurP2Wins / heurP2Games) * 100 : 0,
    p1Wins: heurP1Wins,
    p1Losses: heurP1Losses,
    p1Draws: heurP1Draws,
    p1Games: heurP1Games,
    p2Wins: heurP2Wins,
    p2Losses: heurP2Losses,
    p2Draws: heurP2Draws,
    p2Games: heurP2Games,
    aiScores: heurAIScores,
    opponentScores: heurOppScores
  };
}

if (require.main === module) {
  const netPath = process.argv[2] || path.resolve(__dirname, '../dashboard/data/TF-B12_network.json');
  const games = parseInt(process.argv[3] || '100', 10);
  evaluateCheckpoint(netPath, games).then(res => {
    const meanAI = (res.aiScores.reduce((a, b) => a + b, 0) / res.aiScores.length).toFixed(1);
    const meanOpp = (res.opponentScores.reduce((a, b) => a + b, 0) / res.opponentScores.length).toFixed(1);
    console.log(JSON.stringify({
      file: path.basename(netPath),
      aiWinRate: res.aiWinRate,
      p1WinRate: res.p1WinRate,
      p2WinRate: res.p2WinRate,
      aiWins: res.aiWins,
      losses: res.losses,
      draws: res.draws,
      meanAIScore: parseFloat(meanAI),
      meanOppScore: parseFloat(meanOpp),
      uniqueAIScoresCount: new Set(res.aiScores).size
    }, null, 2));
  });
}

module.exports = { evaluateCheckpoint };
