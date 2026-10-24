const brain = require('brain.js');
const fs = require('fs');

async function main() {
  const { GridManager } = await import('./src/core/GridManager.js');

  // Load the game's default network
  const net = new brain.NeuralNetwork();
  const weights = JSON.parse(fs.readFileSync('dist/models/network.json', 'utf-8'));
  net.fromJSON(weights);

  // Test output range
  const testState = new Array(401).fill(0.5);
  const testOut = net.run(testState);
  console.log('Game network output range:', Math.min(...testOut).toFixed(3), '-', Math.max(...testOut).toFixed(3));

  // State encoding (relative to current player)
  const getState = (grid) => {
    const state = [];
    for (let r = 0; r < grid.rows; r++) {
      for (let c = 0; c < grid.cols; c++) {
        const cell = grid.grid[r][c];
        for (const t of [cell.triangles[0], cell.triangles[1]]) {
          state.push(t ? (t.player === grid.currentPlayer ? 1 : -1) : 0);
          state.push(t?.rotation !== undefined ? t.rotation / 4 : -1);
        }
      }
    }
    state.push(1);
    return state;
  };

  const isValid = (grid, x, y, rot) => {
    const cell = grid.grid[y][x];
    if (cell.triangles.length >= 2) return false;
    if (!grid.isValidPlacement(x, y)) return false;
    if (cell.triangles.length === 1) {
      const existing = cell.triangles[0].rotation;
      if (existing !== rot && grid.getDiagonalGroup(existing) === grid.getDiagonalGroup(rot)) return false;
    }
    return true;
  };

  const getRandomMove = (grid) => {
    const valid = [];
    for (let y = 0; y < grid.rows; y++) {
      for (let x = 0; x < grid.cols; x++) {
        for (let r = 0; r < 4; r++) {
          if (isValid(grid, x, y, r)) valid.push({ x, y, rotation: r });
        }
      }
    }
    return valid.length ? valid[Math.floor(Math.random() * valid.length)] : null;
  };

  const getAIMove = (grid) => {
    const state = getState(grid);
    const output = net.run(state);
    const moves = [];
    for (let i = 0; i < 400; i++) {
      const rot = i % 4;
      const idx = Math.floor(i / 4);
      moves.push({ x: idx % grid.cols, y: Math.floor(idx / grid.cols), rotation: rot, score: output[i] });
    }
    moves.sort((a, b) => b.score - a.score);
    for (const m of moves) {
      if (isValid(grid, m.x, m.y, m.rotation)) return m;
    }
    return getRandomMove(grid);
  };

  console.log('Verifying game network (dist/models/network.json) vs Random: 100 games');
  const grid = new GridManager(10, 10);
  let wins = 0, losses = 0, draws = 0;

  for (let g = 0; g < 100; g++) {
    grid.reset();
    let cur = 1, m = 0;
    while (!grid.isGridFull() && m < 300) {
      grid.currentPlayer = cur;
      const move = cur === 1 ? getAIMove(grid) : getRandomMove(grid);
      if (!move) break;
      grid.selectedRotation = move.rotation;
      if (!grid.placeTriangle(move.x, move.y)) break;
      const claimed = grid.claimEnclosedTriangles();
      grid.extraTurns += Math.floor((1 + claimed) / 2);
      if (grid.extraTurns > 0) grid.extraTurns--;
      else cur = cur === 1 ? 2 : 1;
      m++;
    }
    const s1 = grid.getLargestContiguous(1);
    const s2 = grid.getLargestContiguous(2);
    if (s1 > s2) wins++; else if (s2 > s1) losses++; else draws++;

    if ((g + 1) % 10 === 0) process.stdout.write(`\r${g + 1}/100: W:${wins} L:${losses} D:${draws}`);
  }
  console.log(`\n\nResult: ${wins}% vs Random (${wins}W / ${losses}L / ${draws}D)`);
}

main().catch(e => console.error('Error:', e));
