/**
 * GameLogger - Unified logging for experiments
 * 
 * Works in browser and Node.js
 * Exports to JSON for training data
 */

export class GameLogger {
  constructor(experimentId = 'default') {
    this.experimentId = experimentId;
    this.startTime = Date.now();
    this.games = [];
    this.metadata = {
      platform: typeof window !== 'undefined' ? 'browser' : 'node',
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : process.version,
    };
  }

  /**
   * Log a completed game
   */
  logGame(gameResult, players) {
    const record = {
      id: `game-${this.games.length + 1}`,
      timestamp: new Date().toISOString(),
      experimentId: this.experimentId,
      players: {
        1: { type: players[1].constructor.name, playerNumber: 1 },
        2: { type: players[2].constructor.name, playerNumber: 2 },
      },
      winner: gameResult.winner,
      finalScores: gameResult.finalScores,
      moves: gameResult.moves,
      duration_ms: gameResult.duration,
      moveHistory: gameResult.moveHistory,
      trainingSamples: gameResult.trainingSamples,
    };

    this.games.push(record);
    return record;
  }

  /**
   * Get summary statistics
   */
  getSummary() {
    if (this.games.length === 0) {
      return { error: 'No games logged' };
    }

    const wins = { 1: 0, 2: 0, draws: 0 };
    let totalMoves = 0;
    let totalDuration = 0;
    let totalSamples = 0;

    this.games.forEach(game => {
      if (game.winner === 0) wins.draws++;
      else wins[game.winner]++;

      totalMoves += game.moves;
      totalDuration += game.duration_ms;
      totalSamples += game.trainingSamples?.length || 0;
    });

    return {
      experimentId: this.experimentId,
      gamesPlayed: this.games.length,
      wins,
      winRates: {
        1: (wins[1] / this.games.length * 100).toFixed(1) + '%',
        2: (wins[2] / this.games.length * 100).toFixed(1) + '%',
        draws: (wins.draws / this.games.length * 100).toFixed(1) + '%',
      },
      avgMoves: (totalMoves / this.games.length).toFixed(1),
      avgDuration_ms: (totalDuration / this.games.length).toFixed(0),
      totalTrainingSamples: totalSamples,
      elapsed_s: ((Date.now() - this.startTime) / 1000).toFixed(1),
    };
  }

  /**
   * Print summary to console
   */
  printSummary() {
    const summary = this.getSummary();
    console.log('\n' + '='.repeat(60));
    console.log(`EXPERIMENT: ${summary.experimentId}`);
    console.log('='.repeat(60));
    console.log(`Games played: ${summary.gamesPlayed}`);
    console.log(`Player 1 wins: ${summary.winRates[1]}`);
    console.log(`Player 2 wins: ${summary.winRates[2]}`);
    console.log(`Draws: ${summary.winRates.draws}`);
    console.log(`Avg moves/game: ${summary.avgMoves}`);
    console.log(`Avg duration: ${summary.avgDuration_ms}ms`);
    console.log(`Training samples: ${summary.totalTrainingSamples}`);
    console.log(`Elapsed: ${summary.elapsed_s}s`);
    console.log('='.repeat(60) + '\n');
  }

  /**
   * Export all data as JSON
   */
  toJSON() {
    return {
      experimentId: this.experimentId,
      metadata: this.metadata,
      summary: this.getSummary(),
      games: this.games,
    };
  }

  /**
   * Export to string
   */
  export() {
    return JSON.stringify(this.toJSON(), null, 2);
  }

  /**
   * Download as file (browser only)
   */
  download(filename) {
    if (typeof window === 'undefined') {
      console.warn('download() only works in browser');
      return;
    }

    const data = this.export();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || `experiment-${this.experimentId}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Save to file (Node.js only)
   */
  saveToFile(filepath) {
    return import('fs').then(fs => {
      if (typeof window !== 'undefined') {
        console.warn('saveToFile() only works in Node.js');
        return;
      }
      const data = this.export();
      return fs.promises.writeFile(filepath, data, 'utf-8').then(() => {
        console.log(`Saved to ${filepath}`);
      });
    });
  }
}

/**
 * HumanGameLogger - FIFO rotating buffer for Human vs. AI match telemetry
 *
 * Persists completed Human vs. AI matches in localStorage (up to maxGames)
 * with deduplication and manual export-and-flush.
 * Captures PRE-MOVE 10x10x18 canonical state (1,800 floats, player-relative)
 * and 401-float legacy state for each ply, along with actionIndex (0..399)
 * and terminal valueTarget (+1, 0, -1).
 */
export class HumanGameLogger {
  constructor(config = {}) {
    this.maxGames = config.maxGames || 100;
    this.storageKey = config.storageKey || 'triantica_human_games';
    this.storage = config.storage || (typeof localStorage !== 'undefined' ? localStorage : null);
    this.currentGame = null;
    this.games = this._loadGames();
  }

  _loadGames() {
    try {
      if (!this.storage) return [];
      const stored = this.storage.getItem(this.storageKey);
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      return [];
    }
  }

  _save() {
    if (!this.storage) return;
    try {
      this.storage.setItem(this.storageKey, JSON.stringify(this.games));
    } catch (e) {
      if (this.games.length > 0) {
        this.games.shift();
        this._save();
      }
    }
  }

  startGame(gameMode, difficulty, humanPlayerNumber, gridManager) {
    this.currentGame = {
      id: `human-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      mode: gameMode,
      difficulty: difficulty || 'medium',
      humanPlayer: humanPlayerNumber,
      moves: [],
      finalScores: null,
      winner: null,
      forfeitedBy: null,
      gridDimensions: { rows: gridManager.rows, cols: gridManager.cols }
    };
  }

  capturePreMoveState(gridManager, player) {
    return {
      canonicalState: this._getCanonicalState(gridManager, player),
      legacyState: this._getLegacyState(gridManager, player)
    };
  }

  logMove(preMoveSnapshot, move, player, claimed, gridManager = null) {
    if (!this.currentGame) return;

    const snapshot = (preMoveSnapshot && preMoveSnapshot.canonicalState)
      ? preMoveSnapshot
      : this.capturePreMoveState(gridManager || preMoveSnapshot, player);

    const cols = this.currentGame.gridDimensions?.cols || 10;
    const actionIndex = (move.y * cols + move.x) * 4 + move.rotation;

    this.currentGame.moves.push({
      ply: this.currentGame.moves.length,
      player: player,
      isHuman: player === this.currentGame.humanPlayer,
      state: snapshot.canonicalState,
      legacyState: snapshot.legacyState,
      move: { x: move.x, y: move.y, rotation: move.rotation },
      actionIndex: actionIndex,
      claimed: claimed || 0,
      timestamp: Date.now()
    });
  }

  _getCanonicalState(gridManager, player) {
    if (gridManager && typeof gridManager.getCanonicalState === 'function') {
      return gridManager.getCanonicalState(player);
    }
    const rows = gridManager?.rows || 10;
    const cols = gridManager?.cols || 10;
    const state = new Float32Array(rows * cols * 18);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = gridManager.grid[r][c];
        const cellOffset = (r * cols + c) * 18;
        for (let slot = 0; slot < 2; slot++) {
          const slotOffset = cellOffset + slot * 9;
          const tri = cell.triangles.find(t => (slot === 0 ? t.rotation < 2 : t.rotation >= 2));
          if (!tri) {
            state[slotOffset] = 1;
          } else {
            const ownerOffset = tri.player === player ? 0 : 4;
            state[slotOffset + 1 + ownerOffset + tri.rotation] = 1;
          }
        }
      }
    }
    return Array.from(state);
  }

  _getLegacyState(gridManager, player) {
    const state = [];
    for (let y = 0; y < gridManager.rows; y++) {
      for (let x = 0; x < gridManager.cols; x++) {
        const cell = gridManager.grid[y][x];
        const t1 = cell.triangles[0];
        const t2 = cell.triangles[1];

        const normalizePlayer = (p) => {
          if (!p) return 0;
          return p === player ? 1 : -1;
        };

        const normalizeRotation = (r) => {
          return r === undefined || r === -1 ? -1 : r / 4.0;
        };

        state.push(normalizePlayer(t1?.player));
        state.push(normalizeRotation(t1?.rotation));
        state.push(normalizePlayer(t2?.player));
        state.push(normalizeRotation(t2?.rotation));
      }
    }
    state.push(1);
    return state;
  }

  endGame(gridManager, options = {}) {
    if (!this.currentGame) return;

    if (this.currentGame.moves.length === 0) {
      this.currentGame = null;
      return;
    }

    const p1Score = gridManager.getLargestContiguous(1);
    const p2Score = gridManager.getLargestContiguous(2);

    this.currentGame.finalScores = { 1: p1Score, 2: p2Score };
    if (options.winnerOverride !== undefined && options.winnerOverride !== null) {
      this.currentGame.winner = options.winnerOverride;
    } else {
      this.currentGame.winner = p1Score > p2Score ? 1 : (p2Score > p1Score ? 2 : 0);
    }
    if (options.forfeitedBy) {
      this.currentGame.forfeitedBy = options.forfeitedBy;
    }
    this.currentGame.totalMoves = this.currentGame.moves.length;

    const winner = this.currentGame.winner;
    for (const m of this.currentGame.moves) {
      m.valueTarget = winner === 0 ? 0.0 : (winner === m.player ? 1.0 : -1.0);
    }

    const exists = this.games.findIndex(g => g.id === this.currentGame.id);
    if (exists !== -1) {
      this.games[exists] = this.currentGame;
    } else {
      if (this.games.length >= this.maxGames) {
        this.games.shift();
      }
      this.games.push(this.currentGame);
    }

    this._save();
    this.currentGame = null;
  }

  abortGame() {
    this.currentGame = null;
  }

  exportAndFlush() {
    if (this.games.length === 0) {
      return null;
    }

    const exportData = {
      schemaVersion: '2.0-canonical',
      exportedAt: new Date().toISOString(),
      gameCount: this.games.length,
      stateEncoding: {
        state: '10x10x18 canonical one-hot (1800 floats, pre-move, player-relative)',
        legacyState: '10x10x4 + 1 legacy vector (401 floats, pre-move, rotation=r/4.0)',
        actionIndex: '(y * 10 + x) * 4 + rotation (0..399)',
        valueTarget: '+1 if move.player won, -1 if lost, 0 if draw'
      },
      games: this.games
    };

    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
      const json = JSON.stringify(exportData, null, 2);
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);

      const a = document.createElement('a');
      a.href = url;
      a.download = `triantica_human_games_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1500);
    }

    this.games = [];
    if (this.storage) {
      this.storage.removeItem(this.storageKey);
    }

    return exportData;
  }

  getStatus() {
    return {
      count: this.games.length,
      capacity: this.maxGames,
      oldestGame: this.games[0]?.timestamp || null,
      newestGame: this.games[this.games.length - 1]?.timestamp || null,
      humanWins: this.games.filter(g => g.winner === g.humanPlayer).length,
      aiWins: this.games.filter(g => g.winner !== g.humanPlayer && g.winner !== 0).length,
      draws: this.games.filter(g => g.winner === 0).length
    };
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.GameLogger = GameLogger;
  window.HumanGameLogger = HumanGameLogger;
}
