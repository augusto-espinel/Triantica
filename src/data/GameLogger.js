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
  async saveToFile(filepath) {
    if (typeof window !== 'undefined') {
      console.warn('saveToFile() only works in Node.js');
      return;
    }

    const fs = await import('fs');
    const data = this.export();
    await fs.promises.writeFile(filepath, data, 'utf-8');
    console.log(`Saved to ${filepath}`);
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.GameLogger = GameLogger;
}
