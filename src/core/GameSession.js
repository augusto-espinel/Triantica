/**
 * GameSession - Orchestrates a game between two players
 * 
 * Platform-agnostic: Works in browser and Node.js
 * Collects data for training and analysis
 */

import { GridManager, GRID_ROWS, GRID_COLS } from './GridManager.js';

export class GameSession {
  constructor(config = {}) {
    this.config = {
      rows: config.rows || GRID_ROWS,
      cols: config.cols || GRID_COLS,
      maxMoves: config.maxMoves || (GRID_ROWS * GRID_COLS * 2 * 1.5), // Safety limit
      logMoves: config.logMoves !== false,
      onMove: config.onMove || null, // Callback after each move
      onGameEnd: config.onGameEnd || null, // Callback when game ends
    };

    this.gridManager = new GridManager(this.config.rows, this.config.cols);
    this.players = {};
    this.moveHistory = [];
    this.trainingSamples = [];
  }

  /**
   * Set players for the game
   * @param {PlayerInterface} player1 
   * @param {PlayerInterface} player2 
   */
  setPlayers(player1, player2) {
    this.players[1] = player1;
    this.players[2] = player2;
  }

  /**
   * Run a complete game
   * @returns {Object} Game result
   */
  async runGame() {
    // Validate players
    if (!this.players[1] || !this.players[2]) {
      throw new Error('Players not set. Call setPlayers() first.');
    }

    // Reset state
    this.gridManager.reset();
    this.moveHistory = [];
    this.trainingSamples = [];
    this.currentPlayerNum = 1;

    // Initialize players
    await this.players[1].initialize();
    await this.players[2].initialize();

    const startTime = Date.now();
    let moveCount = 0;

    // Game loop
    while (!this.gridManager.isGridFull() && moveCount < this.config.maxMoves) {
      const player = this.players[this.currentPlayerNum];
      const stateBefore = this.gridManager.getStateRepresentation(this.currentPlayerNum);

      // Get move from current player
      const move = await player.getMove(this.gridManager);

      if (!move) {
        // No valid move, game ends
        break;
      }

      // Validate and execute move
      const valid = this.gridManager.placeTriangle(move.x, move.y, move.rotation, this.currentPlayerNum);
      
      if (!valid) {
        console.warn(`Invalid move by player ${this.currentPlayerNum}:`, move);
        continue;
      }

      // Handle claims
      const claims = this.gridManager.claimEnclosedTriangles(this.currentPlayerNum);

      // Record move
      const moveRecord = {
        moveIndex: moveCount,
        player: this.currentPlayerNum,
        x: move.x,
        y: move.y,
        rotation: move.rotation,
        claims,
        timestamp: Date.now() - startTime,
        player1Score: this.gridManager.getLargestContiguous(1),
        player2Score: this.gridManager.getLargestContiguous(2),
      };

      this.moveHistory.push(moveRecord);

      // Store training sample
      this.trainingSamples.push({
        moveIndex: moveCount,
        player: this.currentPlayerNum,
        state: stateBefore,
        move,
        claims,
        player1ScoreAfter: moveRecord.player1Score,
        player2ScoreAfter: moveRecord.player2Score,
      });

      // Callback
      if (this.config.onMove) {
        this.config.onMove(moveRecord, this.gridManager);
      }

      // Handle extra turns
      this.gridManager.extraTurns += Math.floor((1 + claims) / 2);

      // Switch player
      if (this.gridManager.extraTurns > 0) {
        this.gridManager.extraTurns--;
      } else {
        this.gridManager.currentPlayer = this.gridManager.currentPlayer === 1 ? 2 : 1;
        this.currentPlayerNum = this.gridManager.currentPlayer;
      }

      moveCount++;
    }

    // Determine winner
    const winner = this.gridManager.getWinner();
    const duration = Date.now() - startTime;

    // Create result
    const result = {
      winner,
      finalScores: {
        1: this.gridManager.getLargestContiguous(1),
        2: this.gridManager.getLargestContiguous(2),
      },
      moves: moveCount,
      duration,
      moveHistory: this.moveHistory,
      trainingSamples: this.trainingSamples,
    };

    // Notify players
    this.players[1].recordResult(winner);
    this.players[2].recordResult(winner);

    // Callback
    if (this.config.onGameEnd) {
      this.config.onGameEnd(result);
    }

    return result;
  }

  /**
   * Run multiple games
   * @param {number} count - Number of games to run
   * @returns {Object} Aggregate results
   */
  async runGames(count) {
    const results = {
      games: [],
      wins: { 1: 0, 2: 0, draws: 0 },
      totalMoves: 0,
      totalDuration: 0,
    };

    for (let i = 0; i < count; i++) {
      const gameResult = await this.runGame();
      results.games.push(gameResult);

      if (gameResult.winner === 0) results.wins.draws++;
      else results.wins[gameResult.winner]++;

      results.totalMoves += gameResult.moves;
      results.totalDuration += gameResult.duration;

      // Swap players every game for fairness
      const temp = this.players[1];
      this.players[1] = this.players[2];
      this.players[2] = temp;
    }

    results.avgMoves = results.totalMoves / count;
    results.avgDuration = results.totalDuration / count;
    results.winRates = {
      1: results.wins[1] / count,
      2: results.wins[2] / count,
      draws: results.wins.draws / count,
    };

    return results;
  }

  /**
   * Export game data for training
   * @returns {Object} Training data
   */
  exportTrainingData() {
    return {
      samples: this.trainingSamples.length,
      data: this.trainingSamples,
    };
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.GameSession = GameSession;
}
