/**
 * Player Interface - Standard interface for all game players
 * 
 * All players (AI, Random, Human, Agent) implement this interface
 */

export class PlayerInterface {
  constructor(playerNumber, config = {}) {
    this.playerNumber = playerNumber;
    this.config = config;
    this.gameHistory = [];
  }

  /**
   * Called when it's this player's turn
   * @param {GridManager} gridManager - Current game state
   * @returns {Promise<{x: number, y: number, rotation: number}>}
   */
  async getMove(gridManager) {
    throw new Error('Must implement getMove()');
  }

  /**
   * Check if player is ready to play
   * @returns {boolean}
   */
  isReady() {
    return true;
  }

  /**
   * Called after game ends
   * @param {number} winner - 0 (draw), 1, or 2
   */
  recordResult(winner) {
    this.lastResult = winner;
  }

  /**
   * Called before game starts
   */
  async initialize() {}

  /**
   * Get training data from this player (if trainable)
   * @returns {Array} Training samples
   */
  getTrainingData() {
    return [];
  }

  /**
   * Reset player state for new game
   */
  reset() {
    this.gameHistory = [];
    this.lastResult = null;
  }
}

// Player types enum
export const PlayerType = {
  AI: 'ai',
  RANDOM: 'random',
  HUMAN: 'human',
  AGENT: 'agent'
};

// Global for browser
if (typeof window !== 'undefined') {
  window.PlayerInterface = PlayerInterface;
  window.PlayerType = PlayerType;
}
