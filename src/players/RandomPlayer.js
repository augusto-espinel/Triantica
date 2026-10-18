/**
 * RandomPlayer - Makes random valid moves
 * 
 * Useful for:
 * - Baseline testing
 * - Generating diverse training data
 * - Quick game simulation
 */

import { PlayerInterface } from './PlayerInterface.js';

export class RandomPlayer extends PlayerInterface {
  constructor(playerNumber, config = {}) {
    super(playerNumber, config);
    this.name = config.name || `Random-${playerNumber}`;
  }

  async getMove(gridManager) {
    const validMoves = gridManager.getValidMoves();
    
    if (validMoves.length === 0) {
      return null; // No valid moves
    }

    // Random selection
    const index = Math.floor(Math.random() * validMoves.length);
    return validMoves[index];
  }

  toString() {
    return this.name;
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.RandomPlayer = RandomPlayer;
}
