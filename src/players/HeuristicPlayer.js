/**
 * HeuristicPlayer - Greedy 1-ply heuristic player for Triantica
 * 
 * Uses no neural network. Picks the move that maximizes immediate reward:
 * - Enclosure claims (+0.3 per claimed triangle)
 * - Connected cluster growth (+0.1 per triangle in cluster)
 * - Largest-cluster lead (+0.2 per lead delta)
 * 
 * Provides a stable, competent opponent for curriculum training.
 * Much stronger than random but deterministic and fast.
 */

import { PlayerInterface, PlayerType } from './PlayerInterface.js';

export class HeuristicPlayer extends PlayerInterface {
  /**
   * @param {number} playerNumber - 1 or 2
   * @param {object} config
   * @param {number} config.explorationFactor - Probability of random move (default 0.05)
   */
  constructor(playerNumber, config = {}) {
    super(playerNumber, config);
    this.type = PlayerType.AI;
    this.explorationFactor = config.explorationFactor ?? 0.05;
  }

  /**
   * Initialize (no-op for heuristic player)
   */
  async initialize() {
    return;
  }

  /**
   * Get move using greedy 1-ply heuristic
   * @param {GridManager} gridManager - Current game state
   * @returns {{ x: number, y: number, rotation: number }}
   */
  getMove(gridManager) {
    // Small exploration: random move
    if (Math.random() < this.explorationFactor) {
      return this._getRandomValidMove(gridManager);
    }

    const { rows, cols } = gridManager;
    const opponent = this.playerNumber === 1 ? 2 : 1;
    
    // Current baseline for lead calculation
    const myCurrentLargest = gridManager.getLargestContiguous(this.playerNumber);
    const oppCurrentLargest = gridManager.getLargestContiguous(opponent);

    let bestMove = null;
    let bestScore = -Infinity;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        for (let rotation = 0; rotation < 4; rotation++) {
          if (!gridManager.isValidMove(x, y, rotation)) continue;

          // Simulate on clone
          const tempManager = gridManager.clone();
          tempManager.currentPlayer = this.playerNumber;
          const simSuccess = tempManager.placeTriangle(x, y, rotation, this.playerNumber);
          if (!simSuccess) continue;

          // Score components
          const simClaimed = tempManager.claimEnclosedTriangles(this.playerNumber);
          const claimScore = simClaimed * 0.3;

          const simClusterSize = tempManager.getConnectedClusterSize(x, y, rotation);
          const clusterScore = simClusterSize * 0.1;

          const myNewLargest = tempManager.getLargestContiguous(this.playerNumber);
          const oppNewLargest = tempManager.getLargestContiguous(opponent);
          const leadDelta = (myNewLargest - oppNewLargest) - (myCurrentLargest - oppCurrentLargest);
          const leadScore = leadDelta * 0.2;

          const totalScore = claimScore + clusterScore + leadScore;

          if (totalScore > bestScore) {
            bestScore = totalScore;
            bestMove = { x, y, rotation };
          }
        }
      }
    }

    return bestMove || this._getRandomValidMove(gridManager);
  }

  /**
   * Get a random valid move
   */
  _getRandomValidMove(gridManager) {
    const { rows, cols } = gridManager;
    const validMoves = [];
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        for (let rotation = 0; rotation < 4; rotation++) {
          if (gridManager.isValidMove(x, y, rotation)) {
            validMoves.push({ x, y, rotation });
          }
        }
      }
    }
    if (validMoves.length === 0) return null;
    return validMoves[Math.floor(Math.random() * validMoves.length)];
  }

  /**
   * No-op: Heuristic player doesn't generate training data
   */
  generateTrainingData() {
    return [];
  }

  /**
   * Reset state
   */
  reset() {
    // No state to reset
  }
}

// Global for browser
if (typeof window !== 'undefined') {
  window.HeuristicPlayer = HeuristicPlayer;
}
