/**
 * Game Auditor - Debug mode for verifying game integrity
 * 
 * Samples random games and replays them to detect bugs:
 * - Invalid move attempts
 * - Failed placements
 * - Player alternation errors
 * - Overlapping placements
 * - Incorrect enclosed claims
 * - Final state corruption
 * 
 * Usage:
 *   const auditor = new GameAuditor({ enabled: true, sampleRate: 0.1 });
 *   auditor.auditSelfPlayGame(gameHistory, gridManager, 'self-play');
 *   auditor.auditEvalGame(aiMoves, randomMoves, finalState);
 */

class GameAuditor {
  constructor(opts = {}) {
    this.enabled = opts.enabled ?? false;
    this.sampleRate = opts.sampleRate ?? 0.05; // 5% of games by default
    this.verbose = opts.verbose ?? false;
    this.issues = [];
    this.audited = 0;
    this.passed = 0;
    this.failed = 0;
  }

  /**
   * Decide whether to audit this game based on sample rate
   */
  shouldAudit() {
    if (!this.enabled) return false;
    return Math.random() < this.sampleRate;
  }

  /**
   * Audit a self-play game from training
   * @param {Array} gameHistory - Array of {player, move, state, actualMoveIndex} objects
   * @param {GridManager} gridManager - Final game state
   * @param {string} gameType - 'self-play', 'vs-random', 'vs-heuristic'
   */
  auditSelfPlayGame(gameHistory, gridManager, gameType = 'self-play') {
    if (!this.enabled) return;
    
    this.audited++;
    const issues = [];
    const trace = [];
    
    // Create a fresh grid to replay
    const GridManager = gridManager.constructor;
    const replayGrid = new GridManager(gridManager.rows, gridManager.cols);
    
    let expectedPlayer = 1;
    let extraTurnsRemaining = 0;
    let moveNum = 0;
    
    for (const step of gameHistory) {
      moveNum++;
      const { player, move, state, actualMoveIndex, claimed } = step;
      
      // Check 1: Player alternation (with extra turns handling)
      if (player !== expectedPlayer) {
        issues.push({
          type: 'PLAYER_ALTERNATION',
          move: moveNum,
          expected: expectedPlayer,
          actual: player,
          message: `Move ${moveNum}: Expected P${expectedPlayer} but got P${player}`
        });
        expectedPlayer = player; // Correct for next check
      }
      
      // Check 2: Move validity
      const isValid = replayGrid.isValidMove(move.x, move.y, move.rotation);
      if (!isValid) {
        issues.push({
          type: 'INVALID_MOVE',
          move: moveNum,
          player,
          position: `${move.x},${move.y}`,
          rotation: move.rotation,
          message: `Move ${moveNum}: P${player} attempted invalid move (${move.x},${move.y}) r${move.rotation}`
        });
      }
      
      // Check 3: Placement succeeds
      const success = replayGrid.placeTriangle(move.x, move.y, move.rotation, player);
      if (!success) {
        issues.push({
          type: 'PLACEMENT_FAILED',
          move: moveNum,
          player,
          position: `${move.x},${move.y}`,
          rotation: move.rotation,
          valid: isValid,
          message: `Move ${moveNum}: P${player} placement failed at (${move.x},${move.y}) r${move.rotation}`
        });
      } else {
        // Claim enclosed and track extra turns (same logic as AIPlayer.runSelfPlayGame)
        const enclosed = replayGrid.claimEnclosedTriangles(player);
        replayGrid.extraTurns += Math.floor((1 + enclosed) / 2);
        if (enclosed > 0) {
          trace.push(`Move ${moveNum}: P${player} enclosed ${enclosed} triangles (+${Math.floor((1 + enclosed) / 2)} extra turns)`);
        }
        
        // Verify claimed count matches history (if recorded)
        if (claimed !== undefined && claimed !== enclosed) {
          issues.push({
            type: 'CLAIM_MISMATCH',
            move: moveNum,
            player,
            message: `Move ${moveNum}: P${player} enclosed mismatch: history=${claimed}, replay=${enclosed}`
          });
        }
      }
      
      // Check 4: No overlapping placements (same rotation placed twice)
      const cell = replayGrid.grid[move.y][move.x];
      const sameRotationTriangles = cell.triangles.filter(t => t.rotation === move.rotation && t.player === player);
      if (sameRotationTriangles.length > 1) {
        issues.push({
          type: 'OVERLAP',
          move: moveNum,
          player,
          position: `${move.x},${move.y}`,
          rotation: move.rotation,
          message: `Move ${moveNum}: P${player} placed same rotation ${move.rotation} twice at (${move.x},${move.y})`
        });
      }
      
      // Switch expected player (mirrors AIPlayer extra turns logic)
      if (replayGrid.extraTurns > 0) {
        replayGrid.extraTurns--;
        // Player keeps turn
        trace.push(`Move ${moveNum}: P${player} gets extra turn (${replayGrid.extraTurns} remaining)`);
      } else {
        expectedPlayer = expectedPlayer === 1 ? 2 : 1;
      }
    }
    
    // Check 5: Final state - grid should be full
    if (!replayGrid.isGridFull()) {
      const emptyCells = [];
      for (let y = 0; y < replayGrid.rows; y++) {
        for (let x = 0; x < replayGrid.cols; x++) {
          const cell = replayGrid.grid[y][x];
          if (cell.triangles.length < 2) {
            emptyCells.push(`${x},${y}:${cell.triangles.length}`);
          }
        }
      }
      issues.push({
        type: 'GRID_NOT_FULL',
        message: `Game ended with ${emptyCells.length} incomplete cells: ${emptyCells.slice(0, 5).join(' ')}${emptyCells.length > 5 ? '...' : ''}`
      });
    }
    
    // Check 6: State encoding matches (if provided)
    if (gameHistory.length > 0 && gameHistory[0].state) {
      const finalState = gameHistory[gameHistory.length - 1].state;
      // Basic sanity: state length should be consistent
      if (finalState && finalState.length !== 401) {
        issues.push({
          type: 'STATE_ENCODING',
          message: `Final state has ${finalState.length} elements (expected 401)`
        });
      }
    }
    
    // Result
    const passed = issues.length === 0;
    if (passed) {
      this.passed++;
      if (this.verbose) {
        console.log(`✅ AUDIT PASSED: ${gameType} game (${gameHistory.length} moves)`);
      }
    } else {
      this.failed++;
      console.log(`❌ AUDIT FAILED: ${gameType} game (${gameHistory.length} moves, ${issues.length} issues)`);
      for (const issue of issues) {
        console.log(`   ⚠️  ${issue.type}: ${issue.message}`);
      }
      if (this.verbose && trace.length > 0) {
        console.log(`   📝 Trace: ${trace.slice(-5).join(' | ')}`);
      }
    }
    
    this.issues.push(...issues);
    return { passed, issues, trace };
  }

  /**
   * Audit an evaluation game (AI vs Random)
   * @param {Array} aiMoves - Array of {x, y, rotation} from AI
   * @param {Array} randomMoves - Array of {x, y, rotation} from Random
   * @param {object} finalState - Final scores and winner
   * @param {GridManager} gridManager - Grid class reference
   */
  auditEvalGame(aiMoves, randomMoves, finalState, GridManager) {
    if (!this.enabled) return;
    
    this.audited++;
    const issues = [];
    
    // Create fresh grid
    const grid = new GridManager(10, 10);
    
    let aiMoveNum = 0;
    let randomMoveNum = 0;
    let currentPlayer = 1; // AI is P1
    
    while (!grid.isGridFull() && (aiMoveNum < aiMoves.length || randomMoveNum < randomMoves.length)) {
      let move;
      const moveNum = aiMoveNum + randomMoveNum + 1;
      
      if (currentPlayer === 1 && aiMoveNum < aiMoves.length) {
        move = aiMoves[aiMoveNum];
        aiMoveNum++;
        
        // Check validity
        if (!grid.isValidMove(move.x, move.y, move.rotation)) {
          issues.push({
            type: 'AI_INVALID_MOVE',
            move: moveNum,
            position: `${move.x},${move.y}`,
            rotation: move.rotation,
            message: `AI move ${aiMoveNum} invalid: (${move.x},${move.y}) r${move.rotation}`
          });
        }
      } else if (currentPlayer === 2 && randomMoveNum < randomMoves.length) {
        move = randomMoves[randomMoveNum];
        randomMoveNum++;
        
        // Check validity
        if (!grid.isValidMove(move.x, move.y, move.rotation)) {
          issues.push({
            type: 'RANDOM_INVALID_MOVE',
            move: moveNum,
            position: `${move.x},${move.y}`,
            rotation: move.rotation,
            message: `Random move ${randomMoveNum} invalid: (${move.x},${move.y}) r${move.rotation}`
          });
        }
      } else {
        break; // No more moves
      }
      
      // Place
      const success = grid.placeTriangle(move.x, move.y, move.rotation, currentPlayer);
      if (!success) {
        issues.push({
          type: 'EVAL_PLACEMENT_FAILED',
          move: moveNum,
          player: currentPlayer,
          position: `${move.x},${move.y}`,
          message: `Eval move ${moveNum} placement failed for P${currentPlayer}`
        });
      } else {
        grid.claimEnclosedTriangles(currentPlayer);
      }
      
      // Handle extra turns
      if (grid.extraTurns > 0) {
        grid.extraTurns--;
      } else {
        currentPlayer = currentPlayer === 1 ? 2 : 1;
      }
    }
    
    // Verify final scores
    const p1Score = grid.getLargestContiguous(1);
    const p2Score = grid.getLargestContiguous(2);
    
    if (finalState) {
      if (finalState.p1Score !== undefined && finalState.p1Score !== p1Score) {
        issues.push({
          type: 'SCORE_MISMATCH',
          message: `AI score mismatch: reported ${finalState.p1Score}, actual ${p1Score}`
        });
      }
      if (finalState.p2Score !== undefined && finalState.p2Score !== p2Score) {
        issues.push({
          type: 'SCORE_MISMATCH',
          message: `Random score mismatch: reported ${finalState.p2Score}, actual ${p2Score}`
        });
      }
      
      // Winner check
      const actualWinner = p1Score > p2Score ? 1 : (p2Score > p1Score ? 2 : 0);
      if (finalState.winner !== undefined && finalState.winner !== actualWinner) {
        issues.push({
          type: 'WINNER_MISMATCH',
          message: `Winner mismatch: reported P${finalState.winner}, actual P${actualWinner}`
        });
      }
    }
    
    const passed = issues.length === 0;
    if (passed) {
      this.passed++;
      if (this.verbose) {
        console.log(`✅ AUDIT PASSED: eval game (AI ${aiMoves.length} moves, Random ${randomMoves.length} moves, score ${p1Score}-${p2Score})`);
      }
    } else {
      this.failed++;
      console.log(`❌ AUDIT FAILED: eval game (${issues.length} issues)`);
      for (const issue of issues) {
        console.log(`   ⚠️  ${issue.type}: ${issue.message}`);
      }
    }
    
    this.issues.push(...issues);
    return { passed, issues, finalScore: { p1: p1Score, p2: p2Score } };
  }

  /**
   * Get summary of all audits
   */
  getSummary() {
    const byType = {};
    for (const issue of this.issues) {
      byType[issue.type] = (byType[issue.type] || 0) + 1;
    }
    
    return {
      enabled: this.enabled,
      audited: this.audited,
      passed: this.passed,
      failed: this.failed,
      passRate: this.audited > 0 ? (this.passed / this.audited * 100).toFixed(1) : 0,
      issuesByType: byType
    };
  }

  /**
   * Print summary to console
   */
  printSummary() {
    const summary = this.getSummary();
    console.log('\n🔍 Game Audit Summary');
    console.log('='.repeat(40));
    console.log(`  Mode: ${summary.enabled ? 'ENABLED' : 'DISABLED'}`);
    console.log(`  Games audited: ${summary.audited}`);
    console.log(`  Passed: ${summary.passed} (${summary.passRate}%)`);
    console.log(`  Failed: ${summary.failed}`);
    
    if (Object.keys(summary.issuesByType).length > 0) {
      console.log('\n  Issues by type:');
      for (const [type, count] of Object.entries(summary.issuesByType)) {
        console.log(`    ${type}: ${count}`);
      }
    }
    console.log('='.repeat(40) + '\n');
  }
}

export { GameAuditor };
