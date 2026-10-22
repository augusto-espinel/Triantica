/**
 * UI Audit Logger - Comprehensive move legality audit for UI gameplay
 * 
 * Similar to GameAuditor for experiments, but injects logging into browser context
 * to capture:
 * - All legal moves available at each turn
 * - The move attempted
 * - Whether it succeeded or failed
 * - Grid state snapshots
 */

import puppeteer from 'puppeteer';
import fs from 'fs';

class UIAuditLogger {
  constructor() {
    this.turns = [];
    this.currentTurn = null;
    this.moveCount = 0;
    this.failCount = 0;
  }

  startTurn(player, playerType, legalMoves) {
    this.currentTurn = {
      moveNumber: ++this.moveCount,
      player,
      playerType,
      legalMoves: legalMoves.map(m => `(${m.x},${m.y})r${m.rotation}`),
      legalMovesCount: legalMoves.length,
      attemptedMove: null,
      success: null,
      error: null,
      gridState: null
    };
  }

  recordAttempt(x, y, rotation) {
    if (this.currentTurn) {
      this.currentTurn.attemptedMove = `(${x},${y})r${rotation}`;
    }
  }

  recordResult(success, error = null) {
    if (this.currentTurn) {
      this.currentTurn.success = success;
      this.currentTurn.error = error;
      if (!success) this.failCount++;
      this.turns.push({ ...this.currentTurn });
    }
  }

  summary() {
    const successCount = this.moveCount - this.failCount;
    return {
      totalMoves: this.moveCount,
      successfulMoves: successCount,
      failedMoves: this.failCount,
      successRate: this.moveCount > 0 ? (successCount / this.moveCount * 100).toFixed(1) : 100,
      turns: this.turns
    };
  }
}

async function auditUIGameplay() {
  console.log('🔍 UI Audit Logger - Comprehensive Move Legality Audit\n');
  console.log('━'.repeat(60));

  const logger = new UIAuditLogger();
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });

  // Expose audit functions to browser context
  await page.exposeFunction('auditStartTurn', (player, playerType, legalMovesJson) => {
    const legalMoves = JSON.parse(legalMovesJson);
    logger.startTurn(player, playerType, legalMoves);
    console.log(`\n📋 Turn ${logger.moveCount}: P${player} (${playerType})`);
    console.log(`   Legal moves available: ${legalMoves.length}`);
    if (legalMoves.length <= 10) {
      console.log(`   ${legalMoves.map(m => `(${m.x},${m.y})r${m.rotation}`).join(', ')}`);
    } else {
      console.log(`   ${legalMoves.slice(0, 5).map(m => `(${m.x},${m.y})r${m.rotation}`).join(', ')}... (${legalMoves.length - 5} more)`);
    }
  });

  await page.exposeFunction('auditRecordAttempt', (x, y, rotation) => {
    logger.recordAttempt(x, y, rotation);
    console.log(`   Attempting: (${x},${y}) rotation=${rotation}`);
  });

  await page.exposeFunction('auditRecordResult', (success, error) => {
    logger.recordResult(success, error);
    if (success) {
      console.log(`   ✅ SUCCESS`);
    } else {
      console.log(`   ❌ FAILED: ${error || 'Move rejected'}`);
    }
  });

  // Intercept console to capture game events
  page.on('console', async msg => {
    const text = msg.text();
    
    // Inject audit logging when moves are being computed
    if (text.includes('getValidMoves') || text.includes('chose:')) {
      await page.evaluate(() => {
        // Hook into GridManager if not already hooked
        if (window._auditInstalled) return;
        
        const gameScene = window.game?.scene?.getScene?.('GameScene');
        if (!gameScene?.gridManager) return;

        const originalGetValidMoves = gameScene.gridManager.getValidMoves.bind(gameScene.gridManager);
        gameScene.gridManager.getValidMoves = function() {
          const moves = originalGetValidMoves();
          window.auditStartTurn(this.currentPlayer, 'unknown', JSON.stringify(moves));
          return moves;
        };

        window._auditInstalled = true;
        console.log('[AUDIT] Hooks installed');
      }).catch(() => {});
    }
  });

  try {
    console.log('📍 Loading game...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2', timeout: 30000 });
    await page.waitForSelector('canvas', { timeout: 10000 });
    await new Promise(r => setTimeout(r, 3000));

    // Install comprehensive audit hooks in browser
    await page.evaluate(() => {
      // Wait for game to be ready
      const installHooks = () => {
        const gameScene = window.game?.scene?.getScene?.('GameScene');
        if (!gameScene) {
          setTimeout(installHooks, 500);
          return;
        }

        console.log('[AUDIT] GameScene found, installing hooks...');

        // Hook getValidMoves to log all legal moves
        const gm = gameScene.gridManager;
        const origGetValidMoves = gm.getValidMoves.bind(gm);
        gm.getValidMoves = function() {
          const moves = origGetValidMoves();
          window.auditStartTurn(this.currentPlayer, 'hooked', JSON.stringify(moves));
          return moves;
        };

        // Hook placeTriangle to log results
        const origPlaceTriangle = gm.placeTriangle.bind(gm);
        gm.placeTriangle = function(x, y) {
          window.auditRecordAttempt(x, y, this.selectedRotation);
          const success = origPlaceTriangle(x, y);
          window.auditRecordResult(success, success ? null : 'Invalid placement');
          return success;
        };

        console.log('[AUDIT] All hooks installed successfully');
      };

      setTimeout(installHooks, 1000);
    });

    console.log('\n🎯 Starting AI vs Random (30 seconds test)...');
    await page.mouse.click(300, 460);

    // Watch for moves
    await new Promise(r => setTimeout(r, 30000));

    // Capture final grid state
    const gridState = await page.evaluate(() => {
      const gameScene = window.game?.scene?.getScene?.('GameScene');
      if (!gameScene?.gridManager) return null;
      
      const gm = gameScene.gridManager;
      const grid = gm.grid;
      let filledCells = 0;
      let totalTriangles = 0;
      
      for (let y = 0; y < gm.rows; y++) {
        for (let x = 0; x < gm.cols; x++) {
          const cell = grid[y][x];
          if (cell.triangles.length > 0) filledCells++;
          totalTriangles += cell.triangles.length;
        }
      }
      
      return {
        rows: gm.rows,
        cols: gm.cols,
        filledCells,
        totalTriangles,
        currentPlayer: gm.currentPlayer,
        extraTurns: gm.extraTurns,
        scores: gm.scores
      };
    });

    console.log('\n' + '='.repeat(60));
    console.log('📊 AUDIT SUMMARY');
    console.log('='.repeat(60));

    const summary = logger.summary();
    console.log(`\nTotal moves: ${summary.totalMoves}`);
    console.log(`Successful: ${summary.successfulMoves}`);
    console.log(`Failed: ${summary.failedMoves}`);
    console.log(`Success rate: ${summary.successRate}%`);

    if (gridState) {
      console.log(`\n📋 Grid State:`);
      console.log(`   Size: ${gridState.cols}x${gridState.rows}`);
      console.log(`   Filled cells: ${gridState.filledCells}/${gridState.cols * gridState.rows}`);
      console.log(`   Total triangles: ${gridState.totalTriangles}`);
      console.log(`   Current player: ${gridState.currentPlayer}`);
      console.log(`   Scores: P1=${gridState.scores[1]}, P2=${gridState.scores[2]}`);
    }

    // Save detailed log
    const auditLog = {
      summary,
      gridState,
      failedMoves: summary.turns.filter(t => !t.success),
      timestamp: new Date().toISOString()
    };

    await fs.promises.writeFile('ui_audit_detailed.json', JSON.stringify(auditLog, null, 2));
    console.log('\n💾 Detailed log saved to ui_audit_detailed.json');

    // Show failed moves analysis
    if (summary.failedMoves > 0) {
      console.log('\n⚠️  FAILED MOVES ANALYSIS:');
      summary.turns.filter(t => !t.success).forEach(t => {
        console.log(`\n   Move #${t.moveNumber}: P${t.player} (${t.playerType})`);
        console.log(`   Legal moves available: ${t.legalMovesCount}`);
        console.log(`   Attempted: ${t.attemptedMove}`);
        console.log(`   Was in legal moves? ${t.legalMoves.includes(t.attemptedMove) ? 'YES' : 'NO'}`);
      });
    }

  } catch (error) {
    console.error('❌ Audit failed:', error.message);
  }

  await browser.close();
  console.log('\n✅ Audit complete');
}

auditUIGameplay().catch(console.error);
