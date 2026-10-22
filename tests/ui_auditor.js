/**
 * UI Auditor - Audits live gameplay for move integrity
 * 
 * Similar to GameAuditor but for browser gameplay.
 * Captures: grid state, legal moves, attempted moves, success/failure
 */

import puppeteer from 'puppeteer';

class UIAuditor {
  constructor() {
    this.auditLog = [];
    this.errors = [];
    this.moveCount = 0;
    this.failedMoves = 0;
  }

  log(event) {
    this.auditLog.push({
      ...event,
      timestamp: Date.now(),
      moveNumber: this.moveCount
    });
  }

  error(event) {
    this.errors.push({
      ...event,
      timestamp: Date.now(),
      moveNumber: this.moveCount
    });
    this.failedMoves++;
  }

  summary() {
    return {
      totalMoves: this.moveCount,
      failedMoves: this.failedMoves,
      successRate: this.moveCount > 0 ? 
        ((this.moveCount - this.failedMoves) / this.moveCount * 100).toFixed(1) : 0,
      errorRate: this.moveCount > 0 ? 
        (this.failedMoves / this.moveCount * 100).toFixed(1) : 0,
      totalErrors: this.errors.length
    };
  }
}

async function auditGameplay() {
  console.log('🔍 UI Auditor - Starting comprehensive move audit...\n');
  
  const auditor = new UIAuditor();
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });
  
  // Capture all console output with detailed parsing
  page.on('console', msg => {
    const text = msg.text();
    const type = msg.type();
    
    // Parse move events
    if (text.includes('chose:')) {
      const match = text.match(/(\w+) (?:Player )?(\d*) chose: \((\d+), (\d+)\), Rot: (\d+)/);
      if (match) {
        auditor.moveCount++;
        auditor.log({
          type: 'move_attempt',
          player: match[2] || (match[1] === 'AI' ? '1' : '2'),
          playerType: match[1],
          x: parseInt(match[3]),
          y: parseInt(match[4]),
          rotation: parseInt(match[5]),
          raw: text
        });
      }
    }
    
    // Parse move failures
    if (text.includes('failed placement') || text.includes('Error:')) {
      const match = text.match(/Move \((\d+), (\d+), (\d+)\) failed/);
      if (match) {
        auditor.error({
          type: 'move_failed',
          x: parseInt(match[1]),
          y: parseInt(match[2]),
          rotation: parseInt(match[3]),
          raw: text
        });
      }
    }
    
    // Parse enclosure events
    if (text.includes('enclosed')) {
      const match = text.match(/enclosed (\d+) triangle/);
      if (match) {
        auditor.log({
          type: 'enclosure',
          triangles: parseInt(match[1]),
          raw: text
        });
      }
    }
    
    // Parse turn switches
    if (text.includes('Switched to player')) {
      const match = text.match(/Switched to player (\d+), type: (\w+)/);
      if (match) {
        auditor.log({
          type: 'turn_switch',
          player: match[1],
          playerType: match[2],
          raw: text
        });
      }
    }
  });
  
  try {
    console.log('📍 Loading game...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2', timeout: 30000 });
    await page.waitForSelector('canvas', { timeout: 10000 });
    await new Promise(r => setTimeout(r, 3000));
    
    // Test 1: AI vs Random
    console.log('\n🎯 Test 1: AI vs Random');
    console.log('━'.repeat(50));
    await page.mouse.click(300, 460);
    await new Promise(r => setTimeout(r, 30000));
    
    console.log('\n📊 Test 1 Results:');
    const summary1 = auditor.summary();
    console.log(`  Total moves: ${summary1.totalMoves}`);
    console.log(`  Failed moves: ${summary1.failedMoves} (${summary1.errorRate}%)`);
    console.log(`  Success rate: ${summary1.successRate}%`);
    
    if (auditor.errors.length > 0) {
      console.log('\n  ❌ Failed moves:');
      auditor.errors.forEach(e => {
        console.log(`     Move (${e.x}, ${e.y}, ${e.rotation}) - move #${e.moveNumber}`);
      });
    }
    
    // Check grid state via browser evaluation
    console.log('\n🔍 Checking grid state integrity...');
    const gridState = await page.evaluate(() => {
      // Try to access the game's grid manager
      if (window.game && window.game.scene) {
        const gameScene = window.game.scene.getScene('GameScene');
        if (gameScene && gameScene.gridManager) {
          const gm = gameScene.gridManager;
          const legalMoves = gm.getValidMoves ? gm.getValidMoves() : null;
          return {
            hasGridManager: true,
            hasGetValidMoves: !!gm.getValidMoves,
            legalMovesCount: legalMoves ? legalMoves.length : null,
            gridCells: gm.grid ? gm.grid.length : null
          };
        }
      }
      return { hasGridManager: false };
    });
    
    console.log('  Grid state:', JSON.stringify(gridState, null, 4));
    
    // Reset for Test 2
    const auditor2 = new UIAuditor();
    
    // Go back to menu
    console.log('\n📍 Returning to menu for Test 2...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 3000));
    
    // Test 2: AI vs Heuristic
    console.log('\n🎯 Test 2: AI vs Heuristic');
    console.log('━'.repeat(50));
    await page.mouse.click(300, 540);  // AI vs Heuristic button
    await new Promise(r => setTimeout(r, 30000));
    
    // Re-attach console listener for auditor2
    page.on('console', msg => {
      const text = msg.text();
      if (text.includes('chose:')) {
        auditor2.moveCount++;
      }
      if (text.includes('failed placement')) {
        auditor2.failedMoves++;
      }
    });
    
    console.log('\n📊 Test 2 Results:');
    const summary2 = auditor2.summary();
    console.log(`  Total moves: ${summary2.totalMoves}`);
    console.log(`  Failed moves: ${summary2.failedMoves}`);
    
    // Final analysis
    console.log('\n' + '='.repeat(50));
    console.log('📋 AUDIT SUMMARY');
    console.log('='.repeat(50));
    
    console.log('\n✅ AI vs Random:');
    console.log(`   ${auditor.moveCount} moves, ${auditor.failedMoves} failures`);
    console.log(`   Success rate: ${auditor.summary().successRate}%`);
    
    if (auditor.errors.length > 0) {
      console.log('\n⚠️  MOVE FAILURES DETECTED:');
      console.log('   This could indicate:');
      console.log('   1. Random player selecting from stale legal moves list');
      console.log('   2. Race condition between move computation and grid state');
      console.log('   3. Move validation bug in UI code');
      console.log('\n   Failed positions:');
      auditor.errors.slice(0, 5).forEach(e => {
        console.log(`   - (${e.x}, ${e.y}) rot=${e.rotation} at move #${e.moveNumber}`);
      });
    }
    
    // Save detailed log
    const fs = await import('fs');
    await fs.promises.writeFile(
      'ui_audit_log.json', 
      JSON.stringify({
        test1_AI_vs_Random: {
          summary: auditor.summary(),
          errors: auditor.errors,
          log: auditor.auditLog.slice(0, 50)
        },
        gridState,
        timestamp: new Date().toISOString()
      }, null, 2)
    );
    console.log('\n💾 Detailed log saved to ui_audit_log.json');
    
  } catch (error) {
    console.error('❌ Audit failed:', error.message);
  }
  
  await browser.close();
  console.log('\n✅ Audit complete');
}

auditGameplay().catch(console.error);
