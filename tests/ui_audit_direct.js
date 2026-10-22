/**
 * Direct UI Audit - Captures move legality from console logs
 */

import puppeteer from 'puppeteer';
import fs from 'fs';

class DirectUIAudit {
  constructor() {
    this.moves = [];
    this.failures = [];
    this.legalMovesSeen = [];
    this.enclosureCount = 0;
  }

  parseLog(text) {
    // Parse move attempts
    if (text.includes('chose:')) {
      const match = text.match(/(\w+)(?:\s+Player)?\s+(\d+) chose: \((\d+), (\d+)\), Rot: (\d+)/);
      if (match) {
        const move = {
          playerType: match[1],
          player: parseInt(match[2]),
          x: parseInt(match[3]),
          y: parseInt(match[4]),
          rotation: parseInt(match[5])
        };
        this.moves.push({ ...move, success: null });
        console.log(`\n🎯 Move ${this.moves.length}: P${move.player} (${move.playerType}) → (${move.x},${move.y}) r${move.rotation}`);
      }
    }

    // Parse move failures
    if (text.includes('failed placement') || text.includes('Error: Move')) {
      const match = text.match(/Move \((\d+), (\d+), (\d+)\) failed/);
      if (match) {
        const failMove = {
          x: parseInt(match[1]),
          y: parseInt(match[2]),
          rotation: parseInt(match[3])
        };
        this.failures.push(failMove);
        // Mark last move as failed
        if (this.moves.length > 0) {
          this.moves[this.moves.length - 1].success = false;
        }
        console.log(`   ❌ FAILED: (${failMove.x},${failMove.y}) r${failMove.rotation}`);
      }
    }

    // Parse enclosures
    if (text.includes('enclosed') && text.includes('extra turn')) {
      this.enclosureCount++;
      console.log(`   🔺 Enclosure! (+extra turn)`);
    }

    // Parse legal moves from debug output
    if (text.includes('valid moves') || text.includes('Legal moves')) {
      const countMatch = text.match(/(\d+)\s+valid moves/);
      if (countMatch) {
        const count = parseInt(countMatch[1]);
        this.legalMovesSeen.push(count);
        console.log(`   📋 Legal moves: ${count}`);
      }
    }
  }

  summary() {
    const successful = this.moves.filter(m => m.success !== false).length;
    const failed = this.moves.filter(m => m.success === false).length;
    return {
      totalMoves: this.moves.length,
      successfulMoves: successful,
      failedMoves: failed,
      successRate: this.moves.length > 0 ? (successful / this.moves.length * 100).toFixed(1) : 100,
      enclosureCount: this.enclosureCount,
      avgLegalMoves: this.legalMovesSeen.length > 0 ? 
        (this.legalMovesSeen.reduce((a,b) => a+b, 0) / this.legalMovesSeen.length).toFixed(1) : 'N/A',
      minLegalMoves: this.legalMovesSeen.length > 0 ? Math.min(...this.legalMovesSeen) : 'N/A',
      maxLegalMoves: this.legalMovesSeen.length > 0 ? Math.max(...this.legalMovesSeen) : 'N/A'
    };
  }
}

async function auditUIGameplay() {
  console.log('🔍 Direct UI Audit - Move Legality Analysis\n');
  console.log('━'.repeat(60));

  const audit = new DirectUIAudit();

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });

  // Capture all console messages
  page.on('console', msg => {
    audit.parseLog(msg.text());
  });

  page.on('pageerror', error => {
    console.log(`   ⚠️  JS ERROR: ${error.message.split('\n')[0]}`);
  });

  try {
    console.log('📍 Loading game...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2', timeout: 30000 });
    await page.waitForSelector('canvas');
    await new Promise(r => setTimeout(r, 3000));

    console.log('\n🎯 Test 1: AI vs Random (30 seconds)');
    console.log('━'.repeat(60));
    await page.mouse.click(300, 460);
    await new Promise(r => setTimeout(r, 30000));

    const summary1 = audit.summary();
    console.log('\n📊 Test 1 Results:');
    console.log(`   Total moves: ${summary1.totalMoves}`);
    console.log(`   Success rate: ${summary1.successRate}%`);
    console.log(`   Enclosures: ${summary1.enclosureCount}`);
    console.log(`   Avg legal moves per turn: ${summary1.avgLegalMoves}`);
    console.log(`   Legal moves range: ${summary1.minLegalMoves} - ${summary1.maxLegalMoves}`);

    // Test 2: AI vs Heuristic
    console.log('\n\n🎯 Test 2: AI vs Heuristic (30 seconds)');
    console.log('━'.repeat(60));
    const audit2 = new DirectUIAudit();
    page.on('console', msg => audit2.parseLog(msg.text()));

    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2' });
    await new Promise(r => setTimeout(r, 3000));
    await page.mouse.click(300, 540);  // AI vs Heuristic button
    await new Promise(r => setTimeout(r, 30000));

    const summary2 = audit2.summary();
    console.log('\n📊 Test 2 Results:');
    console.log(`   Total moves: ${summary2.totalMoves}`);
    console.log(`   Success rate: ${summary2.successRate}%`);
    console.log(`   Enclosures: ${summary2.enclosureCount}`);

    // Final summary
    console.log('\n' + '='.repeat(60));
    console.log('📋 OVERALL AUDIT SUMMARY');
    console.log('='.repeat(60));
    console.log(`\n✅ AI vs Random: ${summary1.totalMoves} moves, ${summary1.successRate}% success`);
    console.log(`✅ AI vs Heuristic: ${summary2.totalMoves} moves, ${summary2.successRate}% success`);

    if (summary1.failedMoves > 0 || summary2.failedMoves > 0) {
      console.log('\n⚠️  ISSUES DETECTED:');
      console.log('   Race conditions or move validation bugs present');
      console.log('   Check failed moves in ui_audit_log.json');
    } else {
      console.log('\n✅ ALL MOVES VALID - No race conditions detected');
    }

    // Save log
    await fs.promises.writeFile('ui_audit_log.json', JSON.stringify({
      test1_AI_vs_Random: summary1,
      test2_AI_vs_Heuristic: summary2,
      timestamp: new Date().toISOString()
    }, null, 2));

  } catch (error) {
    console.error('❌ Audit failed:', error.message);
  }

  await browser.close();
  console.log('\n✅ Audit complete');
}

auditUIGameplay().catch(console.error);
