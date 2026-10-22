/**
 * UI Audit - Turn switching analysis
 */

import puppeteer from 'puppeteer';

async function auditTurnSwitching() {
  console.log('🔍 Turn Switching Audit\n');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });

  let turnSwitches = [];
  let lastPlayer = null;

  page.on('console', msg => {
    const text = msg.text();
    
    // Capture turn switches
    if (text.includes('Switched to player')) {
      const match = text.match(/Switched to player (\d+), type: (\w+)/);
      if (match) {
        const player = parseInt(match[1]);
        const type = match[2];
        turnSwitches.push({ player, type, time: Date.now() });
        
        if (lastPlayer !== player) {
          console.log(`🔄 Turn switch: P${player} (${type})`);
          lastPlayer = player;
        }
      }
    }

    // Capture extra turns
    if (text.includes('Same player') || text.includes('extra turn')) {
      console.log(`  ↪️  Extra turn granted`);
    }

    // Capture moves
    if (text.includes('chose:')) {
      const match = text.match(/(\w+)(?:\s+Player)?\s+(\d+) chose: \((\d+), (\d+)\), Rot: (\d+)/);
      if (match) {
        console.log(`  📍 P${match[2]} (${match[1]}): (${match[3]},${match[4]}) r${match[5]}`);
      }
    }
  });

  try {
    console.log('📍 Testing AI vs Heuristic (45 seconds)...');
    console.log('━'.repeat(50));
    
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2' });
    await page.waitForSelector('canvas');
    await new Promise(r => setTimeout(r, 3000));
    
    // Start AI vs Heuristic
    await page.mouse.click(300, 540);
    
    await new Promise(r => setTimeout(r, 45000));

    // Analyze turn switches
    console.log('\n📊 Analysis:');
    console.log(`   Total turn switches: ${turnSwitches.length}`);
    
    const player1Turns = turnSwitches.filter(t => t.player === 1).length;
    const player2Turns = turnSwitches.filter(t => t.player === 2).length;
    
    console.log(`   P1 (AI) turns: ${player1Turns}`);
    console.log(`   P2 (Heuristic) turns: ${player2Turns}`);

    if (player2Turns === 0) {
      console.log('\n⚠️  BUG: Heuristic player never got a turn!');
      console.log('   AI might be getting infinite extra turns.');
    }

  } catch (error) {
    console.error('❌ Audit failed:', error.message);
  }

  await browser.close();
}

auditTurnSwitching().catch(console.error);
