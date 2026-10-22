/**
 * Headless browser test for UI race condition
 * 
 * Tests the triggerNonHumanMove timing issue by:
 * 1. Starting a game with AI players  
 * 2. Observing the game loop for errors
 * 3. Checking for proper AI move execution
 */

import puppeteer from 'puppeteer';

async function testUI() {
  console.log('🚀 Launching headless browser...');
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  
  // Set viewport to match game resolution
  await page.setViewport({ width: 600, height: 800 });
  
  // Capture ALL console logs
  const logs = [];
  page.on('console', msg => {
    const text = msg.text();
    logs.push({ type: msg.type(), text, time: Date.now() });
    console.log(`[${msg.type().toUpperCase().padEnd(6)}] ${text}`);
  });
  
  // Capture errors
  page.on('pageerror', error => {
    logs.push({ type: 'pageerror', text: `PAGE ERROR: ${error.message}`, time: Date.now() });
    console.error(`[PAGE ERROR] ${error.message}`);
  });
  
  console.log('📍 Navigating to http://localhost:3001/...');
  
  try {
    await page.goto('http://localhost:3001/', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });
    
    console.log('✅ Page loaded');
    
    // Wait for Phaser
    await page.waitForSelector('canvas', { timeout: 10000 });
    console.log('✅ Canvas found');
    
    // Wait for game to be ready
    await new Promise(r => setTimeout(r, 3000));
    
    // Click on "AI vs Random" button
    // Canvas is 600x800, centerX=300, centerY=400
    // AI vs Random is at centerY + 60 = 460
    console.log('\n🎯 Clicking "AI vs Random" at (300, 460)...');
    await page.mouse.click(300, 460);
    
    // Wait for game to start
    await new Promise(r => setTimeout(r, 2000));
    
    // Take screenshot
    await page.screenshot({ path: 'test_after_click.png' });
    console.log('📸 Screenshot after click saved');
    
    // Now observe AI game
    console.log('\n⏱️  Watching AI game for 20 seconds...');
    
    await new Promise(r => setTimeout(r, 20000));
    
    // Final screenshot
    await page.screenshot({ path: 'test_final_state.png' });
    console.log('📸 Final screenshot saved');
    
    // Analyze
    console.log('\n📊 Analysis:');
    console.log(`  Total logs: ${logs.length}`);
    
    const errors = logs.filter(l => l.type === 'error' || l.type === 'pageerror');
    const warnings = logs.filter(l => l.type === 'warning');
    const moveLogs = logs.filter(l => l.text.toLowerCase().includes('move'));
    const turnLogs = logs.filter(l => l.text.toLowerCase().includes('turn'));
    
    console.log(`  Errors: ${errors.length}`);
    console.log(`  Warnings: ${warnings.length}`);
    console.log(`  Move logs: ${moveLogs.length}`);
    console.log(`  Turn logs: ${turnLogs.length}`);
    
    if (errors.length > 0) {
      console.log('\n❌ Errors found:');
      errors.forEach(e => console.log(`  [${e.type}] ${e.text}`));
    }
    
    if (moveLogs.length > 0 || turnLogs.length > 0) {
      console.log('\n📝 Game activity:');
      [...moveLogs, ...turnLogs].forEach(l => console.log(`  ${l.text}`));
    }
    
    // Check if game started by looking for game-related logs
    const gameStarted = logs.some(l => 
      l.text.includes('GameScene') || 
      l.text.includes('grid') ||
      l.text.includes('triangle')
    );
    
    console.log(`\n🎮 Game started: ${gameStarted ? 'YES' : 'NO'}`);
    
    if (!gameStarted) {
      console.log('\n⚠️  Game did not start - click may not have registered');
      console.log('    Trying to get more info about game state...');
      
      const gameState = await page.evaluate(() => {
        return {
          hasPhaser: typeof Phaser !== 'undefined',
          url: window.location.href,
          canvasCount: document.querySelectorAll('canvas').length
        };
      });
      
      console.log('    State:', gameState);
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    await page.screenshot({ path: 'test_error.png' });
  }
  
  await browser.close();
  console.log('\n✅ Done');
}

testUI().catch(console.error);
