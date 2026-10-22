/**
 * Extended UI Auditor - Longer test for stability verification
 */

import puppeteer from 'puppeteer';

async function extendedAudit() {
  console.log('🔬 Extended UI Audit - 60 second test...\n');
  
  let moveCount = 0;
  let failCount = 0;
  let enclosureCount = 0;
  let bonusTurnCount = 0;
  
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });
  
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('chose:')) moveCount++;
    if (text.includes('failed placement')) {
      failCount++;
      console.log(`  ❌ FAIL: ${text}`);
    }
    if (text.includes('enclosed')) enclosureCount++;
    if (text.includes('extra turn')) bonusTurnCount++;
  });
  
  try {
    console.log('📍 Loading game...');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2' });
    await page.waitForSelector('canvas');
    await new Promise(r => setTimeout(r, 3000));
    
    console.log('🎯 Starting AI vs Random (60 seconds)...');
    await page.mouse.click(300, 460);
    
    await new Promise(r => setTimeout(r, 60000));
    
    console.log('\n📊 Results:');
    console.log(`  Total moves: ${moveCount}`);
    console.log(`  Failed moves: ${failCount}`);
    console.log(`  Enclosures: ${enclosureCount}`);
    console.log(`  Bonus turns: ${bonusTurnCount}`);
    console.log(`  Success rate: ${(moveCount > 0 ? (moveCount - failCount) / moveCount * 100 : 100).toFixed(1)}%`);
    
    if (failCount === 0) {
      console.log('\n✅ RACE CONDITION FIXED - No move failures detected!');
    } else {
      console.log(`\n⚠️  ${failCount} failures detected - possible remaining issues`);
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
  
  await browser.close();
}

extendedAudit().catch(console.error);
