/**
 * Heuristic Player Debug - Captures errors in HeuristicPlayer
 */

import puppeteer from 'puppeteer';

async function debugHeuristicPlayer() {
  console.log('🔍 Heuristic Player Debug\n');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 600, height: 800 });

  page.on('console', msg => {
    const text = msg.text();
    const type = msg.type();
    
    // Show all logs related to heuristic
    if (text.toLowerCase().includes('heuristic') || 
        text.includes('triggerNonHuman') ||
        text.includes('ERROR')) {
      console.log(`[${type.toUpperCase().padEnd(5)}] ${text}`);
    }
    
    // Show all errors
    if (type === 'error') {
      console.log(`[ERROR] ${text}`);
    }
  });

  page.on('pageerror', error => {
    console.log(`[PAGE ERROR] ${error.message}`);
  });

  try {
    console.log('📍 Loading game...\n');
    await page.goto('http://localhost:3001/', { waitUntil: 'networkidle2' });
    await page.waitForSelector('canvas');
    await new Promise(r => setTimeout(r, 3000));
    
    console.log('🎯 Starting AI vs Heuristic (20 seconds)...\n');
    await page.mouse.click(300, 540);
    
    await new Promise(r => setTimeout(r, 20000));

  } catch (error) {
    console.error('❌ Debug failed:', error.message);
  }

  await browser.close();
}

debugHeuristicPlayer().catch(console.error);
