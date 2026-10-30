import { defineConfig } from 'vite';
import { resolve } from 'path';
import fs from 'fs';

// Custom plugin to copy PWA assets
function copyPwaAssets() {
  return {
    name: 'copy-pwa-assets',
    closeBundle() {
      const distDir = resolve(__dirname, 'dist');
      
      // Copy manifest.json
      const manifestSrc = resolve(__dirname, 'manifest.json');
      const manifestDest = resolve(distDir, 'manifest.json');
      if (fs.existsSync(manifestSrc)) {
        fs.copyFileSync(manifestSrc, manifestDest);
        console.log('Copied manifest.json');
      }
      
      // Copy sw.js
      const swSrc = resolve(__dirname, 'sw.js');
      const swDest = resolve(distDir, 'sw.js');
      if (fs.existsSync(swSrc)) {
        fs.copyFileSync(swSrc, swDest);
        console.log('Copied sw.js');
      }
      
      // Copy game.js (main game logic, non-module)
      const gameSrc = resolve(__dirname, 'game.js');
      const gameDest = resolve(distDir, 'game.js');
      if (fs.existsSync(gameSrc)) {
        fs.copyFileSync(gameSrc, gameDest);
        console.log('Copied game.js');
      }
      
      // Copy AIPlayer.js
      const aiSrc = resolve(__dirname, 'AIPlayer.js');
      const aiDest = resolve(distDir, 'AIPlayer.js');
      if (fs.existsSync(aiSrc)) {
        fs.copyFileSync(aiSrc, aiDest);
        console.log('Copied AIPlayer.js');
      }
      
      // Copy icons directory
      const iconsSrc = resolve(__dirname, 'icons');
      const iconsDest = resolve(distDir, 'icons');
      if (fs.existsSync(iconsSrc)) {
        if (!fs.existsSync(iconsDest)) {
          fs.mkdirSync(iconsDest, { recursive: true });
        }
        const files = fs.readdirSync(iconsSrc);
        files.forEach(file => {
          fs.copyFileSync(
            resolve(iconsSrc, file),
            resolve(iconsDest, file)
          );
        });
        console.log(`Copied ${files.length} icon files`);
      }
      
      // Copy src/players directory (for RandomPlayer, HeuristicPlayer imports)
      const playersSrc = resolve(__dirname, 'src/players');
      const playersDest = resolve(distDir, 'src/players');
      if (fs.existsSync(playersSrc)) {
        if (!fs.existsSync(playersDest)) {
          fs.mkdirSync(playersDest, { recursive: true });
        }
        const files = fs.readdirSync(playersSrc);
        files.forEach(file => {
          fs.copyFileSync(
            resolve(playersSrc, file),
            resolve(playersDest, file)
          );
        });
        console.log(`Copied ${files.length} player files`);
      }
    }
  };
}

export default defineConfig({
  root: '.',
  base: './',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    port: 3000,
    open: true,
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  plugins: [copyPwaAssets()],
});
