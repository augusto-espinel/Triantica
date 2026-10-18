# Triantica Quick Start

## Prerequisites

- Node.js 18+ 
- npm

## Installation

```bash
cd ~/.openclaw/workspace/triantica
npm install
```

## Running Experiments

### Option 1: CLI (Recommended for batch experiments)

```bash
# Run 100 games vs random opponent
npm run experiment -- --type vs-random --count 100

# Run 50 self-play games
npm run experiment -- --type self-play --count 50

# Custom output filename
npm run experiment -- --type vs-random --count 100 --output baseline.json
```

Results saved to `experiments/data/`.

### Option 2: Local Server

```bash
# Start server on port 8080
npm run serve

# In another terminal, run experiment via API
curl -X POST http://localhost:8080/api/experiment \
  -H "Content-Type: application/json" \
  -d '{"type":"vs-random","count":50}'
```

### Option 3: Browser (GitHub Pages)

1. Deploy to GitHub Pages: `git push origin main`
2. Open `https://[username].github.io/Triantica/`
3. Open browser console (F12)
4. Run: `await runExperiment('vs-random', 100)`
5. Results download automatically as JSON

## Viewing Results

```bash
# List experiments
ls experiments/data/

# View summary (requires jq)
cat experiments/data/exp-*.json | jq '.summary'
```

## Game Controls (Browser Mode)

| Key | Action |
|-----|--------|
| Click | Place triangle |
| R / Space | Rotate preview |
| Mouse wheel | Rotate preview |

## Project Structure

```
triantica/
├── src/
│   ├── core/          # Game logic (platform-agnostic)
│   ├── players/      # AI and human players
│   ├── data/         # Logging and export
│   └── server/       # Node.js server
├── browser/          # Browser-only code
├── docs/             # Documentation
└── experiments/      # Generated data
```

## Documentation

| Document | Purpose |
|----------|---------|
| [TRAINING_BOOK.md](docs/TRAINING_BOOK.md) | Main experiment journal |
| [AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) | Neural network reference |
| [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Game rules cheat sheet |

## Troubleshooting

**Port 8080 already in use:**
```bash
PORT=3000 npm run serve
```

**Module not found errors:**
```bash
npm install
```

**CORS errors in browser:**
Use local server instead of file:// protocol:
```bash
npm run dev
```
