# Triantica Folder Structure

Last updated: 2026-10-23

## Deployment (gh-pages branch)

The `gh-pages` branch contains the deployed site:

```
gh-pages/
├── index.html                    # Game (playable)
├── game.js                       # Game logic
├── assets/                       # Vite-bundled JS
├── models/network.json           # AI model
├── trainedAI_network.json         # AI network weights
├── .nojekyll                     # Prevent Jekyll processing
│
└── docs/                         # Training Dashboard
    ├── index.html                # Dashboard entry point
    ├── experiments.json          # Experiment metadata
    └── data/                     # Training run data
        └── training-*.json       # Individual training run files
```

### URLs
- **Game:** `https://augusto-espinel.github.io/Triantica/`
- **Dashboard:** `https://augusto-espinel.github.io/Triantica/docs/`

## Source Code (main branch)

```
triantica/
├── docs/                          # Dashboard source (deployed to gh-pages/docs/)
│   ├── index.html
│   ├── experiments.json
│   └── data/
├── experiments/                   # Training artifacts
│   ├── logs/                      # Training logs
│   └── backups/                   # Network checkpoints
├── dist/                          # Vite build output (game)
├── data/                          # Training data (also in docs/data/)
├── public/models/                 # Network files
├── browser/                       # Browser-specific code
├── src/                           # Source code
│   ├── core/                      # Game logic
│   ├── players/                   # Player implementations
│   ├── server/                    # Training pipeline
│   └── data/                      # Data utilities
└── tests/                         # Test files
```

## Deployment Workflow

```bash
# After changes on main, update gh-pages:
git checkout gh-pages
# Copy game from dist/ and dashboard from docs/
git checkout main -- dist/
# ... arrange files and commit
git push origin gh-pages
```

## Data Flow

### Training Output
```
train.js → docs/data/training-TIMESTAMP.json
         → experiments/logs/training-TIMESTAMP.log
```

### Key Files
| File | Purpose | Edit |
|------|---------|------|
| `docs/index.html` | Dashboard HTML/JS | Yes |
| `docs/experiments.json` | Experiment summaries | Yes (after training) |
| `docs/data/*.json` | Training run data | No (auto-generated) |
| `src/server/train.js` | Training pipeline | Yes |
| `docs/TRAINING_BOOK.md` | Experiment journal | Yes |

## Historical Context

**2026-10-23:** Restructured for GitHub Pages with dual serving: game at root `/`, dashboard at `/docs/`. Uses `gh-pages` orphan branch for deployment (GitHub Pages API only supports `gh-pages`, `master`, or `master /docs` as sources).
