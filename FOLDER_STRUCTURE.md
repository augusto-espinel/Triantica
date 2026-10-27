# Triantica Folder Structure

Last updated: 2026-10-27

## Deployment (`main` branch root `/`)

GitHub Pages serves directly from the `main` branch root (`/`):

```
main/
├── index.html                    # Game (playable)
├── game.js                       # Game logic
├── trainedAI_network.json        # Deployed AI network weights
├── .nojekyll                     # Prevent Jekyll processing
│
└── dashboard/                    # Training Dashboard (served at /dashboard/)
    ├── index.html                # Dashboard entry point
    ├── experiments.json          # Experiment metadata
    └── data/                     # Training run data & checkpoints
        └── TF-B*.json            # Individual training run files
```

### URLs
- **Game:** `https://augusto-espinel.github.io/Triantica/`
- **Dashboard:** `https://augusto-espinel.github.io/Triantica/dashboard/`

## Source Code (`main` branch)

```
triantica/
├── dashboard/                     # Dashboard source (served at /dashboard/)
│   ├── index.html
│   ├── experiments.json
│   └── data/
├── docs/                          # Project documentation & experiment journals
│   ├── TRAINING_BOOK.md
│   └── QUIESCENCE_MCTS.md
├── experiments/                   # Training scripts & artifacts
│   └── backups/                   # Network checkpoints
├── dist/                          # Vite build output (game)
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
# Validate and clean dashboard/data/ against dashboard/experiments.json:
./scripts/publish-dashboard.sh

# Commit and push to main (GitHub Pages deploys automatically from main /):
git add dashboard/
git commit -m "Update dashboard data"
git push origin main
```

## Data Flow

### Training Output
```
train_tfjs.cjs → dashboard/data/TF-B12.json
               → dashboard/data/TF-B12_network.json
```

### Key Files
| File | Purpose | Edit |
|------|---------|------|
| `dashboard/index.html` | Dashboard HTML/JS | Yes |
| `dashboard/experiments.json` | Experiment summaries | Yes (after training) |
| `dashboard/data/*.json` | Training run data | No (auto-generated) |
| `src/server/train_tfjs.cjs` | Training pipeline | Yes |
| `docs/TRAINING_BOOK.md` | Experiment journal | Yes |

## Historical Context

**2026-10-27:** Consolidated dashboard hosting exclusively under `/dashboard/` (`dashboard/` directory) while keeping `docs/` purely for documentation (`TRAINING_BOOK.md`, `QUIESCENCE_MCTS.md`, etc.). GitHub Pages deploys directly from `main` branch `/`.
