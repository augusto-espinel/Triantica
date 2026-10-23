# Triantica Folder Structure

Last updated: 2026-10-23

---

## Directory Layout

```
triantica/
├── docs/                          # GitHub Pages (auto-served)
│   ├── index.html                 # Dashboard entry point
│   ├── experiments.json           # Experiment metadata (manually updated)
│   └── data/                      # Training run data (auto-generated)
│       └── training-*.json        # Individual training run files
│
├── experiments/                   # Training artifacts
│   ├── logs/                      # Training logs (auto-generated)
│   │   └── *.log                  # Human-readable logs
│   └── backups/                   # Network checkpoints (manual)
│       └── exp*/                  # Per-experiment backups
│
├── dist/                          # Vite build output (game)
│   ├── index.html                 # Game entry point
│   ├── assets/                    # JS/CSS bundles
│   └── models/                    # Pre-trained networks
│
├── public/                        # Static assets (dev)
│   └── models/                    # Network files
│
├── browser/                       # Browser-specific code
├── src/                           # Source code
│   ├── core/                      # Game logic (GridManager)
│   ├── players/                   # Player implementations
│   ├── server/                    # Training pipeline
│   └── data/                      # Data utilities
│
└── tests/                         # Test files
```

---

## GitHub Pages

**URL:** `https://augusto-espinel.github.io/Triantica/`

**Serves from:** `docs/` directory

**Entry point:** `docs/index.html` (Training Dashboard)

### What's Served

| Path | Content | Notes |
|------|---------|-------|
| `/` | Dashboard | Interactive training metrics |
| `/experiments.json` | Metadata | Experiment summaries |
| `/data/*.json` | Run data | Full training cycles |

---

## Data Flow

### Training Output

```
train.js → docs/data/training-TIMESTAMP.json
         → experiments/logs/training-TIMESTAMP.log
```

Training data goes directly to `docs/data/` so it's immediately available on GitHub Pages after push.

### Dashboard Update

```bash
# After training, update experiment metadata:
# 1. Edit docs/experiments.json (add new entry)
# 2. git add docs/ && git commit -m "Add exp N results"
# 3. git push
```

---

## Key Files

| File | Purpose | Edit |
|------|---------|------|
| `docs/index.html` | Dashboard HTML/JS | Yes (dashboard changes) |
| `docs/experiments.json` | Experiment summaries | Yes (after training) |
| `docs/data/*.json` | Training run data | No (auto-generated) |
| `experiments/logs/*.log` | Human-readable logs | No (auto-generated) |
| `src/server/train.js` | Training pipeline | Yes (algorithm changes) |
| `docs/TRAINING_BOOK.md` | Experiment journal | Yes (documentation) |

---

## Common Tasks

### Run Training

```bash
node src/server/train.js --cycles 10 --output docs/data/expN.json
```

### Update Dashboard

1. Add entry to `docs/experiments.json`
2. Commit and push

### View Dashboard

1. Open `https://augusto-espinel.github.io/Triantica/`
2. Or locally: `npx serve docs`

---

## Historical Context

**2026-10-23:** Restructured to consolidate GitHub Pages assets. Previously had duplicate `docs/dashboard/`, `docs/experiments/`, and root `dashboard/` directories. Unified to single `docs/` structure with training data written directly to `docs/data/`.
