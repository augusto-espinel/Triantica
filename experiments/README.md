# Experiments Directory

This directory contains experiment results and artifacts.

## Structure

```
experiments/
├── EXP-001-baseline/           # First experiment
│   ├── config.json             # Parameters used
│   ├── network.json            # Trained weights
│   ├── metrics.json            # Performance summary
│   └── notes.md                # Observations
├── EXP-002-architecture/       # Architecture experiment
│   └── ...
└── experiments.csv             # Summary spreadsheet
```

## Naming Convention

Format: `EXP-NNN-description`

- `EXP-001-baseline`: Establishing baseline performance
- `EXP-002-smaller-net`: Testing smaller architecture
- `EXP-003-no-immediate`: Removing immediate reward shaping
- `EXP-004-higher-exploration`: Testing 30% exploration

## Quick Start

1. Create experiment directory: `mkdir experiments/EXP-NNN-desc`
2. Run training with metrics
3. Export metrics JSON
4. Write notes.md with observations
5. Update experiments.csv summary

## Summary CSV Format

```csv
id,date,description,architecture,win_rate_vs_random,win_rate_vs_prev,avg_decision_ms,training_error,notes
EXP-001,2026-10-18,baseline,401-200-400,,,,,
```

---

See `docs/EXPERIMENT_TEMPLATE.md` for full template.
