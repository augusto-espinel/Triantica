#!/usr/bin/env bash
#
# publish-dashboard.sh — Copy dashboard experiment data from docs/ to dashboard/
#
# Usage:
#   ./scripts/publish-dashboard.sh              # Copy all experiments listed in experiments.json
#   ./scripts/publish-dashboard.sh TF-B8-V2      # Copy specific experiment(s)
#
# The dashboard at /dashboard/ reads experiments.json and data/*.json
# This script ensures the dashboard folder only contains the experiments
# that should be visible, keeping it clean and small.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DASHBOARD_DIR="$PROJECT_DIR/dashboard"
DOCS_DIR="$PROJECT_DIR/docs"
DOCS_DATA_DIR="$DOCS_DIR/data"
DASHBOARD_DATA_DIR="$DASHBOARD_DIR/data"

# Create dashboard data directory
mkdir -p "$DASHBOARD_DATA_DIR"

# Copy experiments manifest
cp "$DOCS_DIR/experiments.json" "$DASHBOARD_DIR/experiments.json"
echo "✅ Copied experiments.json"

# Determine which experiments to publish
if [ $# -gt 0 ]; then
    # Specific experiments requested
    EXPS=("$@")
else
    # All experiments from manifest
    EXPS=$(python3 -c "
import json
with open('$DOCS_DIR/experiments.json') as f:
    data = json.load(f)
for e in data:
    df = e.get('dataFile', '')
    if df:
        print(df)
")
fi

# Track what we copy so we can clean up stale files
COPIED_FILES="experiments.json"

for data_file in $EXPS; do
    src="$DOCS_DATA_DIR/$data_file"
    dst="$DASHBOARD_DATA_DIR/$data_file"

    if [ -f "$src" ]; then
        cp "$src" "$dst"
        COPIED_FILES="$COPIED_FILES data/$data_file"
        echo "✅ Copied $data_file"

        # Also copy network weights if they exist
        network_file="${data_file%.json}_network.json"
        network_src="$DOCS_DATA_DIR/$network_file"
        network_dst="$DASHBOARD_DATA_DIR/$network_file"
        if [ -f "$network_src" ]; then
            cp "$network_src" "$network_dst"
            COPIED_FILES="$COPIED_FILES data/$network_file"
            echo "✅ Copied $network_file (network weights)"
        fi
    else
        echo "⚠️  Missing: $data_file"
    fi
done

# Clean up stale files in dashboard/data/ that aren't in the manifest
if [ -d "$DASHBOARD_DATA_DIR" ]; then
    for f in "$DASHBOARD_DATA_DIR"/*; do
        fname="data/$(basename "$f")"
        if ! echo "$COPIED_FILES" | grep -qw "$fname"; then
            rm "$f"
            echo "🗑️  Removed stale: $fname"
        fi
    done
fi

echo ""
echo "📊 Dashboard ready at: dashboard/"
echo "   Experiments: $(echo "$EXPS" | wc -w | tr -d ' ')"
echo "   Run 'git add dashboard/ && git commit && git push' to publish"
