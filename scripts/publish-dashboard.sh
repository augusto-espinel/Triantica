#!/usr/bin/env bash
#
# publish-dashboard.sh — Validate and clean dashboard/data/ against dashboard/experiments.json
#
# Usage:
#   ./scripts/publish-dashboard.sh              # Verify all experiments listed in dashboard/experiments.json and clean stale files
#   ./scripts/publish-dashboard.sh <src_file>   # Copy an external experiment file into dashboard/data/ and clean stale files
#
# The dashboard is hosted exclusively at /dashboard/ and reads:
#   - dashboard/experiments.json
#   - dashboard/data/*.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DASHBOARD_DIR="$PROJECT_DIR/dashboard"
DASHBOARD_DATA_DIR="$DASHBOARD_DIR/data"

mkdir -p "$DASHBOARD_DATA_DIR"

# If specific external file paths are passed as arguments, copy them into dashboard/data/
if [ $# -gt 0 ]; then
    for arg in "$@"; do
        if [ -f "$arg" ]; then
            base="$(basename "$arg")"
            cp "$arg" "$DASHBOARD_DATA_DIR/$base"
            echo "✅ Copied $arg -> dashboard/data/$base"
            net_arg="${arg%.json}_network.json"
            if [ -f "$net_arg" ]; then
                net_base="$(basename "$net_arg")"
                cp "$net_arg" "$DASHBOARD_DATA_DIR/$net_base"
                echo "✅ Copied $net_arg -> dashboard/data/$net_base"
            fi
        fi
    done
fi

# Read all dataFiles from dashboard/experiments.json
EXPS=$(python3 -c "
import json
with open('$DASHBOARD_DIR/experiments.json') as f:
    data = json.load(f)
for e in data:
    df = e.get('dataFile', '')
    if df:
        print(df)
")

VALID_FILES=""
EXP_COUNT=0

for data_file in $EXPS; do
    EXP_COUNT=$((EXP_COUNT + 1))
    dst="$DASHBOARD_DATA_DIR/$data_file"

    if [ -f "$dst" ]; then
        VALID_FILES="$VALID_FILES data/$data_file"
        echo "✅ Verified dashboard/data/$data_file"

        # Also preserve network weights and logs/audits associated with this experiment
        prefix="${data_file%.json}"
        for extra in "$DASHBOARD_DATA_DIR/${prefix}"_* "$DASHBOARD_DATA_DIR/${prefix}.log"; do
            if [ -f "$extra" ]; then
                VALID_FILES="$VALID_FILES data/$(basename "$extra")"
            fi
        done
    else
        echo "⚠️  Missing in dashboard/data/: $data_file"
    fi
done

# Clean up stale files in dashboard/data/ that aren't in the manifest
if [ -d "$DASHBOARD_DATA_DIR" ]; then
    for f in "$DASHBOARD_DATA_DIR"/*; do
        [ -e "$f" ] || continue
        fname="data/$(basename "$f")"
        if ! echo "$VALID_FILES" | grep -qw "$fname"; then
            rm "$f"
            echo "🗑️  Removed stale: $fname"
        fi
    done
fi

echo ""
echo "📊 Dashboard ready at: /dashboard/ (dashboard/)"
echo "   Manifest experiments: $EXP_COUNT"
echo "   Run 'git add dashboard/ && git commit && git push' to publish"
