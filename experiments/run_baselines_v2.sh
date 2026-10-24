#!/bin/bash
set -euo pipefail

# Runner script for corrected TF.js baseline experiments
# All 3 experiments include eval vs Heuristic

cd "$(dirname "$0")/.."
eval "$(fnm env)"
fnm use v20.20.2

echo "=========================================="
echo "🧪 CORRECTED TF.js BASELINE EXPERIMENTS"
echo "=========================================="
echo ""

# TF-B1: Pure self-play (network vs network only)
# + eval vs Random + eval vs Heuristic
echo "=== TF-B1: Pure Self-Play ==="
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 10 \
  --learning-rate 0.05 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix network \
  --audit true \
  --audit-rate 0.05 \
  --output docs/data/TF-B1_pure.json

echo ""
echo "=== TF-B2: Mixed Opponents (50/30/20) ==="
# TF-B2: Mixed opponents - 50% network, 30% random, 20% heuristic
# Curriculum: learn from easy opponents, exposed to hard ones
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 10 \
  --learning-rate 0.05 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix mixed \
  --network-ratio 0.5 \
  --random-ratio 0.3 \
  --heuristic-ratio 0.2 \
  --audit true \
  --audit-rate 0.05 \
  --output docs/data/TF-B2_mixed.json

echo ""
echo "=== TF-B3: Cluster Reward + Network/Random Mix (70/30) ==="
# TF-B3: Cluster reward TD(λ) + mixed network/random opponents
# NO heuristic in training - only in eval to measure Heuristic Wall progress
node src/server/train_tfjs.cjs \
  --cycles 20 \
  --games 100 \
  --eval-games 100 \
  --iterations 10 \
  --learning-rate 0.05 \
  --exploration 0.3 \
  --self-play-mode network \
  --opponent-mix mixed \
  --network-ratio 0.7 \
  --random-ratio 0.3 \
  --heuristic-ratio 0.0 \
  --enable-cluster-reward true \
  --audit true \
  --audit-rate 0.05 \
  --output docs/data/TF-B3_cluster.json

echo ""
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETE"
echo "=========================================="
