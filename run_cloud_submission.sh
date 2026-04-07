#!/bin/bash
# Leaderboard submission — 8xH100
# Usage: Edit the env vars below with best config from experiments, then:
#   bash run_cloud_submission.sh

set -e

echo "=== SETUP ==="
cd /workspace
if [ ! -d parameter-golf ]; then
    git clone https://github.com/NikitaBabenko/parameter-golf.git
fi
cd parameter-golf

if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Downloading data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

echo ""
echo "=== LEADERBOARD SUBMISSION (8xH100, 10-min cap) ==="
echo ""

# TODO: fill in best env vars from experiment results
RUN_ID="submission_v28" \
MOE_NUM_EXPERTS=0 \
NGRAM_EVAL=0 \
TTT_ENABLED=0 \
RYS_LAYERS="" \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== SUBMISSION DONE ==="
grep -E "final_int6_sliding_window|advanced_eval|Total submission size" logs/submission_v28.txt 2>/dev/null || echo "(check logs manually)"
