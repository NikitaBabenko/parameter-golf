#!/bin/bash
# Все эксперименты V27 + V28, последовательно на 1xH100
# Usage: bash run_cloud.sh 2>&1 | tee full_output.log
#
# Порядок:
#   1. Run A — baseline (стандартная архитектура) + N-gram eval
#   2. RYS eval-only (4 варианта на чекпоинте Run A)
#   3. Run B — MoE (4 эксперта) + N-gram eval
#   4. Сводка всех результатов
#
# Время: ~70-80 мин на 1xH100
# Стоимость: ~$3-4

set -e

echo "============================================"
echo "  PARAMETER GOLF: V27 + V28 EXPERIMENTS"
echo "  $(date)"
echo "============================================"

# --- SETUP ---
echo ""
echo "=== SETUP ==="
# Работаем из папки, где лежат train_gpt.py и run_cloud.sh
cd "$(dirname "$0")"
echo "Working dir: $(pwd)"
echo "train_gpt.py: $(ls -l train_gpt.py 2>/dev/null || echo 'NOT FOUND!')"
mkdir -p logs data

# Скачиваем данные, если их нет
if [ ! -f data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin ]; then
    echo "Data not found, downloading..."
    pip install huggingface-hub datasets sentencepiece 2>/dev/null
    python3 data/cached_challenge_fineweb.py --variant sp1024
    echo "Data downloaded."
else
    echo "Data already present."
fi

# --- SMOKE TEST ---
echo ""
echo "=== SMOKE TEST ==="
RUN_ID=smoke ITERATIONS=10 WARMUP_STEPS=3 WARMDOWN_ITERS=2 \
TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 SKIP_EVAL=1 \
MAX_WALLCLOCK_SECONDS=0 python train_gpt.py
echo "Smoke OK."

# =============================================
# 1. RUN A — baseline (стандартная архитектура)
#    Даёт: v27a baseline BPB, v27b N-gram BPB, чекпоинт для RYS
# =============================================
echo ""
echo "============================================"
echo "  1/3  RUN A: BASELINE + NGRAM (5k steps)"
echo "  Started: $(date)"
echo "  Expected: ~25-30 min"
echo "============================================"

RUN_ID="run_a_std_5k" \
ITERATIONS=5000 WARMUP_STEPS=50 WARMDOWN_ITERS=500 \
TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=1000 \
GRAD_CLIP_NORM=0.3 MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
MUON_WD=0.02 ADAM_WD=0.01 QAT_ENABLED=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
MTP_ENABLED=1 MTP_N=2 MTP_ALPHA=0.2 USE_LEAKY_RELU=1 \
MOE_NUM_EXPERTS=0 NGRAM_EVAL=1 NGRAM_ORDER=11 NGRAM_BUCKETS=4194304 \
NGRAM_ALPHA=0.4 NGRAM_ENTROPY_ADAPTIVE=1 TTT_ENABLED=0 \
SKIP_EVAL=0 \
python train_gpt.py

echo ""
echo "Run A done: $(date)"
echo "Checkpoint saved: final_model.int6.ptz"

# Сохраняем чекпоинт Run A отдельно (Run B перезапишет)
cp final_model.int6.ptz final_model_run_a.int6.ptz
cp final_model.pt final_model_run_a.pt

# =============================================
# 2. RYS EVAL-ONLY (на чекпоинте Run A)
# =============================================
echo ""
echo "============================================"
echo "  2/3  V28 RYS EVAL-ONLY (4 variants)"
echo "  Started: $(date)"
echo "  Expected: ~20 min"
echo "============================================"

echo ""
echo "--- v28b: RYS layer 5 (1 decoder layer) ---"
RUN_ID="v28b_rys_5" \
EVAL_ONLY="final_model_run_a.int6.ptz" RYS_LAYERS="5" \
NGRAM_EVAL=0 TTT_ENABLED=0 \
python train_gpt.py

echo ""
echo "--- v28c: RYS layers 5,6 (2 decoder layers) ---"
RUN_ID="v28c_rys_56" \
EVAL_ONLY="final_model_run_a.int6.ptz" RYS_LAYERS="5,6" \
NGRAM_EVAL=0 TTT_ENABLED=0 \
python train_gpt.py

echo ""
echo "--- v28d: RYS layers 5,6,7 (3 decoder layers) ---"
RUN_ID="v28d_rys_567" \
EVAL_ONLY="final_model_run_a.int6.ptz" RYS_LAYERS="5,6,7" \
NGRAM_EVAL=0 TTT_ENABLED=0 \
python train_gpt.py

echo ""
echo "--- v28e: RYS layers 2,3,4 (encoder control) ---"
RUN_ID="v28e_rys_234" \
EVAL_ONLY="final_model_run_a.int6.ptz" RYS_LAYERS="2,3,4" \
NGRAM_EVAL=0 TTT_ENABLED=0 \
python train_gpt.py

echo ""
echo "RYS eval done: $(date)"

# =============================================
# 3. RUN B — MoE (4 эксперта)
# =============================================
echo ""
echo "============================================"
echo "  3/3  RUN B: MOE 4 EXPERTS + NGRAM (5k steps)"
echo "  Started: $(date)"
echo "  Expected: ~25-30 min"
echo "============================================"

RUN_ID="run_b_moe4_5k" \
ITERATIONS=5000 WARMUP_STEPS=50 WARMDOWN_ITERS=500 \
TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=100 VAL_LOSS_EVERY=1000 \
GRAD_CLIP_NORM=0.3 MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
MUON_WD=0.02 ADAM_WD=0.01 QAT_ENABLED=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
MTP_ENABLED=1 MTP_N=2 MTP_ALPHA=0.2 USE_LEAKY_RELU=1 \
MOE_NUM_EXPERTS=4 NGRAM_EVAL=1 NGRAM_ORDER=11 NGRAM_BUCKETS=4194304 \
NGRAM_ALPHA=0.4 NGRAM_ENTROPY_ADAPTIVE=1 TTT_ENABLED=0 \
SKIP_EVAL=0 \
python train_gpt.py

echo ""
echo "Run B done: $(date)"

# =============================================
# RESULTS SUMMARY
# =============================================
echo ""
echo "============================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "============================================"
echo ""
echo "=== RUN A: BASELINE (standard arch) ==="
grep -E "train_loss|final_int6_sliding_window|advanced_eval|Total submission size" logs/run_a_std_5k.txt 2>/dev/null | tail -10 || echo "(check logs/run_a_std_5k.txt)"
echo ""
echo "=== V28 RYS EVAL-ONLY ==="
for f in v28b_rys_5 v28c_rys_56 v28d_rys_567 v28e_rys_234; do
    echo "--- $f ---"
    grep -E "sliding_window|advanced_eval" "logs/${f}.txt" 2>/dev/null || echo "(check logs/${f}.txt)"
done
echo ""
echo "=== RUN B: MOE 4 EXPERTS ==="
grep -E "train_loss|final_int6_sliding_window|advanced_eval|Total submission size" logs/run_b_moe4_5k.txt 2>/dev/null | tail -10 || echo "(check logs/run_b_moe4_5k.txt)"
echo ""
echo "============================================"
echo "  LOGS: logs/*.txt"
echo "  CHECKPOINTS: final_model_run_a.int6.ptz, final_model.int6.ptz (run B)"
echo "  FULL OUTPUT: full_output.log (if used tee)"
echo "============================================"
