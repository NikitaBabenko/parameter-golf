$env:RUN_ID="local_test_gqa"
$env:ITERATIONS=30
$env:WARMUP_STEPS=5
$env:WARMDOWN_ITERS=5

# LR (defaults are tuned for 8xH100 big batch, lower for local)
$env:MATRIX_LR="0.04"
$env:SCALAR_LR="0.04"
$env:TIED_EMBED_LR="0.05"
$env:GRAD_CLIP_NORM="0.3"

# Disable QAT and torch.compile for local (no Triton on Windows)
$env:QAT_ENABLED="0"
$env:USE_COMPILE="0"

# Data paths (local data in parameter-golf_old)
$env:DATA_PATH="../parameter-golf_old/data/datasets/fineweb10B_sp1024"
$env:TOKENIZER_PATH="../parameter-golf_old/data/tokenizers/fineweb_1024_bpe.model"

# Small batch for local GPU
$env:TRAIN_BATCH_TOKENS=65536
$env:TRAIN_SEQ_LEN=1024

# Disable validation and eval
$env:VAL_LOSS_EVERY=-1
$env:VAL_BATCH_SIZE=8192
$env:TRAIN_LOG_EVERY=5
$env:SKIP_EVAL="1"

# Disable time limit
$env:MAX_WALLCLOCK_SECONDS=0

# Architecture (use defaults mostly, but can override here)
# $env:NUM_LAYERS=11
# $env:MODEL_DIM=512
# $env:NUM_HEADS=8
# $env:NUM_KV_HEADS=4
# $env:MLP_MULT="3.0"
# $env:XSA_LAST_N=4
# $env:BIGRAM_VOCAB_SIZE=4096
# $env:BIGRAM_DIM=128

# EMA off for local test speed
$env:EMA_ENABLED="0"

Write-Host "Local test: 30 steps, no eval, no QAT, no EMA"
Write-Host "------------------------------------------------------------------"

& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_gpt.py
