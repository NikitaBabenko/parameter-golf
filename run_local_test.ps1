$env:RUN_ID="local_test_ngram"
$env:ITERATIONS=10
$env:WARMUP_STEPS=3
$env:WARMDOWN_ITERS=2

# LR
$env:MATRIX_LR="0.04"
$env:SCALAR_LR="0.04"
$env:TIED_EMBED_LR="0.05"
$env:GRAD_CLIP_NORM="0.3"

# Disable QAT and torch.compile for local (no Triton on Windows)
$env:QAT_ENABLED="0"
$env:USE_COMPILE="0"

# Data paths (HuggingFace cache)
$HF_SNAP="$env:USERPROFILE\.cache\huggingface\hub\datasets--willdepueoai--parameter-golf\snapshots\a85b0e6035c3c94bc23685a07c81a8f3bf89db80\datasets"
$env:DATA_PATH="$HF_SNAP\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH="$HF_SNAP\tokenizers\fineweb_1024_bpe.model"

# Small batch for local GPU
$env:TRAIN_BATCH_TOKENS=65536
$env:TRAIN_SEQ_LEN=1024

# Eval: skip heavy eval for smoke test (only 10 steps, BPB meaningless)
$env:VAL_LOSS_EVERY=-1
$env:VAL_BATCH_SIZE=8192
$env:TRAIN_LOG_EVERY=5
$env:SKIP_EVAL="1"

# Disable time limit
$env:MAX_WALLCLOCK_SECONDS=0

# EMA off for local test speed
$env:EMA_ENABLED="0"

# --- New features ---
$env:USE_LEAKY_RELU="1"
$env:MTP_ENABLED="0"
$env:STOCHASTIC_DEPTH_RATE="0.0"
$env:RECYCLE_MEM_TOKENS="0"
$env:MOE_NUM_EXPERTS="0"

# --- N-gram eval cache ---
$env:NGRAM_EVAL="1"
$env:NGRAM_ORDER="5"
$env:NGRAM_MIN_ORDER="2"
$env:NGRAM_BUCKETS="1048576"
$env:NGRAM_ALPHA="0.4"
$env:NGRAM_ENTROPY_ADAPTIVE="1"

# --- TTT disabled for local test ---
$env:TTT_ENABLED="0"

Write-Host "Local test: 10 steps + N-gram eval (order 5, 1M buckets)"
Write-Host "------------------------------------------------------------------"

& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_gpt.py
