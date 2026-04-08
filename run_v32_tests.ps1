# =============================================================================
# V32 - Weight Hashing (HashedNet): local test suite
# Run from parameter-golf/ directory
# SKIP_EVAL=1  -  only training + val_loss during training
# Compare train_loss @ 1000 steps against V29 baseline (2.6413)
# =============================================================================

$PYTHON = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$SCRIPT = "$env:USERPROFILE\Obsidian\LLM\py\V32\train_gpt.py"
$HF_SNAP = "$env:USERPROFILE\.cache\huggingface\hub\datasets--willdepueoai--parameter-golf\snapshots\a85b0e6035c3c94bc23685a07c81a8f3bf89db80\datasets"
$env:DATA_PATH      = "$HF_SNAP\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH = "$HF_SNAP\tokenizers\fineweb_1024_bpe.model"

# Common: no compile, no QAT (Windows), no time limit, NO EVAL
$env:USE_COMPILE   = "0"
$env:QAT_ENABLED   = "0"
$env:MAX_WALLCLOCK_SECONDS = "0"
$env:SKIP_EVAL     = "1"

$env:TRAIN_BATCH_TOKENS = "65536"
$env:TRAIN_SEQ_LEN = "1024"

$TEST = $args[0]

switch ($TEST) {

    # -------------------------------------------------------------------------
    # A: Baseline 1000-step training (USE_WEIGHT_HASHING=0)  -  sanity == V29-A
    # -------------------------------------------------------------------------
    "A" {
        Write-Host "=== V32-A: Baseline (USE_WEIGHT_HASHING=0) 1000 steps ==="
        $env:RUN_ID         = "v32a_baseline"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_WEIGHT_HASHING = "0"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # B: HashedNet pool=5M (default, ~5.2x sharing)
    # -------------------------------------------------------------------------
    "B" {
        Write-Host "=== V32-B: Pool=5M, 1000 steps ==="
        $env:RUN_ID         = "v32b_hash_5m"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_WEIGHT_HASHING = "1"
        $env:HASH_POOL_SIZE = "5000000"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # C: HashedNet pool=3M (more aggressive sharing, ~8.6x)
    # -------------------------------------------------------------------------
    "C" {
        Write-Host "=== V32-C: Pool=3M, 1000 steps ==="
        $env:RUN_ID         = "v32c_hash_3m"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_WEIGHT_HASHING = "1"
        $env:HASH_POOL_SIZE = "3000000"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # D: HashedNet pool=8M (less sharing, ~3.2x)
    # -------------------------------------------------------------------------
    "D" {
        Write-Host "=== V32-D: Pool=8M, 1000 steps ==="
        $env:RUN_ID         = "v32d_hash_8m"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_WEIGHT_HASHING = "1"
        $env:HASH_POOL_SIZE = "8000000"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # E1: Deeper HashedNet (17L x 512d, pool=6M, low WD)
    #     Uses the ~6.5MB artifact headroom from V32-B by growing depth.
    #     Lower MUON_WD per PR #363 lesson: shared params suffer from
    #     outsized WD effect (pool entries used in many places).
    # -------------------------------------------------------------------------
    "E1" {
        Write-Host "=== V32-E1: Deeper HashedNet 17L, pool=6M, MUON_WD=0.01, 1000 steps ==="
        $env:RUN_ID         = "v32e1_hash_6m_17L"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_WEIGHT_HASHING = "1"
        $env:HASH_POOL_SIZE = "6000000"
        $env:NUM_LAYERS     = "17"
        $env:MODEL_DIM      = "512"
        $env:MUON_WD        = "0.01"
        & $PYTHON $SCRIPT
    }

    default {
        Write-Host "Usage: .\run_v32_tests.ps1 <A|B|C|D|E1>"
        Write-Host ""
        Write-Host "  A  - Baseline (USE_WEIGHT_HASHING=0), sanity check == V29-A"
        Write-Host "  B  - Pool=5M  (default, ~5.2x weight sharing) - DONE, train_loss=2.9185"
        Write-Host "  C  - Pool=3M  (aggressive sharing, ~8.6x)"
        Write-Host "  D  - Pool=8M  (light sharing, ~3.2x)"
        Write-Host "  E1 - 17L x 512d, pool=6M, MUON_WD=0.01 (uses B-headroom for depth)"
        Write-Host ""
        Write-Host "  A-D: 1000 steps, ~30-40 min on RTX 4060"
        Write-Host "  E1:  1000 steps, ~65-75 min on RTX 4060 (17/11 layers)"
        Write-Host "  Reference: V29-A baseline train_loss=2.6413 @ step 1000"
        Write-Host "  E1 success: train_loss <= 2.75 (half-way from V32-B to baseline)"
        Write-Host "  NOTE: Artifact size is INVARIANT to pool size - weights"
        Write-Host "        are reconstructed from pool at export time."
    }
}
