# =============================================================================
# V31 - KAN MLP (B-spline learnable activations): local test suite
# Run from parameter-golf/ directory
# SKIP_EVAL=1  -  only training + val_loss during training
# Compare train_loss @ 1000 steps against V29 baseline (2.6413)
# =============================================================================

$PYTHON = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$SCRIPT = "$env:USERPROFILE\Obsidian\LLM\py\V31\train_gpt.py"
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
    # A: Baseline 1000-step training (KAN off)  -  sanity check equal to V29-A
    # -------------------------------------------------------------------------
    "A" {
        Write-Host "=== V31-A: Baseline (USE_KAN=0) 1000 steps ==="
        $env:RUN_ID         = "v31a_baseline"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_KAN        = "0"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # B: KAN MLP with G=8, k=1 (fast piecewise-linear, O(B*T*F) memory)
    # NOTE: k=3 cubic B-spline is too slow without torch.compile (Windows).
    # k=1 is the local proxy: same learnable-per-neuron idea, gather-based,
    # no big tensor expansion.  Cloud test with k=3 if this shows promise.
    # -------------------------------------------------------------------------
    "B" {
        Write-Host "=== V31-B: KAN G=8 k=1 (linear, fast)  -  1000 steps ==="
        $env:RUN_ID         = "v31b_kan_g8_k1"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_KAN        = "1"
        $env:KAN_GRID_SIZE  = "8"
        $env:KAN_SPLINE_ORDER = "1"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # C: KAN MLP with G=4, k=1  -  fewer knots (smaller params)
    # -------------------------------------------------------------------------
    "C" {
        Write-Host "=== V31-C: KAN G=4 k=1 (linear, small)  -  1000 steps ==="
        $env:RUN_ID         = "v31c_kan_g4_k1"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_KAN        = "1"
        $env:KAN_GRID_SIZE  = "4"
        $env:KAN_SPLINE_ORDER = "1"
        & $PYTHON $SCRIPT
    }

    default {
        Write-Host "Usage: .\run_v31_tests.ps1 <A|B|C>"
        Write-Host ""
        Write-Host "  A - Baseline (USE_KAN=0), sanity check == V29-A"
        Write-Host "  B - KAN G=8 k=1 (piecewise-linear, fast, 9 coeffs/feature)"
        Write-Host "  C - KAN G=4 k=1 (smaller, 5 coeffs/feature)"
        Write-Host ""
        Write-Host "  All: SKIP_EVAL=1, 1000 steps, ~100 min on RTX 4060 (~6s/step)"
        Write-Host "  Reference: V29-A baseline train_loss=2.6413 @ step 1000"
    }
}
