# =============================================================================
# V29 - MC Dropout Ensemble: local test suite
# Run from parameter-golf/ directory
# ALL EVAL SKIPPED locally — only training + val_loss during training
# Full eval (sliding window, MC ensemble) only on cloud H100
# =============================================================================

$PYTHON = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
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
    # A: Baseline 1000-step training (no dropout)
    # -------------------------------------------------------------------------
    "A" {
        Write-Host "=== V29-A: Baseline training 1000 steps (SKIP_EVAL=1) ==="
        $env:RUN_ID         = "v29a_baseline"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:MC_DROP_P      = "0.0"
        & $PYTHON train_gpt.py
    }

    # -------------------------------------------------------------------------
    # D: Train WITH dropout (0.1) — compare val_loss vs A
    #    If val_loss is same or better than A, dropout doesn't hurt training
    #    MC ensemble eval tested on cloud only
    # -------------------------------------------------------------------------
    "D" {
        Write-Host "=== V29-D: Train with dropout=0.1 (SKIP_EVAL=1) ==="
        $env:RUN_ID         = "v29d_train_dropout"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:MC_DROP_P      = "0.1"
        & $PYTHON train_gpt.py
    }

    default {
        Write-Host "Usage: .\run_v29_tests.ps1 <A|D>"
        Write-Host ""
        Write-Host "  A - Baseline training 1000 steps (no dropout)"
        Write-Host "  D - Train with dropout=0.1 (compare val_loss vs A)"
        Write-Host ""
        Write-Host "  Both: SKIP_EVAL=1, ~35 min on 4060"
        Write-Host "  MC Ensemble eval - only on cloud H100"
    }
}
