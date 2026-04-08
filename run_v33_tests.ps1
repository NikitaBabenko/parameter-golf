# =============================================================================
# V33 - Neural ODE Depth (single shared block + Euler integration with FiLM):
# local test suite. Run from parameter-golf/ directory.
# SKIP_EVAL=1  -  only training + val_loss during training.
# Compare train_loss @ 1000 steps against V29 baseline (2.6413).
# =============================================================================

$PYTHON = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$SCRIPT = "$env:USERPROFILE\Obsidian\LLM\py\V33\train_gpt.py"
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

# Helper: clear ODE-shape overrides between tests so the next test re-applies defaults
function Reset-ModelShape {
    Remove-Item Env:MODEL_DIM      -ErrorAction SilentlyContinue
    Remove-Item Env:NUM_HEADS      -ErrorAction SilentlyContinue
    Remove-Item Env:NUM_KV_HEADS   -ErrorAction SilentlyContinue
    Remove-Item Env:MLP_MULT       -ErrorAction SilentlyContinue
    Remove-Item Env:NUM_LAYERS     -ErrorAction SilentlyContinue
    Remove-Item Env:ODE_TRAIN_STEPS -ErrorAction SilentlyContinue
    Remove-Item Env:ODE_EVAL_STEPS  -ErrorAction SilentlyContinue
}

$TEST = $args[0]

switch ($TEST) {

    # -------------------------------------------------------------------------
    # A: Baseline 1000-step training (USE_NEURAL_ODE=0)  -  sanity check == V29-A
    # -------------------------------------------------------------------------
    "A" {
        Write-Host "=== V33-A: Baseline (USE_NEURAL_ODE=0) 1000 steps ==="
        Reset-ModelShape
        $env:RUN_ID         = "v33a_baseline"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_NEURAL_ODE = "0"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # B: ODE d=1024, 16h/8kv, mlp=4x, 11 Euler steps  -  main experiment
    # -------------------------------------------------------------------------
    "B" {
        Write-Host "=== V33-B: ODE d=1024 mlp=4x 11 steps, 1000 iterations ==="
        Reset-ModelShape
        $env:RUN_ID         = "v33b_ode_d1024"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_NEURAL_ODE = "1"
        $env:ODE_TRAIN_STEPS = "11"
        $env:ODE_EVAL_STEPS  = "11"
        $env:MODEL_DIM      = "1024"
        $env:NUM_HEADS      = "16"
        $env:NUM_KV_HEADS   = "8"
        $env:MLP_MULT       = "4.0"
        $env:NUM_LAYERS     = "1"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # C: ODE d=768, 12h/6kv, mlp=3x, 11 steps  -  lighter version (4060-friendly)
    # -------------------------------------------------------------------------
    "C" {
        Write-Host "=== V33-C: ODE d=768 mlp=3x 11 steps, 1000 iterations ==="
        Reset-ModelShape
        $env:RUN_ID         = "v33c_ode_d768"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_NEURAL_ODE = "1"
        $env:ODE_TRAIN_STEPS = "11"
        $env:ODE_EVAL_STEPS  = "11"
        $env:MODEL_DIM      = "768"
        $env:NUM_HEADS      = "12"
        $env:NUM_KV_HEADS   = "6"
        $env:MLP_MULT       = "3.0"
        $env:NUM_LAYERS     = "1"
        & $PYTHON $SCRIPT
    }

    # -------------------------------------------------------------------------
    # D: ODE d=1024 with 22 eval steps (free quality test, only run if B works)
    # -------------------------------------------------------------------------
    "D" {
        Write-Host "=== V33-D: ODE d=1024 train=11 eval=22 steps ==="
        Reset-ModelShape
        $env:RUN_ID         = "v33d_ode_d1024_eval22"
        $env:ITERATIONS     = "1000"
        $env:WARMUP_STEPS   = "20"
        $env:WARMDOWN_ITERS = "100"
        $env:TRAIN_LOG_EVERY = "100"
        $env:VAL_LOSS_EVERY = "500"
        $env:EMA_ENABLED    = "0"
        $env:MTP_ENABLED    = "0"
        $env:USE_NEURAL_ODE = "1"
        $env:ODE_TRAIN_STEPS = "11"
        $env:ODE_EVAL_STEPS  = "22"
        $env:MODEL_DIM      = "1024"
        $env:NUM_HEADS      = "16"
        $env:NUM_KV_HEADS   = "8"
        $env:MLP_MULT       = "4.0"
        $env:NUM_LAYERS     = "1"
        & $PYTHON $SCRIPT
    }

    default {
        Write-Host "Usage: .\run_v33_tests.ps1 <A|B|C|D>"
        Write-Host ""
        Write-Host "  A - Baseline (USE_NEURAL_ODE=0), sanity check == V29-A"
        Write-Host "  B - ODE d=1024 16h/8kv mlp=4x, 11 Euler steps (main)"
        Write-Host "  C - ODE d=768  12h/6kv mlp=3x, 11 steps (4060-friendly)"
        Write-Host "  D - ODE d=1024, train=11 eval=22 steps (free-quality test)"
        Write-Host ""
        Write-Host "  All: SKIP_EVAL=1, 1000 steps, ~30-50 min on RTX 4060"
        Write-Host "  Reference: V29-A baseline train_loss=2.6413 @ step 1000"
        Write-Host "  NOTE: B may OOM on 8GB 4060  -  try C if so."
    }
}
