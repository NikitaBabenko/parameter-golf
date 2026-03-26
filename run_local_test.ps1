$env:RUN_ID="fnet_hyena_stabilized_1000"
$env:ITERATIONS=30
$env:WARMUP_STEPS=5
# Остывание на последних 200 шагах
$env:WARMDOWN_ITERS=5

# Снимаем ручник: агрессивные LR для большого батча
$env:MATRIX_LR="0.0008"
$env:SCALAR_LR="0.0008"
$env:EMBED_LR="0.004"

# Зажимаем градиентные взрывы от Фурье
$env:GRAD_CLIP_NORM="0.5"

# Отключаем промежуточное QAT-квантование для скорости (0 - выкл, 1 - вкл)
$env:USE_QAT="0"

# Увеличиваем батч в 8 раз (65536 токенов)
$env:TRAIN_BATCH_TOKENS=65536

# Контекст оставляем большим
$env:TRAIN_SEQ_LEN=1024

# Защищаем Трансформер
$env:INT8_KEEP_FLOAT_FP32_NAME_PATTERNS="attn_blocks"

# Отключаем промежуточную валидацию для скорости
$env:VAL_LOSS_EVERY=-1 
# Безопасный размер батча для валидации в конце
$env:VAL_BATCH_SIZE=8192
$env:TRAIN_LOG_EVERY=10

# Отключаем турнирный лимит времени для тестов по шагам
$env:MAX_WALLCLOCK_SECONDS=0

# Параметры архитектуры и тестов
$env:NUM_HYENA=7
$env:NUM_ATTN=1
$env:USE_SMEAR_GATE="1"
$env:SKIP_EVAL="1"

Write-Host "🚀 Запускаем План пробития 1.2 (Большой батч, умный старт, 1000 шагов)..."
Write-Host "Логи пишутся в консоль и в logs/fnet_hyena_stabilized_1000.txt"
Write-Host "ВНИМАНИЕ: Из-за большого батча шаги будут долгими, но Loss будет падать ИДЕАЛЬНО ровно."
Write-Host "На шагах 800-1000 начнется 'остывание' для финального заныривания."
Write-Host "------------------------------------------------------------------"

& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" train_gpt.py