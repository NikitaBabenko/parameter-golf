"""
Quick parameter count & estimated compressed size calculator.
No training, no GPU needed. Just architecture math.
"""
import math

def estimate(name, vocab, d_model, n_layers, n_heads, n_kv_heads, mlp_mult,
             bigram_hash_size=0, bigram_emb_dim=0, trigram_hash_size=0, trigram_emb_dim=0,
             tied_embed=True, quant_bits=6, compression_ratio=0.58):
    """
    compression_ratio: typical ratio of zstd-22 on quantized weights.
    For int6 + zstd-22, empirically ~0.80-0.85 of raw quantized payload.
    For int8 + zlib-9, ~0.56-0.60.
    """
    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim
    mlp_hidden = int(d_model * mlp_mult)

    # Embedding
    embed_params = vocab * d_model

    # Per transformer layer:
    # Q, K, V projections (GQA: Q=d_model*d_model, K=d_model*kv_dim, V=d_model*kv_dim)
    qkv_params = d_model * d_model + 2 * d_model * kv_dim
    o_proj_params = d_model * d_model
    attn_params = qkv_params + o_proj_params

    # MLP: up + down (no bias typically)
    mlp_params = 2 * d_model * mlp_hidden

    # Norms: 2 per layer (RMSNorm = d_model each)
    norm_params = 2 * d_model

    layer_params = attn_params + mlp_params + norm_params
    total_layer_params = n_layers * layer_params

    # Final norm
    final_norm = d_model

    # Head (tied or separate)
    head_params = 0 if tied_embed else vocab * d_model

    # BigramHash
    bigram_params = 0
    if bigram_hash_size > 0:
        bigram_params = bigram_hash_size * bigram_emb_dim + bigram_emb_dim * d_model

    trigram_params = 0
    if trigram_hash_size > 0:
        trigram_params = trigram_hash_size * trigram_emb_dim + trigram_emb_dim * d_model

    total_params = embed_params + total_layer_params + final_norm + head_params + bigram_params + trigram_params

    # Size estimation
    # Small tensors (norms, etc.) stay in fp16 = 2 bytes each
    small_params = n_layers * norm_params + final_norm  # norms
    large_params = total_params - small_params

    # Int6 quantization stores values in int8 containers (1 byte each)
    # + fp16 per-row scales (1 scale per row of each 2D weight matrix)
    # Small params stored as fp16 (2 bytes each)

    # For int8 container storage:
    raw_payload_bytes = large_params * 1  # int8 container = 1 byte

    # Per-row scales: approximately #rows * 2 bytes (fp16)
    # Each 2D weight [out, in] has `out` scales
    # Rough estimate: total_2d_params / avg_row_width * 2
    avg_row_width = d_model  # rough avg
    n_scales = large_params / avg_row_width
    raw_payload_bytes += n_scales * 2  # fp16 scales

    raw_payload_bytes += small_params * 2  # fp16 for norms

    # torch.save overhead
    raw_payload_bytes += 30_000

    # zstd-22 compression on int6-in-int8 data: ~0.56-0.60 typical
    # The unused 2 bits per byte compress very well
    compressed_bytes = raw_payload_bytes * compression_ratio

    # Code size estimate
    code_bytes = 60_000  # ~60KB typical

    total_submission = compressed_bytes + code_bytes

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  d_model={d_model}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}")
    print(f"  MLP mult={mlp_mult} (hidden={mlp_hidden})")
    print(f"  Vocab={vocab}, tied_embed={tied_embed}")
    if bigram_hash_size:
        print(f"  BigramHash: {bigram_hash_size}x{bigram_emb_dim}")
    if trigram_hash_size:
        print(f"  TrigramHash: {trigram_hash_size}x{trigram_emb_dim}")
    print(f"  Quantization: int{quant_bits} + zstd-22")
    print(f"  ---")
    print(f"  Embedding:     {embed_params:>12,} params")
    print(f"  Per layer:     {layer_params:>12,} params")
    print(f"    Attention:   {attn_params:>12,}")
    print(f"    MLP:         {mlp_params:>12,}")
    print(f"    Norms:       {norm_params:>12,}")
    print(f"  All layers:    {total_layer_params:>12,} params ({n_layers} layers)")
    print(f"  BigramHash:    {bigram_params:>12,} params")
    print(f"  TrigramHash:   {trigram_params:>12,} params")
    print(f"  Final norm:    {final_norm:>12,} params")
    print(f"  Head:          {head_params:>12,} params")
    print(f"  ---")
    print(f"  TOTAL PARAMS:  {total_params:>12,}")
    print(f"  ---")
    print(f"  Raw payload:   {raw_payload_bytes/1e6:>10.2f} MB")
    print(f"  Compressed:    {compressed_bytes/1e6:>10.2f} MB")
    print(f"  + Code:        {code_bytes/1e6:>10.2f} MB")
    print(f"  TOTAL:         {total_submission/1e6:>10.2f} MB")

    ok = "FITS" if total_submission <= 16_000_000 else "TOO BIG"
    margin = 16_000_000 - total_submission
    print(f"  Status:        {ok} (margin: {margin/1e6:+.2f} MB)")
    print()
    return total_params, total_submission


print("=" * 60)
print("  PARAMETER GOLF SIZE ESTIMATOR")
print("=" * 60)

# --- Current SOTA reference configs ---

estimate("SOTA Reference: 11L int6 MLP3x (1.1271)",
    vocab=1024, d_model=512, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=4096, bigram_emb_dim=128,
    quant_bits=6)

estimate("SOTA Reference: 10L int5-MLP/int6-attn (1.1428)",
    vocab=1024, d_model=512, n_layers=10, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=10240, bigram_emb_dim=64,
    quant_bits=5.5)  # mixed ~5.5 avg

# --- Our candidate configs ---

estimate("Candidate A: 11L int6 MLP3x + BigramHash",
    vocab=1024, d_model=512, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=10240, bigram_emb_dim=64,
    quant_bits=6)

estimate("Candidate B: 11L int6 MLP3x + Bigram + Trigram",
    vocab=1024, d_model=512, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=10240, bigram_emb_dim=64,
    trigram_hash_size=10240, trigram_emb_dim=64,
    quant_bits=6)

estimate("Candidate C: 12L int6 MLP3x (no hash)",
    vocab=1024, d_model=512, n_layers=12, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0,
    quant_bits=6)

estimate("Candidate D: 11L int6 MLP3x (wider d=576)",
    vocab=1024, d_model=576, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0,
    quant_bits=6)

estimate("Candidate E: 10L int6 MLP4x",
    vocab=1024, d_model=512, n_layers=10, n_heads=8, n_kv_heads=4,
    mlp_mult=4.0,
    quant_bits=6)

estimate("Candidate F: 11L int8 MLP3x (baseline quant, zlib)",
    vocab=1024, d_model=512, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=10240, bigram_emb_dim=64,
    quant_bits=8, compression_ratio=0.56)  # int8+zlib typical

estimate("Candidate G: 13L int6 MLP2.5x",
    vocab=1024, d_model=512, n_layers=13, n_heads=8, n_kv_heads=4,
    mlp_mult=2.5,
    quant_bits=6)

estimate("Candidate H: 11L int6 MLP3x + BigramHash (large)",
    vocab=1024, d_model=512, n_layers=11, n_heads=8, n_kv_heads=4,
    mlp_mult=3.0, bigram_hash_size=20480, bigram_emb_dim=128,
    quant_bits=6)
