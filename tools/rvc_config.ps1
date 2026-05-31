# =============================================================
# RVC Pipeline — Config file (inputs / outputs only)
# Loaded by tools/rvc_pipeline.ps1 via dot-sourcing.
# Edit ONLY this file. Do not modify rvc_pipeline.ps1.
# =============================================================

# ── SHARED env (paths required by RVC modules) ───────────────
$env:PYTHONPATH        = "D:\python_extra"
$env:PYTHONIOENCODING  = "utf-8"
$env:USE_LIBUV         = "0"
$env:weight_root       = "assets/weights"
$env:weight_uvr5_root  = "assets/uvr5_weights"
$env:index_root        = "logs"
$env:outside_index_root = "assets/indices"
$env:rmvpe_root        = "assets/rmvpe"

# ═════════════════════════════════════════════════════════════
# TRAIN MODE  (used by:  .\tools\rvc_pipeline.ps1 -Mode train)
# ═════════════════════════════════════════════════════════════
$T_MODEL_NAME   = "MyModel"                    # experiment name (no spaces)
$T_DATASET      = "E:\Data_voice\my_dataset"   # folder with .wav/.mp3 files
$T_SR           = "40k"                        # 40k | 48k | 32k
$T_VERSION      = "v2"                         # v1 (256-dim) | v2 (768-dim)
$T_EPOCHS       = 150                          # total training epochs
$T_BATCH_SIZE   = 4                            # 2 if VRAM <5GB, 4 for 5-6GB, 8+ for ≥8GB
$T_FP16         = $false                       # $true only if Tensor Cores (RTX 20xx+)
$T_F0_METHOD    = "rmvpe"                      # rmvpe | harvest | crepe | pm
$T_GPU          = "0"                          # GPU id
$T_SKIP_PROBE   = $false                       # skip cleanliness probe
$T_FORCE_DIRTY  = $false                       # force-continue if probe says DIRTY
$T_FORCE_SMALL  = $false                       # force-continue if <100 slices

# ═════════════════════════════════════════════════════════════
# CONVERT MODE  (used by:  .\tools\rvc_pipeline.ps1 -Mode convert)
# ═════════════════════════════════════════════════════════════
$C_SOURCE       = "E:\Data_voice\song.mp3"     # source audio file
$C_MODEL_NAME   = "Obito"                      # model name (no .pth); reads assets/weights/$_NAME.pth
$C_OUTPUT       = "E:\Data_voice\output_final.wav"  # final output path
$C_HAS_MUSIC    = $true                        # $true → UVR5 separation; $false → direct RVC
$C_BACKING_MODE = "keep"                       # keep | discard | convert-all
$C_F0UP_KEY     = 0                            # semitones (0 same / +12 m→f / -12 f→m)
$C_INDEX_RATE   = 0.75                         # 0.75 music / 0.88 speech (auto if 0.75 + speech)
$C_F0_METHOD    = "rmvpe"
$C_WORK_DIR     = "E:\Data_voice\convert_tmp"  # intermediate files
$C_HP_STAGE1    = "HP3_all_vocals"             # vocal vs instrumental separator
$C_HP_STAGE2    = "HP5_only_main_vocal"        # main vs backing separator
$C_AGG          = 10                           # UVR5 aggressiveness (5-15)
