# RVC Tools — Usage Guide

Helper scripts for training voice models and converting audio. All scripts are run from the **project root** (`d:\AI_Working_place\Retrieval-based-Voice-Conversion-WebUI`).

---

## 🚀 User Workflow (3 bước)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1 — Đọc file này (README.md)                          │
│           Hiểu tổng quan các tool + workflow phù hợp        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2 — Edit tools/rvc_config.ps1                         │
│           Set $T_* cho train | $C_* cho convert             │
│           (KHÔNG sửa rvc_pipeline.ps1)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3 — Chạy pipeline                                     │
│   .\tools\rvc_pipeline.ps1 -Mode train     ← Train model    │
│   .\tools\rvc_pipeline.ps1 -Mode convert   ← Convert audio  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Theo dõi log + verify output
              ─────────────────────────────
   logs/<MODEL>/epoch_log.txt   ← Train progress (per epoch)
   logs/<MODEL>/G_2333333.pth   ← Generator checkpoint
   assets/weights/<MODEL>.pth   ← Deployable model (52 MB)
   logs/<MODEL>/added_*.index   ← FAISS index
```

### Quy trình train + convert đầy đủ (end-to-end)

```
┌──────────────────────────────────────────────────────────────────┐
│  1. CHUẨN BỊ DATASET                                             │
│  ───────────────────────────────────────────────────────────────│
│  • Thu thập 10-30 phút audio sạch của 1 người                    │
│  • Hoặc tách từ bài hát: edit config cho convert mode (tách main)│
│  • Để file vào 1 folder (.wav hoặc .mp3)                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. TRAIN MODEL                                                  │
│  ───────────────────────────────────────────────────────────────│
│  Edit rvc_config.ps1:                                            │
│    $T_MODEL_NAME = "MyVoice"                                     │
│    $T_DATASET = "E:\Data_voice\my_clean_dataset"                 │
│    $T_EPOCHS = 150                                               │
│  Run: .\tools\rvc_pipeline.ps1 -Mode train                       │
│  Wait ~2-7h (auto cleanliness probe + size gate + 5 substeps)    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. CONVERT AUDIO                                                │
│  ───────────────────────────────────────────────────────────────│
│  Edit rvc_config.ps1:                                            │
│    $C_SOURCE = "E:\Data_voice\song.mp3"                          │
│    $C_MODEL_NAME = "MyVoice"                                     │
│    $C_OUTPUT = "E:\Data_voice\song_in_my_voice.wav"              │
│    $C_HAS_MUSIC = $true / $false                                 │
│    $C_BACKING_MODE = "keep" / "discard" / "convert-all"          │
│  Run: .\tools\rvc_pipeline.ps1 -Mode convert                     │
│  Wait ~2-5 phút (UVR5 + RVC + ffmpeg mix)                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  Output sẵn sàng nghe!
```

**Trường hợp thường gặp:**

| Tình huống | Action |
|---|---|
| Dataset có nhạc nền/noise | Probe sẽ BLOCK → clean trước (dùng convert-voice agent / `-Mode convert` để tách main vocal cho từng file dataset) |
| Dataset quá nhỏ (<5 phút) | Probe BLOCK → thu thêm audio, hoặc set `$T_FORCE_SMALL = $true` để override |
| Out of memory khi train | Giảm `$T_BATCH_SIZE` xuống 2, hoặc set `$T_VERSION = "v1"` (256-dim nhẹ hơn) |
| Training hỏng giữa chừng | Re-run cùng `$T_MODEL_NAME` → tự resume từ checkpoint cuối |
| Convert ra kết quả không tự nhiên | Đổi `$C_BACKING_MODE` (keep / discard) hoặc adjust `$C_F0UP_KEY` |
| Bài hát có giọng phụ rõ | Dùng `$C_BACKING_MODE = "keep"` để giữ harmony gốc |

---

## Files in this folder

### Unified pipeline (recommended)
| File | Purpose |
|---|---|
| `rvc_config.ps1` | Inputs & outputs. **Edit this** before running. |
| `rvc_pipeline.ps1` | Train + convert pipeline. Don't modify. |

### Original RVC tools
| File | Purpose |
|---|---|
| `app.py` | Standalone Gradio inference app |
| `infer_cli.py` | Single-file voice conversion (CLI) |
| `infer_batch_rvc.py` | Batch voice conversion over a folder |
| `calc_rvc_model_similarity.py` | Compare a model against reference models |
| `export_onnx.py` | Export `.pth` → `.onnx` for deployment |
| `onnx_inference_demo.py` | Run inference from an exported ONNX model |
| `download_models.py` | Auto-download pretrained HuBERT/RMVPE/UVR5 weights |
| `dlmodels.bat` / `dlmodels.sh` | Windows/Unix model downloader |
| `rvc_for_realtime.py` | Real-time conversion library (used by `gui_v1.py` and APIs) |

### Generic monitors
| File | Purpose |
|---|---|
| `hang_monitor.ps1` | Watchdog for hang-prone steps (train, UVR5, F0, HuBERT) |
| `watch_train_epochs.py` | Real-time per-epoch progress via TF events |

### Subfolder
| Folder | Purpose |
|---|---|
| `torchgate/` | Built-in noise gate (imported by inference modules) |

---

## Quick Start

### Train a new voice model

```powershell
# 1) Edit tools/rvc_config.ps1
#    Set $T_MODEL_NAME, $T_DATASET, $T_EPOCHS, etc.

# 2) Run training pipeline
.\tools\rvc_pipeline.ps1 -Mode train
```

Output:
- Model: `logs/<MODEL>/G_2333333.pth`
- Deployable weight: `assets/weights/<MODEL>.pth`
- FAISS index: `logs/<MODEL>/added_IVF*_v2.index`
- Per-epoch log: `logs/<MODEL>/epoch_log.txt`

### Convert audio with a trained model

```powershell
# 1) Edit tools/rvc_config.ps1
#    Set $C_SOURCE, $C_MODEL_NAME, $C_OUTPUT, $C_HAS_MUSIC, etc.

# 2) Run convert pipeline
.\tools\rvc_pipeline.ps1 -Mode convert
```

Output:
- Final mix: path you set in `$C_OUTPUT`
- Intermediates: `$C_WORK_DIR/stage1_hp3/`, `$C_WORK_DIR/stage2_hp5/`, etc.

---

## Config file reference (`rvc_config.ps1`)

### Shared environment variables (always required)
```powershell
$env:PYTHONPATH        = "D:\python_extra"        # torch CUDA lives here
$env:PYTHONIOENCODING  = "utf-8"                   # Vietnamese filenames
$env:USE_LIBUV         = "0"                       # Windows distributed-training fix
$env:weight_root       = "assets/weights"
$env:weight_uvr5_root  = "assets/uvr5_weights"
```

### Train mode variables (`$T_*`)

| Variable | Default | Notes |
|---|---|---|
| `$T_MODEL_NAME` | `MyModel` | Experiment name (no spaces) |
| `$T_DATASET` | — | Folder with `.wav`/`.mp3` files |
| `$T_SR` | `40k` | `40k` / `48k` / `32k` |
| `$T_VERSION` | `v2` | `v1` (256-dim, faster) / `v2` (768-dim, better) |
| `$T_EPOCHS` | `150` | Auto-bumped to 300 if dataset < 200 slices |
| `$T_BATCH_SIZE` | `4` | 2 for VRAM < 5GB; 8+ for ≥ 8GB |
| `$T_FP16` | `$false` | Only `$true` if Tensor Cores (RTX 20xx+) |
| `$T_F0_METHOD` | `rmvpe` | `rmvpe` / `harvest` / `crepe` / `pm` |
| `$T_GPU` | `0` | GPU id |
| `$T_SKIP_PROBE` | `$false` | Skip dataset cleanliness check |
| `$T_FORCE_DIRTY` | `$false` | Force-continue if probe says DIRTY |
| `$T_FORCE_SMALL` | `$false` | Force-continue if < 100 slices |

### Convert mode variables (`$C_*`)

| Variable | Default | Notes |
|---|---|---|
| `$C_SOURCE` | — | Source audio (`.wav`/`.mp3`) |
| `$C_MODEL_NAME` | — | Model name (without `.pth`) — reads `assets/weights/$_NAME.pth` |
| `$C_OUTPUT` | — | Final output file path |
| `$C_HAS_MUSIC` | `$true` | `$true` → UVR5 separation; `$false` → direct RVC |
| `$C_BACKING_MODE` | `keep` | `keep` / `discard` / `convert-all` (see below) |
| `$C_F0UP_KEY` | `0` | Pitch shift in semitones (`+12` = male→female) |
| `$C_INDEX_RATE` | `0.75` | `0.75` for music / `0.88` for speech (auto-adjusts) |
| `$C_F0_METHOD` | `rmvpe` | Pitch extraction method |
| `$C_WORK_DIR` | — | Folder for intermediate files |
| `$C_HP_STAGE1` | `HP3_all_vocals` | Stage 1 separator (vocal vs instrumental) |
| `$C_HP_STAGE2` | `HP5_only_main_vocal` | Stage 2 separator (main vs backing) |
| `$C_AGG` | `10` | UVR5 aggressiveness (5–15) |

### Backing vocals strategy

| Mode | Final mix | When to use |
|---|---|---|
| `keep` | RVC(main) + backing (original) + instrumental | Songs with harmonies — preserves chorus/backing |
| `discard` | RVC(main) + instrumental | Solo voice; clean output |
| `convert-all` | RVC(all vocals) + instrumental | Single-singer effect (may sound artificial) |

---

## Pipeline workflows

### Workflow 1 — Train from raw song
```powershell
# Dataset contains songs with music — clean first via convert mode + reuse main vocals
# OR set $T_FORCE_DIRTY = $true if you trust the probe to be a false positive

# Edit config: $T_MODEL_NAME = "MyVoice", $T_DATASET = "E:\Data_voice\my_raw_songs"
.\tools\rvc_pipeline.ps1 -Mode train
```

The cleanliness probe will BLOCK if music is detected. Either:
- Pre-clean dataset (use convert mode on each file → extract `main_vocal_*.wav`)
- Or set `$T_FORCE_DIRTY = $true` to override (model quality may suffer)

### Workflow 2 — Train from clean speech
```powershell
# Edit config: $T_DATASET = "E:\Data_voice\podcast_audio"
# Probe will pass (clean speech)
.\tools\rvc_pipeline.ps1 -Mode train
```

### Workflow 3 — Convert a song (preserve harmonies)
```powershell
# Edit config:
#   $C_SOURCE = "E:\Data_voice\song.mp3"
#   $C_MODEL_NAME = "MyVoice"
#   $C_OUTPUT = "E:\Data_voice\song_converted.wav"
#   $C_HAS_MUSIC = $true
#   $C_BACKING_MODE = "keep"
.\tools\rvc_pipeline.ps1 -Mode convert
```

### Workflow 4 — Convert a podcast/speech (no music)
```powershell
# Edit config:
#   $C_SOURCE = "E:\Data_voice\interview.mp3"
#   $C_HAS_MUSIC = $false
#   $C_INDEX_RATE auto-adjusts to 0.88
.\tools\rvc_pipeline.ps1 -Mode convert
```

---

## Per-epoch training log

When you run `-Mode train`, every epoch writes a line to `logs/<MODEL>/epoch_log.txt`:

```
[2026-05-30 17:30:15] Training started — 150 epochs, bs=4, fp16=False
[2026-05-30 17:31:00] Epoch 1/150  duration=0:00:42.156  ETA_remaining=01:45:30
[2026-05-30 17:31:42] Epoch 2/150  duration=0:00:42.084  ETA_remaining=01:43:48
...
[2026-05-30 17:35:18] CHECKPOINT saved at epoch 10
...
[2026-05-30 19:15:55] Training is done
```

Tail it in another terminal:
```powershell
Get-Content "logs\<MODEL>\epoch_log.txt" -Wait
```

---

## Other tool usage

### Direct CLI inference (single file)
```powershell
$env:PYTHONPATH = "D:\python_extra"
python tools/infer_cli.py `
  --f0up_key 0 `
  --input_path INPUT.wav `
  --index_path logs/<MODEL>/added_IVF*_v2.index `
  --opt_path OUTPUT.wav `
  --model_name <MODEL>.pth `
  --index_rate 0.88 `
  --f0method rmvpe
```

### Batch inference
```powershell
python tools/infer_batch_rvc.py
```
(Edit defaults in the script for input/output folders.)

### Hang monitor (watch any pipeline step)
```powershell
# Start your hang-prone step in background, capture PID:
$proc = Start-Process python -ArgumentList "infer/modules/train/train.py", "..." -PassThru
.\tools\hang_monitor.ps1 -StepType train -Name "MyModel" `
  -LogFile "logs/MyModel/train.log" -ProcessId $proc.Id -TimeoutMin 480

# Poll status file:
Get-Content "logs\_monitor_MyModel.status" -Tail 1
```

States: `RUNNING` / `DONE` / `HANG` / `ERROR` / `TIMEOUT`

### Watch epoch progress (in-flight training)
```powershell
$env:PYTHONPATH = "D:\python_extra"
python tools/watch_train_epochs.py logs/<MODEL>
# Emits one line per ~6 epochs (based on TF events flush interval)
```

### Model similarity check
```powershell
python tools/calc_rvc_model_similarity.py
# Edit ModelPath + reference dir at top of script
```

### Export to ONNX
```powershell
python tools/export_onnx.py
# Edit ModelPath + ExportedPath at top of script
```

### Download required pretrained models
```powershell
# First-time setup:
python tools/download_models.py
# OR:
.\tools\dlmodels.bat
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | PYTHONPATH not set | `$env:PYTHONPATH = "D:\python_extra"` |
| `CUDA=False` despite GPU present | torch CPU build | Reinstall torch cu124 into `D:\python_extra` |
| `UnpicklingError: GLOBAL fairseq.data.dictionary` | PyTorch 2.6 `weights_only=True` default | Edit `fairseq/checkpoint_utils.py` line 315: add `weights_only=False` |
| `use_libuv was requested but PyTorch was build without libuv` | Windows distributed training | `$env:USE_LIBUV = "0"` (already in config) |
| `cuDNN error: CUDNN_STATUS_NOT_SUPPORTED` in GRU | Tensor non-contiguous on certain shapes | Already fixed in `infer/lib/rmvpe.py` (cuDNN disabled for GRU) |
| `AttributeError: 'FigureCanvasAgg' has no attribute 'tostring_rgb'` | matplotlib upgrade | Already fixed in `infer/lib/train/utils.py` (uses `buffer_rgba`) |
| Training stuck at "Loading rmvpe model" | rmvpe.pt missing | `python tools/download_models.py` |
| UVR5 instrumental silent (rms = 0) | fp16 overflow on certain GPUs | UVR5 modules force `is_half=False` automatically |
| Cleanliness probe BLOCKs but I want to proceed | Source has music/noise the probe detected | Set `$T_FORCE_DIRTY = $true` OR pre-clean dataset |

---

## When to use the unified pipeline vs the agents

| Need | Use |
|---|---|
| One-shot run with known params | `rvc_pipeline.ps1` (faster, deterministic) |
| Interactive guidance + hang recovery | `train-voice` / `convert-voice` agents (Claude Code) |
| Automated CI / scheduled task | `rvc_pipeline.ps1` |
| Reproducible builds (config in git) | `rvc_pipeline.ps1` + commit `rvc_config.ps1` |
| Exploratory work, unfamiliar dataset | Agents (they auto-probe and guide you) |

---

## File output locations

```
logs/<MODEL_NAME>/
├── 0_gt_wavs/                      # original waveforms (preprocess output)
├── 1_16k_wavs/                     # 16kHz resampled slices
├── 2a_f0/                          # F0 (pitch) numpy files
├── 2b-f0nsf/                       # F0 for NSF vocoder
├── 3_feature768/                   # HuBERT features (v2)
├── 3_feature256/                   # HuBERT features (v1)
├── config.json                     # auto-generated training config
├── filelist.txt                    # filelist for train.py
├── G_2333333.pth                   # Generator (rolling, last save)
├── D_2333333.pth                   # Discriminator (rolling, last save)
├── added_IVF*_Flat_*.index         # FAISS index
├── train.log                       # train.py logger output
├── epoch_log.txt                   # rvc_pipeline.ps1 per-epoch log
└── eval/                           # TensorBoard event files

assets/weights/<MODEL_NAME>.pth     # deployable small weight (52 MB)
```
