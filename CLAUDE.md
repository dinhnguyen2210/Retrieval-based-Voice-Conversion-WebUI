# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Retrieval-based Voice Conversion WebUI (RVC) is a VITS-based voice conversion framework supporting both a Gradio web interface and FastAPI REST API. It converts a source voice to a target voice using HuBERT feature extraction, FAISS-based retrieval, and a neural vocoder.

## User Use Cases — What You Can Do

### 1. Voice Cloning / Voice Conversion

| Goal | How | Output |
|---|---|---|
| Convert my voice to a target character/person | Train model from dataset → `convert-voice` agent | Audio in target voice |
| Make an AI cover of a song | 2-stage pipeline: HP3 → HP5 → RVC → mix with instrumental | Full cover track |
| Cross-gender voice change | `--f0up_key +12` (up an octave) or `-12` (down) | Gender-shifted voice |
| Blend two singers into one voice | WebUI "ckpt Processing" → Model A+B blend | Hybrid model |

### 2. Karaoke & Vocal Isolation

| Goal | How | Output |
|---|---|---|
| Make karaoke (remove vocals) | UVR5 with HP3 → take `instrument_*.wav` | Instrumental only |
| Extract vocals only (acapella) | UVR5 with HP5 → take `vocal_*.wav` | Acapella |
| Separate main vocal from backing vocals | 2-stage: HP3 → HP5 | `main_vocal` + `backing_vocals` |
| Replace only main vocal, keep harmonies | 2-stage + 3-stream mix (mode `keep`) | Main=new voice, backing=original |

### 3. Voice Model Training (Create Custom AI Voice)

| Goal | Dataset requirements | How |
|---|---|---|
| Train AI voice from 50–200 files (~10–30 min audio) | Clean `.wav`/`.mp3` of one speaker | `train-voice` agent / `tools/rvc_pipeline.ps1` |
| Train from a song (singing voice clone) | Vocal pre-separated with UVR5 | Same pipeline, `F0_METHOD=rmvpe` |
| Fine-tune existing model on new data | New dataset + existing checkpoint | Train with `-pg/-pd` pointing to old G/D |
| Score model similarity vs reference set | Reference models folder | `tools/calc_rvc_model_similarity.py` |

### 4. Audio Cleanup & Restoration

| Problem | Tool | Effect |
|---|---|---|
| Room reverb / wet vocals | UVR5 `VR-DeEchoNormal` | Remove mild echo |
| Heavy reverb (radio, hall) | UVR5 `VR-DeEchoAggressive` | Strong echo removal |
| Stereo reverb tail | UVR5 `onnx_dereverb_By_FoxJoy` (MDX-Net) | Best for stereo reverb |
| Background noise floor | `tools/torchgate/` (noise gate) | Threshold-based gating |
| **Cleanest production chain** | MDX-Net → DeEchoAggressive → HP3 → RVC | Combined cleanup workflow |

### 5. Real-Time Voice Changer

| Goal | Tool | Use case |
|---|---|---|
| Live mic voice change (gaming, Discord, OBS) | `gui_v1.py` desktop app | Real-time voice changer |
| Stream as a character voice | Real-time GUI → virtual audio cable output | Twitch / YouTube live |
| Integrate into a custom app | REST API `api_240604.py` | `/start`, `/stop`, `/config` endpoints |
| Build voice-chat AI | API + WebSocket integration | Custom apps |

### 6. Batch Processing & Production

| Goal | Tool |
|---|---|
| Convert 100+ audio files | `tools/infer_batch_rvc.py` or WebUI "Batch Inference" tab |
| Dub anime/film with custom voice | Batch convert raw dialogue clips |
| Automate full training pipeline | `tools/rvc_pipeline.ps1` + scheduled task |
| Deploy to production (mobile, edge, TensorRT) | `tools/export_onnx.py` → `.onnx` model |

### 7. Music Production Workflows

| Workflow | Steps |
|---|---|
| **Full AI cover song** | UVR5 separate → RVC convert vocal → ffmpeg mix |
| **Replace lead vocal in existing mix** | Isolate main vocal → RVC → 3-stream mix (keep backing) |
| **Mash-up of two singers** | Model A+B blend → convert |
| **Voice replacement in film/TV** | Batch convert each dialogue line |
| **AI voice acting** | Train character-voice model → batch convert TTS output |

---

## Full Feature Matrix

| # | Feature | Entry point | Where to use |
|---|---|---|---|
| 1 | **Single-file voice conversion** | WebUI "Model Inference" tab / `tools/infer_cli.py` | Convert one audio file to a target RVC voice |
| 2 | **Batch voice conversion** | WebUI "Batch Inference" tab / `tools/infer_batch_rvc.py` | Convert a whole folder of audio files at once |
| 3 | **Real-time voice conversion** | `gui_v1.py` (desktop) / `api_240604.py` REST | Live mic → speaker conversion with low latency |
| 4 | **Vocal/instrumental separation (UVR5)** | WebUI "Vocals/Accompaniment Separation" tab / `infer/modules/uvr5/modules.py` | Isolate vocals from background music |
| 5 | **De-reverb / de-echo** | UVR5 tab with `VR-DeEcho*` models | Clean reverb from recordings before training/inference |
| 6 | **Dataset preprocessing** | `infer/modules/train/preprocess.py` | Slice + normalize + resample audio to 16 kHz for training |
| 7 | **F0 (pitch) extraction** | `infer/modules/train/extract/extract_f0_print.py` (RMVPE/Harvest/Crepe/PM/Dio) | Extract pitch curves for training |
| 8 | **HuBERT feature extraction** | `infer/modules/train/extract_feature_print.py` | Extract 256-dim (v1) or 768-dim (v2) content features |
| 9 | **VITS model training** | WebUI "Training" tab / `infer/modules/train/train.py` / `tools/rvc_pipeline.ps1` | Train a custom voice model |
| 10 | **FAISS index building** | WebUI "Training" tab / inline Python in `tools/rvc_pipeline.ps1` | Build retrieval index from extracted features |
| 11 | **Model checkpoint editing** | WebUI "ckpt Processing" tab | Modify metadata, merge models, extract small weights |
| 12 | **Model merging (A+B blend)** | WebUI "ckpt Processing" tab | Mix two voice models with adjustable weight |
| 13 | **Model similarity comparison** | `tools/calc_rvc_model_similarity.py` | Compare a model against a reference set, get similarity score |
| 14 | **ONNX export** | WebUI "Export Onnx" tab / `tools/export_onnx.py` | Export model for TensorRT / mobile / DirectML deployment |
| 15 | **ONNX inference demo** | `tools/onnx_inference_demo.py` | Run inference from an exported ONNX model |
| 16 | **JIT compilation** | `infer/lib/jit/` (HuBERT, RMVPE, Synthesizer) | Speed up startup by JIT-compiling models |
| 17 | **TorchGate noise gate** | `tools/torchgate/` | Built-in noise gate for cleaner inference input |
| 18 | **Audio slicer** | `infer/lib/slicer2.py` | Slice long audio by silence detection (used by preprocess) |
| 19 | **F0 curve file override** | WebUI inference settings — `f0_file` param | Provide custom pitch curve to override auto F0 detection |
| 20 | **Multi-GPU training** | `train.py -g 0-1-2` | Train using multiple GPUs in parallel |
| 21 | **Multi-GPU feature extraction** | RMVPE GPU config `0-0-1` | Parallel F0 extraction across GPUs |
| 22 | **REST API (FastAPI)** | `api_240604.py` (modern) / `api_231006.py` (legacy) | Programmatic real-time conversion control |
| 23 | **Multi-language WebUI (i18n)** | `i18n/locale/*.json` (13 languages) | en_US, zh_CN/TW/HK/SG, ja_JP, ko_KR, es_ES, fr_FR, it_IT, pt_BR, ru_RU, tr_TR |
| 24 | **Pre-trained model auto-download** | `tools/download_models.py` / `tools/dlmodels.bat` | Fetch required HuBERT, RMVPE, UVR5 weights |
| 25 | **CLI training pipeline** | `tools/rvc_pipeline.ps1 -Mode train` (Windows PowerShell) | Run all 5 training steps with one command |
| 26 | **CLI convert pipeline** | `tools/rvc_pipeline.ps1 -Mode convert` (Windows PowerShell) | Run UVR5 → RVC → mix with one command |
| 27 | **Hang detector** | `tools/hang_monitor.ps1` | Generic watchdog for hang-prone steps (train, UVR5, F0, HuBERT) |
| 28 | **Epoch progress watcher** | `tools/watch_train_epochs.py` | Real-time per-epoch reporting via TF events |

### REST API endpoints (`api_240604.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/inputDevices` | GET | List available audio input devices |
| `/outputDevices` | GET | List available audio output devices |
| `/config` | POST | Configure model path, index, pitch, device, latency |
| `/start` | POST | Start real-time voice conversion |
| `/stop` | POST | Stop real-time voice conversion |

### WebUI Tabs (`infer-web.py`)

| Tab | Purpose |
|---|---|
| **Model Inference** → Single Inference | Convert one audio file |
| **Model Inference** → Batch Inference | Convert a folder of files |
| **Vocals/Accompaniment Separation** | UVR5 vocal isolation, de-reverb, de-echo |
| **Training** | Full training pipeline (preprocess → F0 → features → train → index) |
| **ckpt Processing** | Modify model metadata, merge models, extract small weights, view info |
| **Export Onnx** | Convert .pth to .onnx for deployment |
| **FAQ** | Common questions and answers |

### F0 (Pitch) Extraction Methods

| Method | Module | Speed | Quality | Notes |
|---|---|---|---|---|
| `rmvpe` (default) | `infer/lib/rmvpe.py` | Fast on GPU | **Best** | Robust against noise, recommended |
| `rmvpe+` | `extract_f0_rmvpe_dml.py` | Fast | Best | DirectML variant for AMD/Intel GPUs |
| `harvest` | `infer/lib/infer_pack/modules/F0Predictor/HarvestF0Predictor.py` | Medium | High | Reliable for clean singing |
| `crepe` | torchcrepe | Slow | High | Deep-learning based |
| `pm` | `PMF0Predictor.py` | Very fast | Medium | Praat method, fallback |
| `dio` | `DioF0Predictor.py` | Fast | Medium | World vocoder method |

### UVR5 Models (`assets/uvr5_weights/`)

| Model | Category | Use case |
|---|---|---|
| `HP2_all_vocals` | Preserve all vocals | Audio without backing harmonies |
| `HP3_all_vocals` | Preserve all vocals | Stage 1 of 2-stage main vocal isolation (lowest corr in benchmarks) |
| `HP5_only_main_vocal` | Main vocal only | Audio with backing vocals — Stage 2 of 2-stage isolation |
| `HP2-人声...` (Chinese name) | Preserve all vocals | Original HP2 release |
| `HP5-主旋律人声...` | Main vocal only | Original HP5 release |
| `VR-DeEchoNormal` | De-echo | Remove delay/echo, normal strength |
| `VR-DeEchoAggressive` | De-echo | Remove delay/echo, stronger removal |
| `VR-DeEchoDeReverb` | De-echo + de-reverb | Combined (slowest, ~2x slower than DeEcho) |
| `onnx_dereverb_By_FoxJoy` | MDX-Net dereverb | Best for stereo reverb only (not mono) |

**Recommended cleanest chain:** MDX-Net dereverb → DeEcho-Aggressive → HP3 → HP5 → RVC

### CLI Inference Parameters (`tools/infer_cli.py`)

| Flag | Purpose | Typical value |
|---|---|---|
| `--f0up_key` | Pitch shift in semitones | `0` (same gender) / `±12` (cross gender) |
| `--input_path` | Source audio | path/to/file.wav |
| `--index_path` | FAISS index | logs/model/added_*.index |
| `--opt_path` | Output file | path/to/output.wav |
| `--model_name` | Model in `assets/weights/` | model.pth |
| `--index_rate` | Retrieval blend ratio | `0.75` (music) / `0.88` (speech) |
| `--f0method` | Pitch extraction | `rmvpe` (default best) |
| `--device` | cuda:0 / cpu | auto-detected |
| `--is_half` | FP16 inference | True if VRAM >= 6 GB |
| `--filter_radius` | Median filter on F0 | `3` |
| `--resample_sr` | Output resample rate | `0` = keep |
| `--rms_mix_rate` | Volume envelope mix | `0.25` |
| `--protect` | Protect voiceless consonants | `0.33` |

## Running the Application

```bash
# WebUI (primary interface)
python infer-web.py

# WebUI with port and host options
python infer-web.py --port 7865 --colab

# Command-line inference (no GUI needed)
python tools/infer_cli.py --f0up_key 0 --input_path audio.wav --index_path logs/model/model.index --opt_path output.wav --model_name model.pth --index_rate 0.75

# Batch inference
python tools/infer_batch_rvc.py

# REST API (modern, FastAPI-based)
python api_240604.py
```

**Windows launchers:** `go-web.bat` (NVIDIA), `go-web-dml.bat` (AMD/Intel DirectML)

## Environment Setup

```bash
# Install with pip (NVIDIA CUDA)
pip install -r requirements.txt

# AMD/Intel on Windows (DirectML)
pip install -r requirements-dml.txt

# AMD on Linux (ROCM)
pip install -r requirements-amd.txt

# Intel on Linux (IPEX)
pip install -r requirements-ipex.txt

# Python 3.11+
pip install -r requirements-py311.txt

# Download required model weights
python tools/download_models.py
# or on Windows: tools/dlmodels.bat
```

**Required assets** (must be downloaded separately):
- `assets/hubert/hubert_base.pt`
- `assets/pretrained/` and `assets/pretrained_v2/` — base model weights
- `assets/rmvpe/rmvpe.pt` — pitch extraction model
- `assets/uvr5_weights/` — vocal separation models
- `ffmpeg.exe` / `ffprobe.exe` in root (Windows) or system PATH

**Environment variables** (`.env` at root):
```
weight_root=assets/weights
weight_uvr5_root=assets/uvr5_weights
index_root=logs
outside_index_root=assets/indices
rmvpe_root=assets/rmvpe
```

## Testing

```bash
# Run unit tests
python -m pytest

# GitHub CI runs tests via .github/workflows/unitest.yml
```

## Architecture

### Voice Conversion Pipeline

The full inference flow: `VC.vc_single()` → pipeline → output audio

1. **Audio input** → resample to 16 kHz, normalize
2. **HuBERT embedding** — `fairseq` model extracts content features (`infer/lib/infer_pack/commons.py`)
3. **F0 extraction** — pitch extracted via RMVPE (default), Crepe, Harvest, or PyWorld (`infer/lib/rmvpe.py`, `infer/lib/infer_pack/F0Predictor/`)
4. **FAISS retrieval** — nearest neighbors from training set features blend with input features (`infer/modules/vc/pipeline.py`)
5. **VITS synthesis** — `SynthesizerTrnMs256NSFsid` (v1) or `SynthesizerTrnMs768NSFsid` (v2) generates waveform
6. **Post-processing** — resample to target sample rate

### Key Modules

| Path | Purpose |
|---|---|
| `infer-web.py` | Main Gradio WebUI entry point |
| `configs/config.py` | Singleton `Config` — device detection, model config loading |
| `infer/modules/vc/modules.py` | `VC` class — primary inference interface |
| `infer/modules/vc/pipeline.py` | Audio processing pipeline |
| `infer/lib/infer_pack/models.py` | VITS synthesizer model definitions |
| `infer/lib/infer_pack/F0Predictor/` | F0 extraction method implementations |
| `infer/modules/train/train.py` | Training loop |
| `infer/modules/train/preprocess.py` | Dataset preprocessing |
| `infer/lib/train/data_utils.py` | Dataset class and loader |
| `infer/modules/uvr5/modules.py` | Vocal separation (UVR5) |
| `infer/lib/rtrvc.py` | Real-time voice conversion |
| `api_240604.py` | FastAPI REST interface |
| `tools/infer_cli.py` | CLI inference wrapper |

### Model Versions

- **v1**: `SynthesizerTrnMs256NSFsid` — 256 hidden channels, configs in `configs/v1/`
- **v2**: `SynthesizerTrnMs768NSFsid` — 768 hidden channels, higher quality, configs in `configs/v2/`
- Each supports 32k, 40k, and 48k sample rates; active config symlinked/copied to `configs/inuse/`

### Configuration System

`configs/config.py` implements a singleton `Config` that:
- Detects hardware: CUDA → Intel XPU → CPU fallback
- Loads the JSON model config from `configs/inuse/`
- Parses CLI args: `--port`, `--pycmd`, `--colab`, `--dml`
- Exposes `x_pad`, `x_query`, `x_center`, `x_max` padding parameters used throughout inference

### Training Pipeline

Run from the WebUI **Training** tab, or fully via CLI using `tools/rvc_pipeline.ps1` (Windows/PowerShell).

**One-command CLI pipeline:**
```powershell
# 1) Edit tools/rvc_config.ps1 (set $T_MODEL_NAME, $T_DATASET, $T_EPOCHS, ...)
# 2) Run:
.\tools\rvc_pipeline.ps1 -Mode train
```

**Manual step-by-step (PowerShell):**
```powershell
$MODEL = "MyModel"; $DATASET = "E:\Data\dataset"; $SR_HZ = 40000; $EXP = "logs/$MODEL"

# 1. Preprocess — slice & resample audio to 16kHz
python infer/modules/train/preprocess.py "$DATASET" $SR_HZ 4 "$EXP" False 3.0

# 2. Extract F0 — pitch with RMVPE (6 workers)
python infer/modules/train/extract/extract_f0_print.py "$EXP" 6 rmvpe

# 3. Extract HuBERT features (v2 = 768-dim)
python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 "$EXP" v2 False

# 4. Train VITS model
$env:USE_LIBUV = "0"   # required on Windows
python infer/modules/train/train.py -e "$MODEL" -sr 40k -f0 1 -bs 4 -g 0 -te 100 -se 10 -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth -l 1 -c 0 -sw 0 -v v2

# 5. Build FAISS index
python -c "
import os,numpy as np,faiss
d='$EXP/3_feature768'
big=np.concatenate([np.load(f'{d}/{f}') for f in sorted(os.listdir(d))])
n=min(int(16*np.sqrt(len(big))),len(big)//39)
idx=faiss.index_factory(768,f'IVF{n},Flat'); faiss.extract_index_ivf(idx).nprobe=1
idx.train(big); idx.add(big)
faiss.write_index(idx,f'{\"$EXP\"}/added_IVF{n}_Flat_nprobe_1_{\"$MODEL\"}_v2.index')
"
```

**Script parameters** (in `tools/rvc_config.ps1`, train mode uses `$T_*` prefix):

| Variable | Default | Description |
|---|---|---|
| `$T_MODEL_NAME` | `MyModel` | Experiment name, no spaces |
| `$T_DATASET` | — | Folder with `.wav`/`.mp3` source audio |
| `$T_SR` | `40k` | Sample rate: `40k`, `48k`, `32k` |
| `$T_VERSION` | `v2` | Model version: `v1` or `v2` |
| `$T_EPOCHS` | `150` | Total training epochs (auto-bumped to 300 if < 200 slices) |
| `$T_BATCH_SIZE` | `4` | Reduce to `2` if VRAM < 6 GB |
| `$T_F0_METHOD` | `rmvpe` | Pitch method: `rmvpe`, `harvest`, `crepe`, `pm` |

Training outputs go to `logs/<experiment_name>/`:
- `G_2333333.pth` / `D_2333333.pth` — latest checkpoint (overwritten each save with `-l 1`)
- `added_IVF*_v2.index` — FAISS index for inference
- `3_feature768/` — HuBERT features (768-dim for v2)
- `2a_f0/`, `2b-f0nsf/` — F0 pitch files

### Hardware Support

| Platform | Requirements file | Backend |
|---|---|---|
| NVIDIA GPU | `requirements.txt` | CUDA via PyTorch |
| AMD/Intel (Windows) | `requirements-dml.txt` | DirectML |
| AMD (Linux) | `requirements-amd.txt` | ROCM |
| Intel (Linux) | `requirements-ipex.txt` | IPEX |
| CPU only | any | Torch CPU fallback |

Intel GPU path uses `infer/modules/ipex/` patching loaded at startup.

### Vocal Separation + Conversion Pipeline (UVR5 → RVC → FFmpeg)

**Use the `convert-voice` agent** for one-command runs. The pipeline below is the manual recipe.

#### 2-Stage main vocal isolation (preserves backing vocals)

When the source has backing vocals/harmonies, use 2-stage separation. This isolates only the main vocal for RVC conversion while keeping backing vocals untouched in the final mix.

```
Source song.mp3
  └─[HP3]──► all_vocals.wav + instrumental.wav        ← Stage 1
              └─[HP5]──► main_vocal.wav + backing_vocals.wav  ← Stage 2
                          └─[RVC]──► main_converted.wav
                                      │
            instrumental.wav ─────────┤
            backing_vocals.wav ───────┴──[ffmpeg amix]──► FINAL.wav
```

```powershell
$env:PYTHONPATH = "D:\python_extra"        # torch CUDA lives here — see Environment recovery
$env:weight_uvr5_root = "assets/uvr5_weights"
$INPUT  = "E:\Data_voice\song.mp3"
$OUT    = "E:\Data_voice\out"
$MODEL  = "MyModel"
$INDEX  = "logs/$MODEL/added_IVF*_v2.index"

# Stage 1 — HP3 (lowest corr in benchmarks): vocals vs instrumental
python -c "
import sys,os; sys.path.insert(0,'.')
from infer.modules.uvr5.modules import uvr
os.makedirs(r'$OUT/stage1', exist_ok=True)
list(uvr('HP3_all_vocals','',r'$OUT/stage1',[type('F',(),{'name':r'$INPUT'})()],r'$OUT/stage1',10,'wav'))
"

# Stage 2 — HP5 isolates main from backing vocals (run on Stage 1 vocal output)
python -c "
import sys,os; sys.path.insert(0,'.')
from infer.modules.uvr5.modules import uvr
os.makedirs(r'$OUT/stage2', exist_ok=True)
src = [f for f in os.listdir(r'$OUT/stage1') if f.startswith('vocal')][0]
list(uvr('HP5_only_main_vocal','',r'$OUT/stage2',[type('F',(),{'name':os.path.join(r'$OUT/stage1',src)})()],r'$OUT/stage2',10,'wav'))
"

# Stage 3 — RVC convert ONLY the main vocal
$MAIN = (Get-ChildItem "$OUT/stage2" -Filter "vocal_*").FullName
python tools/infer_cli.py --f0up_key 0 --input_path "$MAIN" --index_path "$INDEX" --opt_path "$OUT/main_converted.wav" --model_name "$MODEL.pth" --index_rate 0.75 --f0method rmvpe

# Stage 4 — 3-stream mix: converted main + backing (preserved) + instrumental
$BACKING = (Get-ChildItem "$OUT/stage2" -Filter "instrument_*").FullName
$INST    = (Get-ChildItem "$OUT/stage1" -Filter "instrument_*").FullName
ffmpeg -y -i "$OUT/main_converted.wav" -i "$BACKING" -i "$INST" `
  -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.7[a1];[2:a]volume=1.0[a2];[a0][a1][a2]amix=inputs=3:duration=longest:normalize=0" `
  "E:\Data_voice\song_converted_final.wav"
```

**Backing vocals strategy:**
| Mode | Final mix | Use case |
|---|---|---|
| `keep` (default) | main_RVC + backing (original) + instrumental | Songs with harmonies — preserves chorus/backing voices |
| `discard` | main_RVC + instrumental only | Solo voice, clean output |
| `convert-all` | RVC(main + backing) + instrumental | Single-singer effect (may sound artificial) |

**UVR5 model benchmarks** (lonely.mp3 test):

| Model | corr (lower=better) | Best for |
|---|---|---|
| **HP3_all_vocals** | **0.074** | Stage 1 — vocals vs instrumental (winner) |
| HP5_only_main_vocal | 0.115 | Stage 2 — main vs backing vocals |
| HP2_all_vocals | 0.118 | Alternative Stage 1 if HP3 underperforms |
| VR-DeEchoNormal | n/a | Optional pre-RVC reverb cleanup if `corr > 0.15` |
| VR-DeEchoAggressive | n/a | Heavy reverb removal (radio recordings) |

### Environment recovery (PYTHONPATH=D:\python_extra)

`torch 2.6.0+cu124` lives in `D:\python_extra` (not the default `site-packages`) because the system drive C: ran out of space. The user environment variable `PYTHONPATH` is set permanently to that path. If RVC fails with `ModuleNotFoundError: torch` or `CUDA=False`, restore with:
```powershell
[Environment]::SetEnvironmentVariable("PYTHONPATH", "D:\python_extra", "User")
$env:PYTHONPATH = "D:\python_extra"
```

**DO NOT install `audio-separator`, `demucs`, or `mdx-net` directly** — they upgrade torch to a CPU-only version and break numpy/librosa. If MDX-Net is needed, create a separate venv:
```powershell
python -m venv D:\rvc_venv_mdx
D:\rvc_venv_mdx\Scripts\Activate.ps1
pip install audio-separator
```

### Hang detection (`tools/hang_monitor.ps1`)

Generic background watchdog for hang-prone steps. Writes one line per check to `logs/_monitor_<Name>.status`; last line state ∈ `{RUNNING, DONE, HANG, ERROR, TIMEOUT}`.

| Step | Stall window (min) | Watches | Completion signal |
|---|---|---|---|
| `train` | 5 | log line count + `nvidia-smi` GPU util | log contains `Training is done` / `saving final ckpt` |
| `uvr5` | 3 | log line count | output file > 1 KB |
| `hubert` | 2 | file count in `3_feature768/` | output file > 1 KB |
| `preprocess` | 2 | file count in `1_16k_wavs/` | output file > 1 KB |
| `f0` | 2 | file count in `2a_f0/` | output file > 1 KB |
| `rvc-infer` | 3 | log line count | output `.wav` > 1 KB |

Usage in agents: launch the python step in background → capture its PID → launch `hang_monitor.ps1` with matching `-StepType -Name -ProcessId` and a target (`-LogFile` / `-OutputDir` / `-OutputFile`). Poll the `.status` file periodically; on `HANG` kill the PID, apply the documented fix for that step, retry. Both `train-voice` and `convert-voice` agents wire this in automatically — see their respective "Hang monitoring" sections.

**Train-step ETA + progress reports**: when `-StepType train`, also pass `-TotalEpochs <N>` and optionally `-ReportEveryEpochs <N>` (default 5). The monitor parses `====> Epoch: N ... (H:MM:SS.fff)` lines, computes sliding-window avg epoch time from the last 5 epochs, and writes a progress summary to `logs/_monitor_<Name>.progress` every N epochs. Each line: `[HH:MM:SS] epoch X/Y (Z%) avg=Ws/epoch elapsed=... eta=...`. Agents should tail this file every 5 minutes and surface it to the user.

### Known Bugs & Fixes (PyTorch 2.6+ / Windows)

| Bug | Location | Fix |
|---|---|---|
| `UnpicklingError: GLOBAL fairseq.data.dictionary.Dictionary` | `fairseq/checkpoint_utils.py:315` | Add `weights_only=False` to `torch.load()` |
| `RuntimeError: use_libuv was requested but PyTorch was build without libuv support` | `infer/modules/train/train.py` | Set `os.environ["USE_LIBUV"] = "0"` before `mp.Process` spawn |
| `AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` | `infer/lib/train/utils.py` | Replace `np.fromstring(fig.canvas.tostring_rgb(),...)` with `np.asarray(fig.canvas.buffer_rgba(),...)[..., :3]` |
| `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED` in GRU | `infer/lib/rmvpe.py:174` | Wrap GRU call: `with torch.backends.cudnn.flags(enabled=False): return self.gru(x.contiguous())[0]` |

### Unified CLI Pipeline (`tools/rvc_pipeline.ps1`)

A single-script alternative to the agents. Combines training and conversion into one entry point with config externalized to a separate file.

> **For end-user workflow & detailed examples, see `tools/README.md`** — it has the full 3-step user guide, end-to-end train+convert flow, common scenarios, and troubleshooting.

**Files:**
- `tools/README.md` — user-facing usage guide (start here)
- `tools/rvc_config.ps1` — inputs/outputs only. Edit this.
- `tools/rvc_pipeline.ps1` — logic. Do not modify.

**Usage:**
```powershell
# 1) Edit tools/rvc_config.ps1 to set your inputs (model name, paths, etc.)

# 2) Train a new voice model
.\tools\rvc_pipeline.ps1 -Mode train

# 3) Convert audio with a trained model
.\tools\rvc_pipeline.ps1 -Mode convert
```

**Config file layout** (`tools/rvc_config.ps1`):

| Section | Variables | Used by mode |
|---|---|---|
| SHARED env | `$env:PYTHONPATH`, `$env:USE_LIBUV`, etc. | both |
| `$T_*` (train) | `$T_MODEL_NAME`, `$T_DATASET`, `$T_EPOCHS`, `$T_BATCH_SIZE`, `$T_VERSION`, `$T_FP16`, `$T_F0_METHOD`, `$T_SKIP_PROBE`, `$T_FORCE_DIRTY`, `$T_FORCE_SMALL` | `-Mode train` |
| `$C_*` (convert) | `$C_SOURCE`, `$C_MODEL_NAME`, `$C_OUTPUT`, `$C_HAS_MUSIC`, `$C_BACKING_MODE`, `$C_F0UP_KEY`, `$C_INDEX_RATE`, `$C_WORK_DIR`, `$C_HP_STAGE1`, `$C_HP_STAGE2` | `-Mode convert` |

**Train mode tasks** (14 steps, all automatic):
1. Pre-flight (GPU/CPU detection)
2. Cleanliness probe (perc/low/flat metrics — blocks if 2/3 dirty signals)
3. Preprocess (slice audio to 3s chunks @ 16kHz)
4. Dataset size gate (blocks if < 100 slices)
5. F0 extraction (rmvpe / harvest / crepe)
6. HuBERT feature extraction (256-dim v1 / 768-dim v2)
7. Generate config.json + filelist.txt
8. **Train VITS with per-epoch logging** → see below
9. Build FAISS index
10. Final report

**Per-epoch training log** (auto-generated): `logs/<MODEL>/epoch_log.txt`
```
[2026-05-30 17:30:15] Training started — 150 epochs, bs=4, fp16=False
[2026-05-30 17:31:00] Epoch 1/150  duration=0:00:42.156  ETA_remaining=01:45:30
[2026-05-30 17:31:42] Epoch 2/150  duration=0:00:42.084  ETA_remaining=01:43:48
[2026-05-30 17:35:18] CHECKPOINT saved at epoch 10
...
[2026-05-30 19:15:55] Training is done
```

**Convert mode tasks** (12 steps):
1. Pre-flight (verify model + index exist)
2. Branch: `$C_HAS_MUSIC=$false` → skip UVR5 → direct RVC → done
3. UVR5 Stage 1 (HP3) — vocal vs instrumental + QC
4. UVR5 Stage 2 (HP5) — main vs backing + QC (skipped if `convert-all`)
5. Apply backing strategy (keep / discard / convert-all)
6. RVC convert + QC (rms_ratio, file size)
7. FFmpeg mix (3-stream for `keep`, 2-stream otherwise)
8. Final QC (duration, rms, size)
9. Final report

**When to use scripts vs agents:**

| Need | Use |
|---|---|
| One-shot run with known params | `rvc_pipeline.ps1` (faster, deterministic) |
| Interactive guidance + hang recovery | `train-voice` / `convert-voice` agents |
| Automated CI / scheduled task | `rvc_pipeline.ps1` |
| Reproducible builds (config in git) | `rvc_pipeline.ps1` (commit `rvc_config.ps1`) |

### Project Agents (`.claude/agents/`)

Seven sub-agents are defined for common audio workflows. Invoke via the Agent tool or by asking Claude Code to use them.

| Agent | Job | Maps to User Use Case |
|---|---|---|
| `train-voice` | Train new RVC voice model from clean dataset | Section 3 (Training) |
| `convert-voice` | Convert audio to target voice (single file, with optional UVR5 + mix) | Sections 1, 7 (Voice cloning, Music production) |
| `isolate-vocals` | Separate stems with UVR5 only — karaoke / acapella / main-backing split | Section 2 (Karaoke & vocal isolation) |
| `clean-audio` | Multi-stage cleanup chain (DeEcho, DeReverb, noise gate) | Section 4 (Audio cleanup) |
| `batch-convert` | Loop RVC over a folder of audio files | Section 6 (Batch processing) |
| `prepare-dataset` | Raw audio (songs, podcasts) → clean training-ready dataset folder | Section 3 prep stage |
| `merge-models` | Blend 2 RVC `.pth` models with alpha weighting | Section 7 (Model A+B blend) |

**Workflow chains** (compose agents for complex pipelines):
- Raw song → trained model: `prepare-dataset` → `train-voice`
- Trained model → produced cover: `convert-voice` (handles UVR5 + mix internally)
- Many dialogue lines, one voice: `batch-convert`
- Hybrid singer: `merge-models` → `convert-voice` with merged `.pth`

#### `train-voice` — Train a new RVC voice model
Scope: project | File: `.claude/agents/train-voice.md`

Workflow:
1. Asks user for **dataset path** and **model name**
2. Auto-detects GPU VRAM → sets batch size (2/4/8/12) and FP16
3. Auto-detects dataset size → sets epoch count (100–300)
4. Runs full 5-step pipeline: preprocess → F0 → HuBERT features → train → FAISS index
5. Applies known bug fixes automatically (libuv, fairseq, tostring_rgb, cuDNN GRU)
6. Reports final model + index paths and ready-to-use inference command

**Parameter auto-selection:**
- VRAM ≥ 10 GB → batch 12, FP16 on
- VRAM 6–10 GB → batch 8, FP16 on
- VRAM 4–6 GB → batch 4, FP16 off
- VRAM < 4 GB → batch 2, FP16 off
- Dataset < 50 files → 300 epochs; 50–200 → 200; 200–500 → 150; > 500 → 100

#### `convert-voice` — Convert voice in audio file
Scope: project | File: `.claude/agents/convert-voice.md`

Workflow:
1. Asks user for **source audio** and **target model**
2. Asks if source has background music (yes → UVR5 separation first)
3. Auto-selects pitch shift (f0up_key) and index_rate based on voice type
4. Runs: UVR5 → RVC infer_cli → FFmpeg amix (or RVC only for speech)
5. Verifies output file and reports results

**UVR5 model used:** `HP5_only_main_vocal` (main vocal isolation)
**Default params:** `f0method=rmvpe`, `index_rate=0.75` (music) / `0.88` (speech), `f0up_key=0`

#### `isolate-vocals` — Extract stems with UVR5 (no RVC)
Scope: project | File: `.claude/agents/isolate-vocals.md`

Modes: `karaoke` (instrumental only) / `acapella` (vocal only) / `main-backing-split` (3 stems). Stage 1 uses HP3, Stage 2 uses HP5. Has fallback chain to HP2 when QC fails.

#### `clean-audio` — De-reverb, de-echo, noise gate
Scope: project | File: `.claude/agents/clean-audio.md`

Cleanup levels: `mild` (DeEchoNormal) / `standard` (DeEchoAggressive) / `aggressive` (MDX-Net → DeEchoAggressive → HP3 → torchgate). QC via spectral flatness monotonic decrease across stages.

#### `batch-convert` — RVC over a folder
Scope: project | File: `.claude/agents/batch-convert.md`

Loops `tools/infer_cli.py` per file with skip-existing + 3-min per-file timeout. Per-file QC tracks rms_ratio (0.5–2.0 PASS) and duration drift. Reports done/failed/skipped/timed-out counts.

#### `prepare-dataset` — Raw → training-ready dataset
Scope: project | File: `.claude/agents/prepare-dataset.md`

Source types: `song` (HP3→HP5→DeEchoNormal) / `podcast` (HP3→DeEchoAggressive) / `clean-recording` (normalize only). Final dataset size gate enforces ≥10 min minimum before recommending `train-voice`.

#### `merge-models` — Blend 2 RVC `.pth` models
Scope: project | File: `.claude/agents/merge-models.md`

Linear weight interpolation via `infer.lib.train.process_ckpt.merge`. Validates architecture/version/sr match before merging. Merged model reuses index from one of the parents (no dedicated index built).

### i18n

Locale strings live in `i18n/locale/` as JSON files. `i18n/i18n.py` loads based on system locale. To add a language, add a JSON file and run the generation workflow (`.github/workflows/genlocale.yml`).
