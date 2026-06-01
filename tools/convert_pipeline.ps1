# =============================================================
# RVC Convert-Voice Pipeline - Standalone PowerShell Script
# (Alternative to rvc_pipeline.ps1 -Mode convert; uses embedded config)
#
# Implements all 12 tasks of the convert-voice agent.
# Usage:
#   1. Edit the CONFIG block below
#   2. Run: .\tools\convert_pipeline.ps1
# Run from project root: d:\AI_Working_place\Retrieval-based-Voice-Conversion-WebUI
# =============================================================

# ════════════════════════════════════════════════════════════
# CONFIG  — edit these
# ════════════════════════════════════════════════════════════
$SOURCE        = "E:\Data_voice\song.mp3"      # source audio (.wav/.mp3)
$MODEL_NAME    = "Obito"                       # name (without .pth); model = assets/weights/$MODEL_NAME.pth
$OUTPUT        = "E:\Data_voice\output_final.wav"  # final output path
$HAS_MUSIC     = $true                         # true = UVR5 separation first; false = direct RVC (speech)
$BACKING_MODE  = "keep"                        # keep | discard | convert-all  (only relevant if HAS_MUSIC)
$F0UP_KEY      = 0                             # semitones (0 same, +12 male→female, -12 female→male)
$INDEX_RATE    = 0.75                          # 0.75 for music, 0.88 for speech (auto-adjusted below)
$F0_METHOD     = "rmvpe"                       # rmvpe | harvest | crepe | pm
$WORK_DIR      = "E:\Data_voice\convert_tmp"   # intermediate files folder
$HP_STAGE1     = "HP3_all_vocals"              # winner from corr benchmarks (0.074)
$HP_STAGE2     = "HP5_only_main_vocal"         # main vs backing
$AGG           = 10                            # UVR5 aggressiveness
# ════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

# ── env vars required ─────────────────────────────────────────
$env:PYTHONPATH        = "D:\python_extra"
$env:PYTHONIOENCODING  = "utf-8"
$env:weight_root       = "assets/weights"
$env:weight_uvr5_root  = "assets/uvr5_weights"
$env:index_root        = "logs"
$env:rmvpe_root        = "assets/rmvpe"

function Step($n, $msg)  { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)         { Write-Host "  PASS: $msg" -ForegroundColor Green }
function Warn($msg)       { Write-Host "  WARN: $msg" -ForegroundColor Yellow }
function Fail($msg)       { Write-Host "  FAIL: $msg" -ForegroundColor Red; exit 1 }

if ($INDEX_RATE -eq 0.75 -and -not $HAS_MUSIC) { $INDEX_RATE = 0.88 }

Step "0/12" "Pre-flight"
if (-not (Test-Path $SOURCE))       { Fail "Source not found: $SOURCE" }
$model_path = "assets/weights/$MODEL_NAME.pth"
if (-not (Test-Path $model_path))   { Fail "Model not found: $model_path" }
$index = (Get-ChildItem "logs/$MODEL_NAME" -Filter "added_*.index" -EA SilentlyContinue | Select-Object -First 1).FullName
if (-not $index)                    { Fail "No FAISS index in logs/$MODEL_NAME/" }
New-Item -ItemType Directory -Force -Path $WORK_DIR | Out-Null

python -c "import torch; print('  Device:', 'cuda:0' if torch.cuda.is_available() else 'cpu')" 2>&1 | Select-Object -Last 1
Write-Host "  Source:     $SOURCE"
Write-Host "  Model:      $model_path"
Write-Host "  Index:      $index"
Write-Host "  Output:     $OUTPUT"
Write-Host "  Has music:  $HAS_MUSIC"
Write-Host "  Backing:    $BACKING_MODE"
Write-Host "  Pitch:      $F0UP_KEY semitones"
Write-Host "  Index rate: $INDEX_RATE"

# SPEECH MODE — skip UVR5
if (-not $HAS_MUSIC) {
    Step "1/12" "SPEECH MODE — skipping UVR5"
    Step "8/12" "RVC convert (direct)"
    python tools/infer_cli.py --f0up_key $F0UP_KEY --input_path "$SOURCE" `
        --index_path "$index" --opt_path "$OUTPUT" --model_name "$MODEL_NAME.pth" `
        --index_rate $INDEX_RATE --f0method $F0_METHOD
    if ($LASTEXITCODE -ne 0) { Fail "RVC failed" }
    Step "11/12" "Final QC"
    python -c @"
import os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
y, sr = librosa.load(r'$OUTPUT', sr=None, mono=True)
print(f'  duration={len(y)/sr:.1f}s rms={float(np.sqrt(np.mean(y**2))):.4f} size={os.path.getsize(r\"$OUTPUT\")//1024}KB')
"@
    Write-Host "`n===== DONE (speech) =====" -ForegroundColor Green
    Write-Host "Output: $OUTPUT"
    exit 0
}

# Stage 1 — HP3
$STAGE1 = "$WORK_DIR\stage1_hp3"
Step "3/12" "UVR5 Stage 1 [$HP_STAGE1]"
python -c @"
import sys, os, shutil, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from infer.modules.uvr5.modules import uvr
out = r'$STAGE1'
if os.path.exists(out): shutil.rmtree(out)
os.makedirs(out, exist_ok=True)
list(uvr('$HP_STAGE1', '', out, [type('F',(),{'name':r'$SOURCE'})()], out, $AGG, 'wav'))
"@
if ($LASTEXITCODE -ne 0) { Fail "Stage 1 failed" }

Step "4/12" "QC Stage 1"
$qc1 = python -c @"
import os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
out = r'$STAGE1'
vf = next(os.path.join(out,f) for f in os.listdir(out) if f.startswith('vocal'))
if_ = next(os.path.join(out,f) for f in os.listdir(out) if f.startswith('instrument'))
v,_ = librosa.load(vf, sr=None, mono=True); i,_ = librosa.load(if_, sr=None, mono=True)
m = min(len(v), len(i))
corr = float(abs(np.corrcoef(v[:m], i[:m])[0,1]))
print(f'  vocal_rms={float(np.sqrt(np.mean(v**2))):.4f} inst_rms={float(np.sqrt(np.mean(i**2))):.4f} corr={corr:.3f}')
print(f'VOCAL_PATH={vf}')
print(f'INST_PATH={if_}')
"@
Write-Host $qc1
$corr1 = [double](($qc1 | Select-String 'corr=([\d.]+)').Matches[0].Groups[1].Value)
$inst_rms = [double](($qc1 | Select-String 'inst_rms=([\d.]+)').Matches[0].Groups[1].Value)
$stage1_vocal = ($qc1 | Select-String '^VOCAL_PATH=(.+)$').Matches[0].Groups[1].Value
$stage1_inst  = ($qc1 | Select-String '^INST_PATH=(.+)$').Matches[0].Groups[1].Value
if ($inst_rms -lt 0.01) { Fail "Instrumental silent — fp16 overflow bug" }
if ($corr1 -gt 0.40)    { Warn "corr=$corr1 poor — try HP2/HP5" }
else                    { Ok "corr=$corr1" }

# Stage 2 — HP5
if ($BACKING_MODE -eq "convert-all") {
    Write-Host "  [Stage 2 skipped — convert-all mode]"
    $main_vocal = $stage1_vocal
    $backing_vocal = $null
} else {
    $STAGE2 = "$WORK_DIR\stage2_hp5"
    Step "5/12" "UVR5 Stage 2 [$HP_STAGE2]"
    python -c @"
import sys, os, shutil, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from infer.modules.uvr5.modules import uvr
out = r'$STAGE2'
if os.path.exists(out): shutil.rmtree(out)
os.makedirs(out, exist_ok=True)
list(uvr('$HP_STAGE2', '', out, [type('F',(),{'name':r'$stage1_vocal'})()], out, $AGG, 'wav'))
"@
    if ($LASTEXITCODE -ne 0) { Fail "Stage 2 failed" }

    Step "6/12" "QC Stage 2"
    $qc2 = python -c @"
import os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
out = r'$STAGE2'
mv = next(os.path.join(out,f) for f in os.listdir(out) if f.startswith('vocal'))
bv = next(os.path.join(out,f) for f in os.listdir(out) if f.startswith('instrument'))
m,_ = librosa.load(mv, sr=None, mono=True); b,_ = librosa.load(bv, sr=None, mono=True)
n = min(len(m), len(b))
print(f'  main_rms={float(np.sqrt(np.mean(m**2))):.4f} backing_rms={float(np.sqrt(np.mean(b**2))):.4f} corr={float(abs(np.corrcoef(m[:n], b[:n])[0,1])):.3f}')
print(f'MAIN_PATH={mv}')
print(f'BACKING_PATH={bv}')
"@
    Write-Host $qc2
    $main_vocal    = ($qc2 | Select-String '^MAIN_PATH=(.+)$').Matches[0].Groups[1].Value
    $backing_vocal = ($qc2 | Select-String '^BACKING_PATH=(.+)$').Matches[0].Groups[1].Value
}

# Backing strategy → RVC input
Step "7/12" "Backing strategy: $BACKING_MODE"
$rvc_input = switch ($BACKING_MODE) {
    "keep"        { $main_vocal }
    "discard"     { $main_vocal }
    "convert-all" { $stage1_vocal }
    default       { Fail "Unknown BACKING_MODE: $BACKING_MODE" }
}
Write-Host "  RVC input: $rvc_input"

# RVC convert
$RVC_OUT = "$WORK_DIR\main_converted.wav"
Step "8/12" "RVC convert [model: $MODEL_NAME, f0up=$F0UP_KEY, rate=$INDEX_RATE]"
python tools/infer_cli.py --f0up_key $F0UP_KEY --input_path "$rvc_input" `
    --index_path "$index" --opt_path "$RVC_OUT" --model_name "$MODEL_NAME.pth" `
    --index_rate $INDEX_RATE --f0method $F0_METHOD
if ($LASTEXITCODE -ne 0) { Fail "RVC failed — check cuDNN GRU fix in infer/lib/rmvpe.py" }

Step "9/12" "QC RVC output"
$qcr = python -c @"
import os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
src,_ = librosa.load(r'$rvc_input', sr=None, mono=True)
out,_ = librosa.load(r'$RVC_OUT', sr=None, mono=True)
m = min(len(src), len(out))
ratio = float(np.sqrt(np.mean(out[:m]**2)) / (np.sqrt(np.mean(src[:m]**2)) + 1e-8))
print(f'  rms_ratio={ratio:.3f} size={os.path.getsize(r\"$RVC_OUT\")//1024}KB')
"@
Write-Host $qcr
$sizeKB = [int](($qcr | Select-String 'size=(\d+)').Matches[0].Groups[1].Value)
if ($sizeKB -lt 100) { Fail "Output silent/empty — cuDNN GRU bug?" }

# FFmpeg mix
Step "10/12" "FFmpeg mix ($BACKING_MODE)"
switch ($BACKING_MODE) {
    "keep" {
        ffmpeg -y -i "$RVC_OUT" -i "$backing_vocal" -i "$stage1_inst" `
            -filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.7[a1];[2:a]volume=1.0[a2];[a0][a1][a2]amix=inputs=3:duration=longest:normalize=0" `
            "$OUTPUT" 2>&1 | Select-String "Output|error" | Select-Object -First 2
    }
    default {
        ffmpeg -y -i "$RVC_OUT" -i "$stage1_inst" `
            -filter_complex "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0" `
            "$OUTPUT" 2>&1 | Select-String "Output|error" | Select-Object -First 2
    }
}
if (-not (Test-Path $OUTPUT)) { Fail "Mix failed" }
Ok "Mix done"

# Final QC
Step "11/12" "Final QC"
$qcf = python -c @"
import os, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
y, sr = librosa.load(r'$OUTPUT', sr=None, mono=True)
src,_ = librosa.load(r'$SOURCE', sr=None, mono=True)
dur = len(y)/sr; src_dur = len(src)/sr
print(f'  duration={dur:.1f}s (src={src_dur:.1f}s) rms={float(np.sqrt(np.mean(y**2))):.4f} size={os.path.getsize(r\"$OUTPUT\")/1024**2:.1f}MB')
"@
Write-Host $qcf

Write-Host "`n===== CONVERSION COMPLETE =====" -ForegroundColor Green
Write-Host "Source       : $SOURCE"
Write-Host "Model        : $MODEL_NAME (corr Stage 1 = $corr1)"
Write-Host "UVR5 stages  : $HP_STAGE1 + $HP_STAGE2"
Write-Host "Backing mode : $BACKING_MODE"
Write-Host "Output       : $OUTPUT"
Write-Host "Intermediates: $WORK_DIR/"
