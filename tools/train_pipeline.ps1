# =============================================================
# RVC Training Pipeline - Standalone PowerShell Script
# (Alternative to rvc_pipeline.ps1 -Mode train; uses embedded config)
#
# Implements all 14 tasks of the train-voice agent.
# Usage:
#   1. Edit the CONFIG block below
#   2. Run: .\tools\train_pipeline.ps1
# Run from project root: d:\AI_Working_place\Retrieval-based-Voice-Conversion-WebUI
# =============================================================

# ════════════════════════════════════════════════════════════
# CONFIG  — edit these
# ════════════════════════════════════════════════════════════
$MODEL_NAME    = "MyModel"                     # experiment name (no spaces)
$DATASET       = "E:\Data_voice\my_dataset"    # folder containing .wav/.mp3 files
$SR            = "40k"                         # 40k | 48k | 32k
$VERSION       = "v2"                          # v1 (256-dim, faster) | v2 (768-dim, better quality)
$EPOCHS        = 150                           # total training epochs
$BATCH_SIZE    = 4                             # 4 for VRAM 5-6GB, 2 if OOM, 8+ for ≥8GB
$FP16          = $false                        # true for VRAM ≥6GB + Tensor Cores
$F0_METHOD     = "rmvpe"                       # rmvpe (best) | harvest | crepe | pm
$GPU           = "0"                           # GPU id
$SKIP_PROBE    = $false                        # set $true to skip cleanliness probe (Step 1.5)
$FORCE_DIRTY   = $false                        # force-continue even if probe says DIRTY
$FORCE_SMALL   = $false                        # force-continue if dataset <100 slices
# ════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"
$SR_HZ      = [int]($SR -replace "k","") * 1000
$FEAT_DIR   = if ($VERSION -eq "v1") { "3_feature256" } else { "3_feature768" }
$FEAT_DIM   = if ($VERSION -eq "v1") { 256 } else { 768 }
$EXP_DIR    = "logs/$MODEL_NAME"
$SAVE_EVERY = [Math]::Max(10, [int]($EPOCHS / 10))
$N_CPU      = [Math]::Min([Environment]::ProcessorCount, 8)
$EPOCH_LOG  = "$EXP_DIR/epoch_log.txt"

# ── env vars required by RVC modules ─────────────────────────
$env:PYTHONPATH        = "D:\python_extra"
$env:PYTHONIOENCODING  = "utf-8"
$env:USE_LIBUV         = "0"
$env:weight_root       = "assets/weights"
$env:weight_uvr5_root  = "assets/uvr5_weights"
$env:index_root        = "logs"
$env:outside_index_root = "assets/indices"
$env:rmvpe_root        = "assets/rmvpe"

function Step($n, $msg)  { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)         { Write-Host "  OK:   $msg" -ForegroundColor Green }
function Warn($msg)       { Write-Host "  WARN: $msg" -ForegroundColor Yellow }
function Fail($msg)       { Write-Host "  FAIL: $msg" -ForegroundColor Red; exit 1 }
function Block($msg, $force) { Write-Host "  BLOCK: $msg" -ForegroundColor Red; if (-not $force) { exit 1 } }
function LogEpochLine($path, $line) { Add-Content -Path $path -Value $line; Write-Host $line -ForegroundColor Green }

Step "0/14" "Pre-flight & hardware"
python -c @"
import torch, os
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'  GPU: {p.name} ({p.total_memory // 1024**3} GB)')
else:
    print('  GPU: none')
print(f'  CPU: {os.cpu_count()} cores  (using {$N_CPU} workers)')
"@
if (-not (Test-Path $DATASET)) { Fail "Dataset not found: $DATASET" }
$srcCount = (Get-ChildItem $DATASET -Recurse -Include "*.wav","*.mp3" -EA SilentlyContinue).Count
Write-Host "  Dataset: $srcCount file(s) at $DATASET"
Write-Host "  Output:  $EXP_DIR"

# Task 1.5 — Cleanliness probe
if (-not $SKIP_PROBE) {
    Step "1/14" "Cleanliness probe"
    $probe = python -c @"
import os, random, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
src = r'$DATASET'
files = [f for f in os.listdir(src) if f.lower().endswith(('.wav','.mp3'))]
sample = random.sample(files, min(5, len(files)))
ps, ls, fs = [], [], []
for fn in sample:
    y, sr = librosa.load(os.path.join(src, fn), sr=22050, mono=True, duration=15)
    _, perc = librosa.effects.hpss(y)
    ps.append(float(np.sqrt(np.mean(perc**2)) / (np.sqrt(np.mean(y**2)) + 1e-8)))
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    ls.append(float(S[freqs < 120].sum() / (S.sum() + 1e-8)))
    fs.append(float(librosa.feature.spectral_flatness(y=y).mean()))
ap = np.mean(ps); al = np.mean(ls); af = np.mean(fs)
dirty = int(ap>0.35) + int(al>0.20) + int(af>0.15)
print(f'  perc={ap:.3f} low={al:.3f} flat={af:.4f} dirty={dirty}/3')
"@
    Write-Host $probe
    $dirty = [int](($probe | Select-String 'dirty=(\d)').Matches[0].Groups[1].Value)
    if ($dirty -ge 2)      { Block "DIRTY ($dirty/3). Run UVR5 first or set `$FORCE_DIRTY=`$true." $FORCE_DIRTY }
    elseif ($dirty -eq 1)  { Warn "1/3 dirty — borderline" }
    else                   { Ok "Clean (0/3)" }
} else {
    Step "1/14" "Probe SKIPPED"
}

# Task 5 — Preprocess
Step "2/14" "Preprocess (slice + 16kHz)"
New-Item -ItemType Directory -Force -Path $EXP_DIR | Out-Null
python infer/modules/train/preprocess.py "$DATASET" $SR_HZ $N_CPU "$EXP_DIR" False 3.0
if ($LASTEXITCODE -ne 0) { Fail "Preprocess failed" }
$wavCount = (Get-ChildItem "$EXP_DIR/1_16k_wavs" -Filter *.wav -EA SilentlyContinue).Count
Ok "$wavCount slices"

# Task 6 — Dataset size gate
Step "3/14" "Dataset size gate"
$mins = [math]::Round($wavCount * 3 / 60, 1)
$avgKB = [math]::Round((Get-ChildItem "$EXP_DIR/1_16k_wavs" -Filter *.wav | Measure-Object Length -Average).Average / 1KB, 1)
Write-Host "  $wavCount slices ≈ $mins min, avg $avgKB KB"
if ($avgKB -lt 30) { Warn "avg file $avgKB KB low — may be silent" }
if ($wavCount -lt 100) {
    Block "Too small ($wavCount). Set `$FORCE_SMALL=`$true to override." $FORCE_SMALL
} elseif ($wavCount -lt 200) {
    Warn "<10min — bumping epochs to 300"
    $EPOCHS = 300; $SAVE_EVERY = 30
} elseif ($wavCount -gt 1200) {
    Warn ">60min — capping epochs to 100"
    $EPOCHS = 100; $SAVE_EVERY = 10
} else {
    Ok "$mins min — sweet spot"
}

# Task 7 — F0
Step "4/14" "Extract F0 [$F0_METHOD]"
python infer/modules/train/extract/extract_f0_print.py "$EXP_DIR" $N_CPU $F0_METHOD
if ($LASTEXITCODE -ne 0) { Fail "F0 failed" }
Ok "$((Get-ChildItem "$EXP_DIR/2a_f0" -Filter *.npy).Count) F0 files"

# Task 8 — HuBERT features
Step "5/14" "HuBERT [$VERSION → $FEAT_DIM-dim]"
python infer/modules/train/extract_feature_print.py "cuda:0" 1 0 0 "$EXP_DIR" $VERSION $FP16.ToString().ToLower()
if ($LASTEXITCODE -ne 0) { Fail "HuBERT failed (check fairseq weights_only fix)" }
Ok "$((Get-ChildItem "$EXP_DIR/$FEAT_DIR" -Filter *.npy).Count) feature files"

# Task 9 — Generate config.json + filelist.txt
Step "6/14" "Generate config + filelist"
python -c @"
import os, json
exp = r'$EXP_DIR'; ver = '$VERSION'; sr = '$SR'; feat = '$FEAT_DIR'
src_cfg = f'configs/{ver}/{sr}.json'
if not os.path.exists(src_cfg): src_cfg = f'configs/v1/{sr}.json'
with open(src_cfg) as f: cfg = json.load(f)
cfg['train']['batch_size'] = $BATCH_SIZE
cfg['train']['fp16_run']   = $($FP16.ToString().ToLower())
with open(os.path.join(exp, 'config.json'), 'w') as f: json.dump(cfg, f, indent=2)

gt = os.path.join(exp, '0_gt_wavs')
names = sorted({os.path.splitext(f)[0] for f in os.listdir(gt) if f.endswith('.wav')})
lines = []
for n in names:
    parts = [
        os.path.join(gt, n + '.wav').replace('\\\\','/'),
        os.path.join(exp, feat, n + '.npy').replace('\\\\','/'),
        os.path.join(exp, '2a_f0',    n + '.wav.npy').replace('\\\\','/'),
        os.path.join(exp, '2b-f0nsf', n + '.wav.npy').replace('\\\\','/'),
        '0',
    ]
    lines.append('|'.join(parts))
mute = '|'.join([
    f'logs/mute/0_gt_wavs/mute{sr}.wav',
    f'logs/mute/{feat}/mute.npy',
    'logs/mute/2a_f0/mute.wav.npy',
    'logs/mute/2b-f0nsf/mute.wav.npy',
    '0',
])
lines.extend([mute, mute])
with open(os.path.join(exp, 'filelist.txt'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print(f'  {len(lines)} filelist entries')
"@
if ($LASTEXITCODE -ne 0) { Fail "Config/filelist failed" }
Ok "config + filelist done"

# Task 10 — Train VITS (with per-epoch logging)
Step "7/14" "Train [epochs=$EPOCHS bs=$BATCH_SIZE fp16=$FP16]"
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training started — $EPOCHS epochs, bs=$BATCH_SIZE, fp16=$FP16" | Out-File $EPOCH_LOG -Encoding utf8
$epochStart = Get-Date
$epochCount = 0

python infer/modules/train/train.py `
    -e $MODEL_NAME -sr $SR -f0 1 -bs $BATCH_SIZE -g $GPU `
    -te $EPOCHS -se $SAVE_EVERY `
    -pg "assets/pretrained_$VERSION/f0G$SR.pth" `
    -pd "assets/pretrained_$VERSION/f0D$SR.pth" `
    -l 1 -c 0 -sw 0 -v $VERSION 2>&1 | ForEach-Object {
        $line = $_
        Write-Host $line
        if ($line -match '====>\s*Epoch:\s*(\d+).*\|\s*\((.+?)\)') {
            $ep = $matches[1]; $dur = $matches[2]
            $epochCount++
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            $elapsed = (Get-Date) - $epochStart
            $eta_s = if ($epochCount -gt 0) { [int](($elapsed.TotalSeconds / $epochCount) * ($EPOCHS - $ep)) } else { 0 }
            $eta = if ($eta_s -gt 0) { [TimeSpan]::FromSeconds($eta_s).ToString("hh\:mm\:ss") } else { "-" }
            LogEpochLine $EPOCH_LOG "[$ts] Epoch $ep/$EPOCHS  duration=$dur  ETA_remaining=$eta"
        }
        if ($line -match 'Saving model.*at epoch (\d+)') {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            LogEpochLine $EPOCH_LOG "[$ts] CHECKPOINT saved at epoch $($matches[1])"
        }
        if ($line -match 'Training is done') {
            $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            LogEpochLine $EPOCH_LOG "[$ts] Training is done"
        }
    }
if ($LASTEXITCODE -ne 0) { Fail "Training failed" }
Ok "Training complete — log: $EPOCH_LOG"

# Task 11 — Build FAISS index
Step "8/14" "Build FAISS index"
python -c @"
import os, numpy as np, faiss
exp = r'$EXP_DIR'; feat = os.path.join(exp, '$FEAT_DIR'); dim = $FEAT_DIM
big = np.concatenate([np.load(os.path.join(feat, f)) for f in sorted(os.listdir(feat))])
np.random.shuffle(big)
n = min(int(16*np.sqrt(len(big))), len(big)//39)
idx = faiss.index_factory(dim, f'IVF{n},Flat')
faiss.extract_index_ivf(idx).nprobe = 1
idx.train(big); idx.add(big)
out = os.path.join(exp, f'added_IVF{n}_Flat_nprobe_1_$MODEL_NAME`_$VERSION.index')
faiss.write_index(idx, out)
print(out)
"@
if ($LASTEXITCODE -ne 0) { Fail "Index build failed" }
$idx = (Get-ChildItem $EXP_DIR -Filter "added_*.index" | Select-Object -First 1).Name
Ok "Index: $EXP_DIR/$idx"

Write-Host "`n===== TRAINING DONE =====" -ForegroundColor Green
Write-Host "Model      : $EXP_DIR/G_2333333.pth"
Write-Host "Deployable : assets/weights/$MODEL_NAME.pth"
Write-Host "Index      : $EXP_DIR/$idx"
Write-Host "Epoch log  : $EPOCH_LOG"
