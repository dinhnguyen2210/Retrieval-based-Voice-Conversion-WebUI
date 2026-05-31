# =============================================================
# RVC Pipeline — Combined Train + Convert
# Usage:
#   .\tools\rvc_pipeline.ps1 -Mode train     # train a new voice model
#   .\tools\rvc_pipeline.ps1 -Mode convert   # convert audio with trained model
#
# Config: edit tools\rvc_config.ps1  (this file should not be modified)
# Per-epoch training log: logs\<MODEL>\epoch_log.txt
# Run from project root: d:\AI_Working_place\Retrieval-based-Voice-Conversion-WebUI
# =============================================================

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("train","convert")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$cfg = Join-Path $PSScriptRoot "rvc_config.ps1"
if (-not (Test-Path $cfg)) { Write-Host "Config not found: $cfg" -ForegroundColor Red; exit 1 }
. $cfg

function Step($n, $msg)  { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)         { Write-Host "  PASS: $msg" -ForegroundColor Green }
function Warn($msg)       { Write-Host "  WARN: $msg" -ForegroundColor Yellow }
function Fail($msg)       { Write-Host "  FAIL: $msg" -ForegroundColor Red; exit 1 }
function Block($msg, $force) { Write-Host "  BLOCK: $msg" -ForegroundColor Red; if (-not $force) { exit 1 } }
function LogEpochLine($path, $line) {
    Add-Content -Path $path -Value $line
    Write-Host $line -ForegroundColor Green
}

# ============================================================
# ============================  TRAIN  =======================
# ============================================================
if ($Mode -eq "train") {

    $MODEL_NAME = $T_MODEL_NAME; $DATASET = $T_DATASET; $SR = $T_SR
    $VERSION    = $T_VERSION;     $EPOCHS  = $T_EPOCHS; $BATCH_SIZE = $T_BATCH_SIZE
    $FP16       = $T_FP16;        $F0_METHOD = $T_F0_METHOD; $GPU = $T_GPU
    $SR_HZ      = [int]($SR -replace "k","") * 1000
    $FEAT_DIR   = if ($VERSION -eq "v1") { "3_feature256" } else { "3_feature768" }
    $FEAT_DIM   = if ($VERSION -eq "v1") { 256 } else { 768 }
    $EXP_DIR    = "logs/$MODEL_NAME"
    $SAVE_EVERY = [Math]::Max(10, [int]($EPOCHS / 10))
    $N_CPU      = [Math]::Min([Environment]::ProcessorCount, 8)
    $EPOCH_LOG  = "$EXP_DIR/epoch_log.txt"

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

    # === Task 2 — Cleanliness probe ===
    if (-not $T_SKIP_PROBE) {
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
        if ($dirty -ge 2)      { Block "DIRTY ($dirty/3). Run UVR5 first or set `$T_FORCE_DIRTY=`$true." $T_FORCE_DIRTY }
        elseif ($dirty -eq 1)  { Warn "1/3 dirty — borderline" }
        else                   { Ok "Clean (0/3)" }
    } else {
        Step "1/14" "Probe SKIPPED"
    }

    # === Task 5 — Preprocess ===
    Step "2/14" "Preprocess (slice + 16kHz resample)"
    New-Item -ItemType Directory -Force -Path $EXP_DIR | Out-Null
    python infer/modules/train/preprocess.py "$DATASET" $SR_HZ $N_CPU "$EXP_DIR" False 3.0
    if ($LASTEXITCODE -ne 0) { Fail "Preprocess failed" }
    $wavCount = (Get-ChildItem "$EXP_DIR/1_16k_wavs" -Filter *.wav -EA SilentlyContinue).Count
    Ok "$wavCount slices"

    # === Task 6 — Dataset size gate ===
    Step "3/14" "Dataset size gate"
    $mins = [math]::Round($wavCount * 3 / 60, 1)
    $avgKB = [math]::Round((Get-ChildItem "$EXP_DIR/1_16k_wavs" -Filter *.wav | Measure-Object Length -Average).Average / 1KB, 1)
    Write-Host "  $wavCount slices ≈ $mins min, avg $avgKB KB"
    if ($avgKB -lt 30) { Warn "avg file $avgKB KB low — may be silent" }
    if ($wavCount -lt 100) {
        Block "Too small ($wavCount slices). Need ≥10 min. Set `$T_FORCE_SMALL=`$true to override." $T_FORCE_SMALL
    } elseif ($wavCount -lt 200) {
        Warn "Below 10min — bumping epochs $EPOCHS → 300"
        $EPOCHS = 300; $SAVE_EVERY = 30
    } elseif ($wavCount -gt 1200) {
        Warn "Above 60min — capping epochs to 100"
        $EPOCHS = 100; $SAVE_EVERY = 10
    } else {
        Ok "$mins min — sweet spot"
    }

    # === Task 7 — F0 ===
    Step "4/14" "Extract F0 [$F0_METHOD]"
    python infer/modules/train/extract/extract_f0_print.py "$EXP_DIR" $N_CPU $F0_METHOD
    if ($LASTEXITCODE -ne 0) { Fail "F0 failed" }
    Ok "$((Get-ChildItem "$EXP_DIR/2a_f0" -Filter *.npy).Count) F0 files"

    # === Task 8 — HuBERT features ===
    Step "5/14" "HuBERT features [$VERSION → $FEAT_DIM-dim]"
    python infer/modules/train/extract_feature_print.py "cuda:0" 1 0 0 "$EXP_DIR" $VERSION $FP16.ToString().ToLower()
    if ($LASTEXITCODE -ne 0) { Fail "HuBERT failed (check fairseq weights_only fix)" }
    Ok "$((Get-ChildItem "$EXP_DIR/$FEAT_DIR" -Filter *.npy).Count) feature files"

    # === Task 9 — Generate config.json + filelist.txt ===
    Step "6/14" "Generate config.json + filelist.txt"
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

    # === Task 10 — Train VITS (with per-epoch logging) ===
    Step "7/14" "Train  [epochs=$EPOCHS bs=$BATCH_SIZE fp16=$FP16]"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training started — $EPOCHS epochs, bs=$BATCH_SIZE, fp16=$FP16" | Out-File $EPOCH_LOG -Encoding utf8
    $epochStart = Get-Date
    $epochCount = 0

    # Pipe train.py output through a filter that captures epoch completions
    python infer/modules/train/train.py `
        -e $MODEL_NAME -sr $SR -f0 1 -bs $BATCH_SIZE -g $GPU `
        -te $EPOCHS -se $SAVE_EVERY `
        -pg "assets/pretrained_$VERSION/f0G$SR.pth" `
        -pd "assets/pretrained_$VERSION/f0D$SR.pth" `
        -l 1 -c 0 -sw 0 -v $VERSION 2>&1 | ForEach-Object {
            $line = $_
            # Echo everything to console (with reduced verbosity below could filter)
            Write-Host $line
            # Detect "====> Epoch: N [timestamp] | (duration)" lines
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

    # === Task 11 — Build FAISS index ===
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
    exit 0
}

# ============================================================
# ===========================  CONVERT  ======================
# ============================================================
if ($Mode -eq "convert") {

    $SOURCE = $C_SOURCE; $MODEL_NAME = $C_MODEL_NAME; $OUTPUT = $C_OUTPUT
    $HAS_MUSIC = $C_HAS_MUSIC; $BACKING_MODE = $C_BACKING_MODE
    $F0UP_KEY = $C_F0UP_KEY; $INDEX_RATE = $C_INDEX_RATE; $F0_METHOD = $C_F0_METHOD
    $WORK_DIR = $C_WORK_DIR; $HP_STAGE1 = $C_HP_STAGE1; $HP_STAGE2 = $C_HP_STAGE2; $AGG = $C_AGG

    if ($INDEX_RATE -eq 0.75 -and -not $HAS_MUSIC) { $INDEX_RATE = 0.88 }
    if (-not (Test-Path $SOURCE))   { Fail "Source not found: $SOURCE" }
    $model_path = "assets/weights/$MODEL_NAME.pth"
    if (-not (Test-Path $model_path)) { Fail "Model not found: $model_path" }
    $index = (Get-ChildItem "logs/$MODEL_NAME" -Filter "added_*.index" -EA SilentlyContinue | Select-Object -First 1).FullName
    if (-not $index) { Fail "No FAISS index in logs/$MODEL_NAME/" }
    New-Item -ItemType Directory -Force -Path $WORK_DIR | Out-Null

    Step "0/12" "Pre-flight"
    python -c "import torch; print('  Device:', 'cuda:0' if torch.cuda.is_available() else 'cpu')" 2>&1 | Select-Object -Last 1
    Write-Host "  Source       : $SOURCE"
    Write-Host "  Model        : $model_path"
    Write-Host "  Index        : $index"
    Write-Host "  Output       : $OUTPUT"
    Write-Host "  Has music    : $HAS_MUSIC"
    Write-Host "  Backing      : $BACKING_MODE"
    Write-Host "  Pitch / rate : $F0UP_KEY semitones / $INDEX_RATE"

    # === SPEECH mode (skip UVR5) ===
    if (-not $HAS_MUSIC) {
        Step "8/12" "RVC direct (speech)"
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

    # === Stage 1 — HP3 ===
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

    # === Stage 2 — HP5 ===
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

    # === Backing strategy → RVC input ===
    Step "7/12" "Backing strategy: $BACKING_MODE"
    $rvc_input = switch ($BACKING_MODE) {
        "keep"        { $main_vocal }
        "discard"     { $main_vocal }
        "convert-all" { $stage1_vocal }
        default       { Fail "Unknown BACKING_MODE: $BACKING_MODE" }
    }
    Write-Host "  RVC input: $rvc_input"

    # === RVC convert ===
    $RVC_OUT = "$WORK_DIR\main_converted.wav"
    Step "8/12" "RVC convert"
    python tools/infer_cli.py --f0up_key $F0UP_KEY --input_path "$rvc_input" `
        --index_path "$index" --opt_path "$RVC_OUT" --model_name "$MODEL_NAME.pth" `
        --index_rate $INDEX_RATE --f0method $F0_METHOD
    if ($LASTEXITCODE -ne 0) { Fail "RVC failed — check cuDNN GRU fix" }

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

    # === FFmpeg mix ===
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

    # === Final QC ===
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

    Write-Host "`n===== CONVERSION DONE =====" -ForegroundColor Green
    Write-Host "Source : $SOURCE"
    Write-Host "Model  : $MODEL_NAME (corr Stage 1 = $corr1)"
    Write-Host "Backing: $BACKING_MODE"
    Write-Host "Output : $OUTPUT"
    Write-Host "Tmp    : $WORK_DIR/"
}
