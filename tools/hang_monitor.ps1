# Generic hang detector for RVC pipeline steps.
# Writes status file logs/_monitor_<Name>.status with one line per check.
# Last line state is one of: RUNNING | DONE | HANG | ERROR | TIMEOUT.
#
# Usage examples:
#   .\tools\hang_monitor.ps1 -StepType train      -Name MyModel -LogFile logs/train.log -ProcessId 1234
#   .\tools\hang_monitor.ps1 -StepType uvr5       -Name song    -LogFile logs/uvr5.log  -ProcessId 1234
#   .\tools\hang_monitor.ps1 -StepType hubert     -Name MyModel -OutputDir logs/MyModel/3_feature768 -ProcessId 1234
#   .\tools\hang_monitor.ps1 -StepType preprocess -Name MyModel -OutputDir logs/MyModel/1_16k_wavs   -ProcessId 1234
#   .\tools\hang_monitor.ps1 -StepType f0         -Name MyModel -OutputDir logs/MyModel/2a_f0        -ProcessId 1234
#   .\tools\hang_monitor.ps1 -StepType rvc-infer  -Name song    -OutputFile out/song_converted.wav   -ProcessId 1234

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('train', 'uvr5', 'hubert', 'preprocess', 'f0', 'rvc-infer')]
    [string]$StepType,

    [Parameter(Mandatory = $true)] [string]$Name,
    [string]$LogFile,
    [string]$OutputDir,
    [string]$OutputFile,
    [int]$ProcessId = 0,
    [int]$TimeoutMin = 60,
    [int]$StallMin = 0,
    [int]$TotalEpochs = 0,     # train only: enables ETA + progress %
    [int]$ReportEveryEpochs = 5  # train only: write summary to .progress file every N epochs
)

# Default stall window per step (minutes of no activity → HANG)
$defaultStall = @{
    'train'      = 5
    'uvr5'       = 3
    'hubert'     = 2
    'preprocess' = 2
    'f0'         = 2
    'rvc-infer'  = 3
}
if ($StallMin -le 0) { $StallMin = $defaultStall[$StepType] }

if (-not (Test-Path 'logs')) { New-Item -ItemType Directory -Path 'logs' | Out-Null }
$statusFile = "logs/_monitor_$Name.status"
"RUNNING start=$(Get-Date -Format o) step=$StepType stall=${StallMin}min pid=$ProcessId" | Set-Content $statusFile -Encoding utf8

function Write-Status($state, $msg) {
    Add-Content $statusFile "$state ts=$(Get-Date -Format o) $msg" -Encoding utf8
    Write-Host "[$state] $msg"
}

function Get-GpuUtil {
    try {
        $out = & nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return [int]($out -split "`n")[0].Trim() }
    } catch {}
    return -1
}

function Get-TrainProgress {
    param($log, $totalEpochs)
    if (-not $log -or -not (Test-Path $log)) { return $null }
    $epochLines = Get-Content $log -ErrorAction SilentlyContinue | Select-String -Pattern '====> Epoch:\s*(\d+).*?\(([\d:.]+)\)'
    if (-not $epochLines) { return @{ epoch = 0; avgSec = 0; etaSec = -1 } }
    $epochs = @($epochLines | ForEach-Object {
            $m = $_.Matches[0]
            $epoch = [int]$m.Groups[1].Value
            $t = $m.Groups[2].Value
            # parse H:MM:SS.fff or MM:SS.fff
            $parts = $t.Split(':')
            $sec = 0.0
            if ($parts.Count -eq 3) { $sec = [double]$parts[0]*3600 + [double]$parts[1]*60 + [double]$parts[2] }
            elseif ($parts.Count -eq 2) { $sec = [double]$parts[0]*60 + [double]$parts[1] }
            else { $sec = [double]$parts[0] }
            @{ epoch = $epoch; sec = $sec }
        })
    $lastEpoch = $epochs[-1].epoch
    # Sliding window: last 5 epochs (more stable than overall avg if dataset/lr changes)
    $windowStart = [math]::Max(0, $epochs.Count - 5)
    $window = $epochs[$windowStart..($epochs.Count - 1)]
    $sum = 0.0
    foreach ($e in $window) { $sum += $e.sec }
    $avgSec = if ($window.Count -gt 0) { [math]::Round($sum / $window.Count, 1) } else { 0 }
    $etaSec = if ($totalEpochs -gt 0 -and $lastEpoch -lt $totalEpochs) { [math]::Round($avgSec * ($totalEpochs - $lastEpoch)) } else { -1 }
    return @{ epoch = $lastEpoch; avgSec = $avgSec; etaSec = $etaSec; epochCount = $epochs.Count }
}

function Format-Duration {
    param([double]$sec)
    if ($sec -lt 0) { return 'n/a' }
    if ($sec -lt 60) { return "$([math]::Round($sec))s" }
    $h = [int][math]::Floor($sec / 3600)
    $m = [int][math]::Floor(($sec % 3600) / 60)
    if ($h -gt 0) { return ('{0}h{1:D2}m' -f $h, $m) } else { return "${m}m" }
}

function Get-Signature {
    param($type, $log, $dir)
    switch ($type) {
        'train' {
            $lines = if ($log -and (Test-Path $log)) { (Get-Content $log -ErrorAction SilentlyContinue | Measure-Object -Line).Lines } else { 0 }
            $gpu = Get-GpuUtil
            $prog = Get-TrainProgress -log $log -totalEpochs $TotalEpochs
            $progStr = if ($prog) {
                $pct = if ($TotalEpochs -gt 0) { [math]::Round(100.0 * $prog.epoch / $TotalEpochs, 1) } else { 0 }
                "epoch=$($prog.epoch)/$TotalEpochs ($pct%) avg=$(Format-Duration $prog.avgSec) eta=$(Format-Duration $prog.etaSec)"
            } else { 'epoch=?' }
            return @{ sig = "$progStr lines=$lines gpu=$gpu"; gpuActive = $gpu -gt 5; progress = $prog }
        }
        { $_ -in 'uvr5', 'rvc-infer' } {
            $lines = if ($log -and (Test-Path $log)) { (Get-Content $log -ErrorAction SilentlyContinue | Measure-Object -Line).Lines } else { 0 }
            return @{ sig = "lines=$lines"; gpuActive = $false }
        }
        { $_ -in 'hubert', 'preprocess', 'f0' } {
            $count = if ($dir -and (Test-Path $dir)) { (Get-ChildItem $dir -File -ErrorAction SilentlyContinue).Count } else { 0 }
            return @{ sig = "files=$count"; gpuActive = $false }
        }
    }
}

function Test-Completion {
    param($type, $log, $file)
    if ($file -and (Test-Path $file) -and (Get-Item $file).Length -gt 1024) { return $true }
    if ($type -eq 'train' -and $log -and (Test-Path $log)) {
        if (Get-Content $log -ErrorAction SilentlyContinue | Select-String -Pattern 'Training is done|saving final ckpt' -Quiet) { return $true }
    }
    return $false
}

$start = Get-Date
$lastActivity = Get-Date
$lastSig = ''
$lastReportedEpoch = -1
$progressFile = "logs/_monitor_$Name.progress"
$checks = $TimeoutMin * 6  # poll every 10s

for ($i = 0; $i -lt $checks; $i++) {
    Start-Sleep -Seconds 10
    $now = Get-Date
    $elapsedMin = ($now - $start).TotalMinutes

    $cur = Get-Signature -type $StepType -log $LogFile -dir $OutputDir
    if ($cur.sig -ne $lastSig -or $cur.gpuActive) { $lastActivity = $now }
    $lastSig = $cur.sig
    $stallNow = ($now - $lastActivity).TotalMinutes

    # Train: periodic .progress report every N epochs
    if ($StepType -eq 'train' -and $cur.progress -and $cur.progress.epoch -gt 0) {
        $ep = $cur.progress.epoch
        $stepDelta = $ep - $lastReportedEpoch
        if ($stepDelta -ge $ReportEveryEpochs -or $lastReportedEpoch -eq -1) {
            $pct = if ($TotalEpochs -gt 0) { [math]::Round(100.0 * $ep / $TotalEpochs, 1) } else { 0 }
            $elapsedMinFmt = Format-Duration ($elapsedMin * 60)
            $line = "[$(Get-Date -Format 'HH:mm:ss')] epoch $ep/$TotalEpochs ($pct%) avg=$(Format-Duration $cur.progress.avgSec)/epoch elapsed=$elapsedMinFmt eta=$(Format-Duration $cur.progress.etaSec)"
            Add-Content $progressFile $line -Encoding utf8
            Write-Host "PROGRESS $line" -ForegroundColor Cyan
            $lastReportedEpoch = $ep
        }
    }

    if (Test-Completion -type $StepType -log $LogFile -file $OutputFile) {
        Write-Status 'DONE' "elapsed=$([math]::Round($elapsedMin,1))min $($cur.sig)"
        return
    }

    if ($ProcessId -gt 0) {
        try { $proc = Get-Process -Id $ProcessId -ErrorAction Stop } catch { $proc = $null }
        if (-not $proc) {
            Write-Status 'ERROR' "process $ProcessId exited unexpectedly elapsed=$([math]::Round($elapsedMin,1))min $($cur.sig)"
            return
        }
    }

    if ($stallNow -ge $StallMin) {
        Write-Status 'HANG' "no activity for $([math]::Round($stallNow,1))min (>=${StallMin}min) pid=$ProcessId $($cur.sig)"
        if ($ProcessId -gt 0 -and (Get-Command py-spy -ErrorAction SilentlyContinue)) {
            try { & py-spy dump --pid $ProcessId 2>&1 | Out-File "logs/_monitor_$Name.stack" -Encoding utf8 } catch {}
            Write-Host "Stack dumped to logs/_monitor_$Name.stack"
        }
        return
    }

    Write-Host "[t+$([math]::Round($elapsedMin,1))min stall=$([math]::Round($stallNow,1))min/${StallMin}min] $($cur.sig)"
}

Write-Status 'TIMEOUT' "monitor reached TimeoutMin=$TimeoutMin without completion last=$lastSig"
