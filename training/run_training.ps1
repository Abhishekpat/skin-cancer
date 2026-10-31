# Train the 3 models, then build the ensemble and the paper figures.
# Models that already finished (runs/<model>/run_info.json exists) are skipped,
# so this script can simply be started again after an interruption.
Set-Location -Path $PSScriptRoot
New-Item -ItemType Directory -Force runs | Out-Null
$log = "runs/pipeline.log"
"[$(Get-Date)] pipeline start" | Out-File -Append -Encoding utf8 $log
foreach ($m in "effnetv2_s", "convnext_t", "swin_t") {
    if (Test-Path "runs/$m/run_info.json") {
        "[$(Get-Date)] $m already trained, skipping" | Out-File -Append -Encoding utf8 $log
        continue
    }
    "[$(Get-Date)] training $m" | Out-File -Append -Encoding utf8 $log
    cmd /c "python -u train.py --model $m > runs\train_$m.log 2>&1"
    "[$(Get-Date)] $m finished with exit code $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
}
cmd /c "python -u ensemble_eval.py > runs\ensemble.log 2>&1"
"[$(Get-Date)] ensemble exit code $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
cmd /c "python -u make_figures.py > runs\figures.log 2>&1"
"[$(Get-Date)] figures exit code $LASTEXITCODE" | Out-File -Append -Encoding utf8 $log
"[$(Get-Date)] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
