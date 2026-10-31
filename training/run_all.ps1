# Run the full pipeline: data -> 3 models -> ensemble -> figures.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
python prepare_data.py --delete-zips
foreach ($m in "effnetv2_s", "convnext_t", "swin_t") {
    python train.py --model $m --epochs 20 --patience 6
}
python ensemble_eval.py
python make_figures.py
