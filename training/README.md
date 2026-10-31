# Skin Cancer Detection: Training the 3-Model Ensemble

This folder trains the models for the IEEE paper and produces the
**real** accuracy values and graphs used in it.

- **2 datasets:**
  - **ISIC 2019** (25,331 dermoscopic images, 8 classes): train / validation / internal test.
  - **PAD-UFES-20** (2,298 smartphone images): external test only.
- **3 algorithms:**
  - **EfficientNetV2-S** (CNN)
  - **ConvNeXt-T** (CNN)
  - **Swin-T** (transformer)
- **USP:** the three models are combined by **weighted soft voting**
  (weights learned on the validation set), with temperature calibration
  and a referral threshold set for 95% malignant sensitivity.

## How to run (Windows, NVIDIA GPU)

```powershell
pip install -r requirements.txt
python prepare_data.py --delete-zips    # download + resize + split (about 20-60 min, depends on internet)
python train.py --model effnetv2_s      # ~20-25 min on an RTX 3060 laptop GPU
python train.py --model convnext_t      # ~20-25 min
python train.py --model swin_t          # ~25-30 min
python ensemble_eval.py                 # combine the 3 models + all metrics (1 min)
python make_figures.py                  # graphs + LaTeX tables for the paper (1 min)
```
Or run everything at once with `./run_all.ps1`.

## How long it takes (measured on this laptop)

| Step | Time |
|---|---|
| Download ISIC 2019 (9.8 GB) + PAD-UFES-20 (3.6 GB) | depends on internet (about 30-90 min) |
| Resize + split | about 5-10 min |
| Training, 3 models x up to 20 epochs | about 1-1.5 hours (240-295 images/s, under 3 GB GPU memory) |
| Ensemble + figures | about 2 min |

Without a GPU (CPU only), training would take days. Use Kaggle or Google
Colab (free T4 GPU) instead; the same scripts run there unchanged.

## What the split looks like
- Images of the same lesion (`lesion_id`) and near-identical images (same
  image hash) are always kept in the **same** split, so there is no leakage.
- ISIC 2019 is split stratified by class: about 71% train, 14% validation,
  14% internal test.
- PAD-UFES-20 is never seen during training. Its 6 classes are mapped to
  ISIC names: ACK→AK, NEV→NV, SEK→BKL; BCC, SCC and MEL stay the same.

## Output files
- `runs/<model>/history.csv`: per-epoch loss, accuracy and balanced accuracy (training curves).
- `runs/<model>/best.pt`: trained weights.
- `results/metrics.json`: every metric for each model and the ensemble,
  with 95% bootstrap confidence intervals.
- `results/paper_assets/`: figures (PDF/PNG) and LaTeX tables used in the paper.

## Files
| File | Purpose |
|---|---|
| `common.py` | classes, dataset, transforms, models, focal loss, metrics |
| `prepare_data.py` | download, resize, de-duplicate, lesion-grouped split |
| `train.py` | fine-tune one backbone (focal loss, AdamW, cosine LR, bf16, early stopping, TTA) |
| `ensemble_eval.py` | temperature scaling, weight search, threshold, evaluation |
| `make_figures.py` | graphs and LaTeX tables for the paper |
| `benchmark_gpu.py` | measures training speed on your GPU |
| `make_evaluation_examples.py` | precision, confusion matrices with counts, example images (`results/explain/`) |

## Results of the run on 16-17 Oct 2026 (RTX 3060 Laptop GPU)

| System | Internal acc. | Internal BA | Macro-AUC | External acc. | External malignant AUC |
|---|---|---|---|---|---|
| EfficientNetV2-S | 75.19% | 58.88% | 0.943 | 33.68% | 0.833 |
| ConvNeXt-T | 76.89% | 60.21% | 0.936 | 34.73% | 0.844 |
| Swin-T | 73.08% | 57.80% | 0.935 | 35.34% | 0.844 |
| **Ensemble (equal weights)** | **77.95%** | **62.99%** | **0.950** | **37.25%** | **0.873** |
| Ensemble (learned weights) | 76.89% | 61.65% | 0.950 | 35.86% | 0.865 |

Training time: 43 / 49 / 36 min. Full metrics: `results/metrics.json`;
ablation over model combinations: `results/ablation_combinations.json`;
skin-type breakdown: `results/fitzpatrick_external.json`.

Note for Windows: `train.py` loads images with threads (not DataLoader
worker processes) and uses one PyTorch CPU thread. Worker processes ran
out of page-file memory on a 16 GB laptop. `run_training.ps1` runs the
whole pipeline and can be restarted; finished models are skipped.
