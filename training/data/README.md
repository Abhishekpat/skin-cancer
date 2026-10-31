# Training data

These files list every image used in the project and its label. They are produced by `prepare_data.py`.

| File | Images | What it is |
|---|---|---|
| `train.csv` | 18,076 | ISIC 2019 images the models learn from |
| `val.csv` | 3,659 | ISIC 2019 images used to pick the best epoch, calibrate and set the ensemble weights |
| `test.csv` | 3,596 | ISIC 2019 internal test, used once at the end |
| `external.csv` | 2,298 | PAD-UFES-20 smartphone images, never seen in training (external test) |
| `dataset_summary.json` | | number of images per class in each split |
| `raw/ISIC_2019_Training_GroundTruth.csv` | 25,331 | official ISIC 2019 labels |
| `raw/ISIC_2019_Training_Metadata.csv` | 25,331 | official ISIC 2019 metadata (age, sex, body site, lesion id) |

## Columns

- `path`: the resized image file that the scripts use (`cache/...`).
- `label`: class number:
  - 0 = MEL (melanoma)
  - 1 = NV (nevus)
  - 2 = BCC (basal cell carcinoma)
  - 3 = AK (actinic keratosis)
  - 4 = BKL (benign keratosis)
  - 5 = DF (dermatofibroma)
  - 6 = VASC (vascular lesion)
  - 7 = SCC (squamous cell carcinoma)
- `group`: images of the same lesion, or near-duplicate images, share a group and are always in the same split, so there is no leakage.
- `image`, `lesion_id`, `age_approx`, `sex`, `anatom_site_general`, `source`: copied from the dataset metadata.

## Images per class

| Class | Train | Val | Test | External |
|---|---|---|---|---|
| MEL | 3,210 | 671 | 641 | 52 |
| NV | 9,236 | 1,810 | 1,829 | 244 |
| BCC | 2,361 | 507 | 455 | 845 |
| AK | 629 | 110 | 128 | 730 |
| BKL | 1,843 | 386 | 395 | 235 |
| DF | 144 | 36 | 59 | 0 |
| VASC | 193 | 39 | 21 | 0 |
| SCC | 460 | 100 | 68 | 192 |

## Images

The image files (about 590 MB after resizing) are not in this repository. They belong to the dataset owners. Download them from the official sources:

- **ISIC 2019** (CC BY-NC 4.0): https://challenge.isic-archive.com/data/#2019
- **PAD-UFES-20** (CC BY 4.0): https://data.mendeley.com/datasets/zr7vgbcyr2/1

Then run `python prepare_data.py`. It recreates the `cache/` folder and the same splits.

Example images with the model's predictions are in [`../results/explain/`](../results/explain/).
