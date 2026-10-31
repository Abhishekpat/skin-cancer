# Paper Guide: "Skin Cancer Detection Using Deep Learning"

This guide explains what each part of the IEEE paper means, why it is
there, and how it meets your teacher's requirements. It also gives a
reviewer-style assessment and a to-do list for before you submit.
It was updated after the models were trained (16–17 Oct 2026).

Files:
- `sc_two_column.pdf` is the main IEEE conference format (14 pages,
  with measured results).
- `sc_one_column.pdf` has the same content in one-column format.
- `Training_Results_Report.pdf` lists all measured numbers and graphs.
- The source files are `sc_body.tex` (all content), `sc_preamble.tex`,
  `references.bib`, `figures/` (result graphs), and
  `sc_two_column.tex` / `sc_one_column.tex`.
- Build with `./build.ps1`, or upload the `.tex`, `.bib` and `figures/`
  to Overleaf with `sc_two_column.tex` as the main file.

---

## 1. Teacher's checklist: where each item is

| Requirement | Where in the paper | What you get |
|---|---|---|
| **Literature review** | Section II + **Table I** (22 studies) | 7 themes. Ends with **6 research gaps (G1–G6)**. |
| **Proposed methodology** | Section IV + **Fig. 3** + **Algorithm 1** + **Table III** + Eqs. (1)–(5) | The 3-model ensemble (EfficientNetV2-S + ConvNeXt-T + Swin-T), focal loss, and calibrated soft voting are implemented. Segmentation, metadata and Grad-CAM are the next stage. |
| **2 datasets, 3 algorithms (2 CNN + 1 transformer)** | Sections III, IV, VI | ISIC 2019 (training + internal test) and PAD-UFES-20 (external test). EfficientNetV2-S and ConvNeXt-T (CNNs) and Swin-T (transformer). |
| **Accuracy with conclusion metrics** | Section VI (**Tables IV–VII**), Abstract, Section IX | **77.95% accuracy**, 62.99% balanced accuracy, 64.79% macro-F1, AUC 0.950, 95.4% cancer sensitivity. |
| **Graph of the trained model's metrics** | **Figs. 4–7** | Training curves, comparison bar charts (internal and external), confusion matrix, ROC curves, all from our own training. |
| **USP: combining multiple algorithms** | **Table VI** (ablation) | 1 model < any 2 models < all 3 models on every metric. |
| **Comparative analysis of multiple algorithms at the end** | Section VII + **Table VIII** + **Figs. 8–12** + **Table IX** | 11 algorithm families from the literature, plus our own result in Panel B. |
| **Plagiarism below 10%** | Whole paper | Written from scratch. See Section 5 for how to verify it. |

---

## 2. What each section means and why it is there

**Title.** "...A Comparative Analysis and a Multi-Model CNN–Transformer
Ensemble with External Validation". It names the two contributions and
the USP (multiple models combined), and tells the reader we also tested
on outside data.

**Abstract.** Problem → what is wrong with existing work → what we did →
the measured numbers (77.95% accuracy, 62.99% BA, AUC 0.950 internally;
37.25% accuracy but best cancer AUC of 0.873 on phone photos).

**I. Introduction.** Why skin cancer matters, what AI has achieved, the
two problems (unfair testing, lab-vs-clinic gap), and our six
contributions.

**II. Literature Review.** It justifies every design choice. The six
research gaps and what we did about them:
- G1 (incomparable evaluation) → de-duplication, lesion-grouped split,
  balanced accuracy with confidence intervals
- G2 (no external validation) → PAD-UFES-20 external test
- G3 (rare cancers) → focal loss, 8 classes, per-class recall
- G4 (skin-tone bias) → per-Fitzpatrick results (DDI planned)
- G5 (weak explainability) → Grad-CAM (next stage)
- G6 (heavy models) → all three models trained on a 6 GB laptop GPU

**III. Datasets.** **Table II** lists 12 datasets and marks the two we
used ("Used") and the others ("Future"). Fig. 1 shows the class
imbalance; Fig. 2 shows why accuracy alone misleads.

**IV. Proposed Methodology.** Subsection A ("Implemented configuration")
says exactly which parts are trained. Table III lists the real settings:
224 px, focal loss, AdamW, 20 epochs, RTX 3060, and 43/49/36 minutes per
model.

**V. Experimental Design.** The test rules: de-duplication (344 groups
merged), lesion-grouped split, test sets used only once, the metrics,
skin-type analysis, and the ablation.

**VI. Experimental Results (new).** Our real numbers:
- **Fig. 4** training curves (learned fast; best epoch kept).
- **Table IV + Fig. 5(a)** internal test: ensemble 77.95% / 62.99% BA,
  better than every single model.
- **Fig. 6** confusion matrix; **Fig. 7** ROC curves.
- **Table VI** ablation, the USP proof.
- **Table VII + Fig. 5(b)** external phone photos: big drop (37.25%), but
  the ensemble is still best at telling cancer from non-cancer (AUC 0.873).
- Skin types: similar for I–II and III–IV; too few V–VI images.

**VII. Comparative Analysis of Algorithms.** The literature results in
four fair panels (Figs. 8–11), the clinic comparison (Fig. 12), and our
result next to the ISIC 2019 winner (63.0% vs 63.6% balanced accuracy,
on different test sets).

**VIII. Discussion.** Why the ensemble helps (the models make different
mistakes), why we don't claim 99%, and honest limitations: one split,
224 px, no segmentation or metadata yet, low phone-photo accuracy, and
too few dark-skin images.

**IX. Conclusion.** The accuracy and metrics, the USP confirmed, and
five future steps.

**References (71).** All real and checked, each with a link.

---

## 3. Reviewer's assessment

**Strengths a reviewer will note**
- Real, honestly measured results with confidence intervals, an
  external test set, and an ablation that directly supports the USP.
- A careful literature comparison grouped by protocol.
- A leakage-free protocol (de-duplication, lesion-grouped split).
- Limitations stated openly, including the external-data drop.

**Weaknesses a reviewer may raise, and how to answer**
- *"Balanced accuracy of 63% is modest."* It is on a strict
  lesion-disjoint test set and matches the ISIC 2019 winner's hidden-test
  score (63.6%). Papers reporting 95%+ usually use leaky random splits.
- *"The ensemble gain is small (1–3 points) and the CIs overlap."* True.
  The gain is consistent across every metric and every combination
  (Table VI). Repeated splits would strengthen it.
- *"External accuracy is low."* This is a finding, not a failure: it
  quantifies the dermoscopy-to-smartphone domain gap.
- *"Segmentation and metadata are described but not evaluated."* The
  paper says so explicitly; they are the next stage.

**To make the paper even stronger (optional, before submission)**
1. Train at 384 px and/or for longer; this usually adds several points.
2. Add the planned baselines (ResNet-50, DenseNet-121, MobileNetV2).
3. Repeat training with 2–3 different random splits and report mean ± std.
4. Fine-tune on part of PAD-UFES-20 to show the domain gap can be reduced.
5. Add a Grad-CAM example figure.

**Other things to do before submission**
- Replace the author placeholders (search `==== AUTHORS` in `sc_body.tex`).
- Check the target conference's page limit. The paper is now 14 pages,
  and many conferences allow 6–8. To shorten, move Tables I, II, VIII
  and IX to a supplement, or drop Figs. 10–12.
- Read the full text of any paper whose number you quote, and confirm
  it matches.

---

## 4. Algorithms and datasets chosen (research summary)

**Recommended algorithm stack (feasible on one 16–24 GB GPU):**
- **Segmentation:** Attention U-Net with an EfficientNet encoder.
- **Classification:** an ensemble of EfficientNetV2-S, ConvNeXt-T and
  Swin-T (all available in the `timm` library with ImageNet weights).
- **Training:** focal loss (γ=2) with balanced sampling; AdamW with a
  cosine learning-rate schedule; 448 px input; 5-fold GroupKFold.
- **Fusion:** a metadata MLP for age, sex and anatomical site.
- **Output:** weighted soft voting with 8× test-time augmentation,
  temperature scaling, and a sensitivity-targeted threshold.
- **Explanation:** Grad-CAM (the `pytorch-grad-cam` package).
- **Optional strong baseline:** a linear probe on frozen DINOv2 features.

**Datasets (all free for academic use):**
- **ISIC 2019:** https://challenge.isic-archive.com/data/
- **ISIC 2020:** https://doi.org/10.34970/2020-ds01 (also on Kaggle)
- **MILK10k:** https://doi.org/10.34970/648456
- **PAD-UFES-20:** https://doi.org/10.17632/zr7vgbcyr2.1
- **DDI:** Stanford AIMI portal (needs a research-use agreement)
- **ISIC 2018 Task 1** (segmentation masks): ISIC challenge data page
- **PH2:** https://www.fc.up.pt/addi/ph2%20database.html

Before splitting, **deduplicate**: ISIC 2019 already contains HAM10000
and BCN20000, and MILK10k shares some images with ISIC.

---

## 5. Keeping plagiarism below 10%

- **Check it yourself.** Run the PDF text through your institution's
  Turnitin or iThenticate. I cannot run those tools, so treat
  "below 10%" as the goal, not a guarantee.
- **Why it should be low:**
  - Every paragraph was written fresh for this paper, and none is
    copied from paper abstracts.
  - Numbers, dataset names and standard terms (e.g., "focal loss",
    "balanced accuracy") are unavoidable and usually count little.
- **Common sources of matches:**
  - The **reference list**. Ask for "exclude bibliography" in the
    Turnitin settings; this is standard.
  - The **equations**. Standard formulas match everywhere, and
    reviewers expect that.
  - **Your old DermaAI paper**, if it was uploaded to Turnitin before.
    The new paper deliberately shares no sentences with it.
- **If a paragraph is flagged,** rewrite it in your own words. Change
  the sentence structure, not just a few words.
- Do **not** use "paraphrasing tools" that swap synonyms. They produce
  unnatural text that reviewers notice, and some checkers flag it.

---

## 6. How to explain the project in one minute (viva / review)

> "Many deep learning papers claim over 95% accuracy for skin cancer,
> but they often use leaky data splits and never test on outside data.
> We reviewed over 40 studies and 12 datasets, and compared more than 11
> algorithm families using published results grouped by fair protocols.
> We found that evaluation choices change results by 10–15 points, more
> than the choice of model. So we trained three different models, two
> CNNs (EfficientNetV2-S, ConvNeXt-T) and one transformer (Swin-T), on
> ISIC 2019 without data leakage, and combined their votes. On 3,596
> unseen images the combined model reached 77.95% accuracy, 62.99%
> balanced accuracy and an AUC of 0.950, and caught 95.4% of cancers,
> better than any single model or any pair. On phone photos from another
> country accuracy dropped to 37%, which shows the real-world gap that
> we will work on next."
