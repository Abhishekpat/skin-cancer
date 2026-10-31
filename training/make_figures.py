"""Step 4: graphs and LaTeX tables for the paper, from the REAL results.

Reads runs/*/history.csv and results/metrics.json and writes to
results/paper_assets/:
    fig_training_curves.pdf      validation balanced accuracy + loss per epoch
    fig_model_comparison.pdf     3 models vs ensemble, internal test
    fig_external_comparison.pdf  3 models vs ensemble, external test
    fig_confusion_ensemble.pdf   ensemble confusion matrix (internal test)
    fig_roc_malignant.pdf        ROC curves, malignant vs benign
    fig_per_class_recall.pdf     per-class recall, best single model vs ensemble
    results_tables.tex           LaTeX tables with all metrics
    results_macros.tex           \\newcommand values for use in the text
"""
from __future__ import annotations

import json

import torch  # noqa: F401  (load torch DLLs before matplotlib on Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from common import (CLASSES, MAL_IDX, MODEL_LABELS, MODELS, RESULTS, RUNS,
                    softmax_np)

OUT = RESULTS / "paper_assets"
COLORS = {"effnetv2_s": "#2166AC", "convnext_t": "#1B9E77",
          "swin_t": "#7570B3", "ensemble_equal": "#E6AB02", "ensemble": "#D95F02"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def training_curves(keys):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k in keys:
        h = pd.read_csv(RUNS / k / "history.csv")
        ax[0].plot(h["epoch"], 100 * h["val_balanced_acc"], "-o", ms=3, color=COLORS[k], label=MODEL_LABELS[k])
        ax[1].plot(h["epoch"], h["train_loss"], "-", color=COLORS[k], label=f"{MODEL_LABELS[k]} (train)")
        ax[1].plot(h["epoch"], h["val_loss"], "--", color=COLORS[k], label=f"{MODEL_LABELS[k]} (val)")
    ax[0].set(xlabel="Epoch", ylabel="Validation balanced accuracy (%)", title="(a) Validation balanced accuracy")
    ax[1].set(xlabel="Epoch", ylabel="Focal loss", title="(b) Training and validation loss")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=6, ncol=1)
    save(fig, "fig_training_curves")


def comparison(metrics, split, name, title):
    systems = [k for k in list(MODELS) + ["ensemble_equal", "ensemble"] if split in metrics["systems"].get(k, {})]
    names = ["accuracy", "balanced_accuracy", "macro_f1", "malignant_sensitivity", "macro_auc"]
    labels = ["Accuracy", "Balanced acc.", "Macro-F1", "Malignant sens.", "Macro AUC"]
    x = np.arange(len(names))
    w = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, k in enumerate(systems):
        m = metrics["systems"][k][split]
        vals = [100 * m[n] for n in names]
        err = np.zeros((2, len(names)))
        for j, n in enumerate(["accuracy", "balanced_accuracy"]):
            lo, hi = m[f"{n}_ci"]
            err[:, j] = [vals[j] - 100 * lo, 100 * hi - vals[j]]
        xs = x + (i - (len(systems) - 1) / 2) * w
        bars = ax.bar(xs, vals, w, color=COLORS[k], label=MODEL_LABELS[k])
        # 95% bootstrap CI only exists for accuracy and balanced accuracy
        ax.errorbar(xs[:2], vals[:2], yerr=err[:, :2], fmt="none", ecolor="black",
                    elinewidth=0.7, capsize=2)
        for j, (b, v) in enumerate(zip(bars, vals)):
            top = v + (err[1, j] if j < 2 else 0)
            ax.text(b.get_x() + b.get_width() / 2, top + 1.0, f"{v:.1f}", ha="center",
                    va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, name)


def confusion(metrics):
    cm = np.array(metrics["systems"]["ensemble"]["internal_test"]["confusion_matrix"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{100*cmn[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Ensemble confusion matrix (% of row)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig_confusion_ensemble")


def roc(metrics, keys):
    y = np.load(next(RUNS / k / "labels_test.npy" for k in keys))
    y_mal = np.isin(y, MAL_IDX).astype(int)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ens = np.load(RESULTS / "ensemble_probs_test.npy")
    curves = {k: softmax_np(np.load(RUNS / k / "logits_test.npy"), metrics["temperatures"][k]) for k in keys}
    curves["ensemble_equal"] = sum(curves[k] for k in keys) / len(keys)
    curves["ensemble"] = ens
    for k, p in curves.items():
        fpr, tpr, _ = roc_curve(y_mal, p[:, MAL_IDX].sum(1))
        auc = metrics["systems"][k]["internal_test"]["malignant_auc"]
        ax.plot(fpr, tpr, color=COLORS[k], lw=1.8 if k.startswith("ensemble") else 1.1,
                label=f"{MODEL_LABELS[k]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="gray", lw=0.8)
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="Malignant vs. benign ROC (internal test)")
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "fig_roc_malignant")


def per_class(metrics, keys):
    best = max(keys, key=lambda k: metrics["systems"][k]["internal_test"]["balanced_accuracy"])
    a = metrics["systems"][best]["internal_test"]["per_class_recall"]
    b = metrics["systems"]["ensemble"]["internal_test"]["per_class_recall"]
    x = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.bar(x - 0.2, [100 * a[c] for c in CLASSES], 0.4, color=COLORS[best], label=f"Best single model ({MODEL_LABELS[best]})")
    ax.bar(x + 0.2, [100 * b[c] for c in CLASSES], 0.4, color=COLORS["ensemble"], label=MODEL_LABELS["ensemble"])
    ax.set_xticks(x, CLASSES)
    ax.set(ylabel="Recall / sensitivity (%)", ylim=(0, 105), title="Per-class recall on the internal test set")
    ax.legend(fontsize=7)
    save(fig, "fig_per_class_recall")


def tex_tables(metrics, info):
    def row(k, split):
        m = metrics["systems"][k][split]
        name = MODEL_LABELS[k]
        if k.startswith("ensemble"):
            name = r"\textbf{" + name + "}"
        return (f"{name} & {100*m['accuracy']:.2f} & {100*m['balanced_accuracy']:.2f} & "
                f"{100*m['macro_f1']:.2f} & {m['macro_auc']:.3f} & {100*m['malignant_sensitivity']:.1f} & "
                f"{100*m['malignant_specificity']:.1f} \\\\")
    lines = []
    for split, cap, lab in [("internal_test", "Internal test set (ISIC 2019, lesion-disjoint)", "tab:res_internal"),
                            ("external_test", "External test set (PAD-UFES-20, smartphone images, 6 shared classes)", "tab:res_external")]:
        keys = [k for k in list(MODELS) + ["ensemble_equal", "ensemble"] if split in metrics["systems"].get(k, {})]
        if not keys:
            continue
        lines += [r"\begin{table}[t]", r"\centering",
                  rf"\caption{{Measured results: {cap}. Accuracy, balanced accuracy (BA), macro-F1 and malignant sensitivity/specificity in \%.}}",
                  rf"\label{{{lab}}}", r"\renewcommand{\arraystretch}{1.15}", r"\footnotesize",
                  r"\begin{tabular}{@{}l c c c c c c@{}}", r"\toprule",
                  r"\textbf{Model} & \textbf{Acc.} & \textbf{BA} & \textbf{F1} & \textbf{AUC} & \textbf{Sens.} & \textbf{Spec.} \\",
                  r"\midrule"]
        lines += [row(k, split) for k in keys]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "results_tables.tex").write_text("\n".join(lines), encoding="utf-8")

    e = metrics["systems"]["ensemble"]
    mac = {"EnsAcc": 100 * e["internal_test"]["accuracy"],
           "EnsBA": 100 * e["internal_test"]["balanced_accuracy"],
           "EnsFone": 100 * e["internal_test"]["macro_f1"],
           "EnsAUC": e["internal_test"]["macro_auc"],
           "EnsSens": 100 * e["internal_test"]["malignant_sensitivity"],
           "EnsSpec": 100 * e["internal_test"]["malignant_specificity"]}
    if "external_test" in e:
        mac.update({"ExtAcc": 100 * e["external_test"]["accuracy"],
                    "ExtBA": 100 * e["external_test"]["balanced_accuracy"],
                    "ExtSens": 100 * e["external_test"]["malignant_sensitivity"]})
    txt = [f"\\newcommand{{\\{k}}}{{{v:.3f}}}" if k == "EnsAUC" else f"\\newcommand{{\\{k}}}{{{v:.2f}}}"
           for k, v in mac.items()]
    (OUT / "results_macros.tex").write_text("\n".join(txt) + "\n", encoding="utf-8")


def main():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    keys = [k for k in MODELS if (RUNS / k / "history.csv").exists()]
    info = {k: json.loads((RUNS / k / "run_info.json").read_text()) for k in keys}
    training_curves(keys)
    comparison(metrics, "internal_test", "fig_model_comparison", "Internal test set (ISIC 2019)")
    if "external_test" in metrics["systems"]["ensemble"]:
        comparison(metrics, "external_test", "fig_external_comparison", "External test set (PAD-UFES-20)")
    confusion(metrics)
    roc(metrics, keys)
    per_class(metrics, keys)
    tex_tables(metrics, info)
    print(f"Wrote figures and tables to {OUT}")


if __name__ == "__main__":
    main()
