"""Step 5 (optional): precision, confusion matrices and example images.

Uses the saved predictions in runs/ (no retraining) and writes to
results/explain/:
    eval_details.json            per-model precision/recall/F1, per-class table,
                                 confusion matrices, binary malignant counts
    fig_cm_counts.pdf/.png       8x8 confusion matrix with image counts
    fig_cm_binary.pdf/.png       2x2 malignant-vs-benign confusion matrix
    fig_prf_per_class.pdf/.png   precision / recall / F1 per class
    fig_examples_correct.png     one correctly classified test image per class
    fig_examples_usp.png         single models disagree, ensemble is right
    fig_examples_errors.png      typical ensemble mistakes
    fig_examples_external.png    PAD-UFES-20 smartphone examples
"""
from __future__ import annotations

import json

import torch  # noqa: F401  (load torch DLLs before matplotlib on Windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import (CLASSES, MAL_IDX, MALIGNANT, MODEL_LABELS, PAD_CLASSES,
                    RESULTS, RUNS, load_split, restrict, save_json, softmax_np)

OUT = RESULTS / "explain"
KEYS = ["effnetv2_s", "convnext_t", "swin_t"]
SHORT = {"effnetv2_s": "EffNetV2-S", "convnext_t": "ConvNeXt-T", "swin_t": "Swin-T"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "savefig.bbox": "tight", "savefig.dpi": 300})


def save(fig, name, pdf=True):
    OUT.mkdir(parents=True, exist_ok=True)
    if pdf:
        fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def main():
    m = json.loads((RESULTS / "metrics.json").read_text())
    temps = m["temperatures"]
    test = load_split("test").reset_index(drop=True)
    ext = load_split("external").reset_index(drop=True)
    y = np.load(RUNS / "effnetv2_s" / "labels_test.npy")
    ye = np.load(RUNS / "effnetv2_s" / "labels_external.npy")
    assert (test["label"].values == y).all() and (ext["label"].values == ye).all()

    P = {k: softmax_np(np.load(RUNS / k / "logits_test.npy"), temps[k]) for k in KEYS}
    P["ensemble"] = sum(P[k] for k in KEYS) / 3           # equal-weight ensemble
    # same order as ensemble_eval.py: average the 8-class probabilities, then
    # restrict to the 6 PAD-UFES-20 classes and renormalize
    PE8 = {k: softmax_np(np.load(RUNS / k / "logits_external.npy"), temps[k]) for k in KEYS}
    PE8["ensemble"] = sum(PE8[k] for k in KEYS) / 3
    PE = {k: restrict(v, PAD_CLASSES) for k, v in PE8.items()}
    ye6 = np.array([PAD_CLASSES.index(CLASSES[i]) for i in ye])
    thr = m["systems"]["ensemble_equal"]["validation"]["threshold"]

    details = {"threshold_equal_ensemble": thr, "models": {}}
    labels = list(range(len(CLASSES)))
    for k, p in P.items():
        pred = p.argmax(1)
        pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        wp, wr, wf, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y, pred, labels=labels)
        p_mal = p[:, MAL_IDX].sum(1)
        y_mal = np.isin(y, MAL_IDX)
        pred_mal = p_mal >= thr
        details["models"][k] = {
            "accuracy": float((pred == y).mean()),
            "correct": int((pred == y).sum()), "total": int(len(y)),
            "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()),
            "macro_f1": float(f1.mean()), "weighted_precision": float(wp),
            "weighted_recall": float(wr), "weighted_f1": float(wf),
            "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                              "f1": float(f1[i]), "support": int(sup[i])}
                          for i, c in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
            "binary": {"TP": int((pred_mal & y_mal).sum()), "FN": int((~pred_mal & y_mal).sum()),
                       "FP": int((pred_mal & ~y_mal).sum()), "TN": int((~pred_mal & ~y_mal).sum())},
        }
        pe = PE[k].argmax(1)
        details["models"][k]["external_accuracy"] = float((pe == ye6).mean())
        epr, erc, ef1, _ = precision_recall_fscore_support(ye6, pe, labels=list(range(6)), zero_division=0)
        details["models"][k]["external_macro_precision"] = float(epr.mean())
    save_json(details, OUT / "eval_details.json")

    # ---------------- confusion matrix with counts (equal ensemble) ----------
    cm = np.array(details["models"]["ensemble"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                    color="white" if frac > 0.55 else "black", fontweight="bold" if i == j else None)
    ax.set_xticks(range(8), CLASSES, rotation=45)
    ax.set_yticks(range(8), [f"{c} ({cm[i].sum()})" for i, c in enumerate(CLASSES)])
    ax.set(xlabel="Predicted class", ylabel="True class (number of test images)",
           title="Confusion matrix, ensemble (equal weights)\ninternal test set, 3,596 images")
    save(fig, "fig_cm_counts")

    # ---------------- binary confusion matrix --------------------------------
    b = details["models"]["ensemble"]["binary"]
    mat = np.array([[b["TP"], b["FN"]], [b["FP"], b["TN"]]])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.imshow([[1, 0.3], [0.3, 1]], cmap="Greens", vmin=0, vmax=1.4)
    names = [["TP\n(cancer caught)", "FN\n(cancer missed)"], ["FP\n(false alarm)", "TN\n(correctly cleared)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=8)
    ax.set_xticks([0, 1], ["Predicted:\nrefer (malignant)", "Predicted:\nbenign"])
    ax.set_yticks([0, 1], ["True:\nmalignant", "True:\nbenign"])
    ax.set_title("Malignant vs. benign (ensemble)")
    save(fig, "fig_cm_binary")

    # ---------------- per-class precision / recall / F1 ----------------------
    pc = details["models"]["ensemble"]["per_class"]
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for off, key, col, lab in [(-0.27, "precision", "#2166AC", "Precision"),
                               (0, "recall", "#D95F02", "Recall (sensitivity)"),
                               (0.27, "f1", "#1B9E77", "F1-score")]:
        vals = [100 * pc[c][key] for c in CLASSES]
        bars = ax.bar(x + off, vals, 0.27, color=col, label=lab)
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_xticks(x, [f"{c}\n(n={pc[c]['support']})" for c in CLASSES])
    ax.set(ylabel="%", ylim=(0, 110), title="Per-class precision, recall and F1 (ensemble, internal test)")
    ax.legend(ncol=3, fontsize=7, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_prf_per_class")

    # ---------------- example images ------------------------------------------
    pred = {k: P[k].argmax(1) for k in P}
    conf = P["ensemble"].max(1)

    def tile(ax, path, title, color):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(2.5)
        ax.set_title(title, fontsize=7.2)

    # (a) one correct, confident example per class
    rows = []
    for c in range(8):
        idx = np.where((y == c) & (pred["ensemble"] == c))[0]
        if len(idx):
            rows.append(idx[np.argsort(-conf[idx])[min(3, len(idx) - 1)]])
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.3))
    for ax, i in zip(axes.flat, rows):
        tile(ax, test.path[i], f"True: {CLASSES[y[i]]}\nPredicted: {CLASSES[pred['ensemble'][i]]} "
             f"({100*conf[i]:.0f}%)", "#1B9E77")
    save(fig, "fig_examples_correct", pdf=False)
    ex_correct = [{"index": int(i), "true": CLASSES[y[i]], "pred": CLASSES[pred["ensemble"][i]],
                   "confidence": float(conf[i])} for i in rows]

    # (b) USP: at least two single models wrong, ensemble right
    wrong_count = sum((pred[k] != y).astype(int) for k in KEYS)
    cand = np.where((pred["ensemble"] == y) & (wrong_count >= 1))[0]
    # prefer different true classes and cancers
    chosen, seen = [], set()
    order = cand[np.argsort(-(wrong_count[cand] * 10 + np.isin(y[cand], MAL_IDX) * 2 + 5 * conf[cand]))]
    for i in order:
        if y[i] not in seen:
            chosen.append(i); seen.add(y[i])
        if len(chosen) == 4:
            break
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9))
    usp = []
    for ax, i in zip(axes, chosen):
        lines = [f"True: {CLASSES[y[i]]}"]
        for k in KEYS:
            mark = "ok" if pred[k][i] == y[i] else "X"
            lines.append(f"{SHORT[k]}: {CLASSES[pred[k][i]]} [{mark}]")
        lines.append(f"Ensemble: {CLASSES[pred['ensemble'][i]]} ({100*conf[i]:.0f}%) [ok]")
        tile(ax, test.path[i], "\n".join(lines), "#D95F02")
        usp.append({"index": int(i), "true": CLASSES[y[i]],
                    **{k: CLASSES[pred[k][i]] for k in KEYS},
                    "ensemble": CLASSES[pred["ensemble"][i]], "confidence": float(conf[i])})
    save(fig, "fig_examples_usp", pdf=False)
    details["usp_cases_total"] = int(len(cand))
    details["ensemble_wrong_but_some_single_right"] = int(((pred["ensemble"] != y) & (wrong_count < 3)).sum())

    # (c) typical mistakes: MEL->NV, AK->BCC, SCC->BCC, DF->NV
    errs = []
    for t, pclass in [("MEL", "NV"), ("AK", "BCC"), ("SCC", "BCC"), ("DF", "NV")]:
        idx = np.where((y == CLASSES.index(t)) & (pred["ensemble"] == CLASSES.index(pclass)))[0]
        if len(idx):
            errs.append(idx[np.argsort(-conf[idx])[len(idx) // 2]])
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, i in zip(axes, errs):
        tile(ax, test.path[i], f"True: {CLASSES[y[i]]}\nPredicted: {CLASSES[pred['ensemble'][i]]} "
             f"({100*conf[i]:.0f}%)", "#B2182B")
    save(fig, "fig_examples_errors", pdf=False)

    # (d) external smartphone examples: 2 correct, 2 wrong
    pe = PE["ensemble"].argmax(1); ce = PE["ensemble"].max(1)
    ext_pick = []
    for c in ["BCC", "NV"]:
        idx = np.where((ye6 == PAD_CLASSES.index(c)) & (pe == ye6))[0]
        ext_pick.append((idx[np.argsort(-ce[idx])[2]], True))
    for c in ["AK", "SCC"]:
        idx = np.where((ye6 == PAD_CLASSES.index(c)) & (pe != ye6))[0]
        ext_pick.append((idx[np.argsort(-ce[idx])[len(idx) // 2]], False))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    for ax, (i, ok) in zip(axes, ext_pick):
        tile(ax, ext.path[i], f"True: {PAD_CLASSES[ye6[i]]}\nPredicted: {PAD_CLASSES[pe[i]]} "
             f"({100*ce[i]:.0f}%)", "#1B9E77" if ok else "#B2182B")
    save(fig, "fig_examples_external", pdf=False)

    details["examples"] = {"correct": ex_correct, "usp": usp,
                           "errors": [{"index": int(i), "true": CLASSES[y[i]],
                                       "pred": CLASSES[pred['ensemble'][i]]} for i in errs]}
    save_json(details, OUT / "eval_details.json")

    for k in list(KEYS) + ["ensemble"]:
        d = details["models"][k]
        print(f"{k:11s} acc {100*d['accuracy']:.2f} ({d['correct']}/{d['total']})  "
              f"macroP {100*d['macro_precision']:.2f}  macroR {100*d['macro_recall']:.2f}  "
              f"macroF1 {100*d['macro_f1']:.2f}  wP {100*d['weighted_precision']:.2f}  "
              f"ext acc {100*d['external_accuracy']:.2f} ext macroP {100*d['external_macro_precision']:.2f}  bin {d['binary']}")
    print("per-class ensemble:")
    for c, v in details["models"]["ensemble"]["per_class"].items():
        print(f"  {c:5s} P {100*v['precision']:.1f} R {100*v['recall']:.1f} F1 {100*v['f1']:.1f} n {v['support']}")
    print("USP cases (ensemble right, >=1 single wrong):", details["usp_cases_total"])
    print("ensemble wrong but >=1 single right:", details["ensemble_wrong_but_some_single_right"])
    print("examples:", json.dumps(details["examples"], indent=0)[:1500])


if __name__ == "__main__":
    main()
