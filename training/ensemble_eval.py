"""Step 3: combine the three models and evaluate everything.

  1. Temperature-scale each model on the validation set.
  2. Search ensemble weights (simplex grid, step 0.05) that maximise
     validation balanced accuracy  -> weighted soft voting.
  3. Pick the referral threshold that gives >= 95% malignant sensitivity
     on the validation set.
  4. Report every metric for each single model and for the ensemble on the
     internal test set (ISIC 2019) and the external test set (PAD-UFES-20).

Writes results/metrics.json and results/per_class_*.csv.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import (CLASSES, MAL_IDX, MODELS, PAD_CLASSES, RESULTS, RUNS,
                    bootstrap_ci, classification_metrics, malignant_metrics,
                    restrict, save_json, softmax_np)

KEYS = list(MODELS)


def load(key, split):
    p = RUNS / key / f"logits_{split}.npy"
    return np.load(p) if p.exists() else None


def labels(split):
    for k in KEYS:
        p = RUNS / k / f"labels_{split}.npy"
        if p.exists():
            return np.load(p)
    return None


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.3), np.log(5.0), 120))
    nll = [-np.log(softmax_np(logits, t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def bal_acc(probs, y):
    pred = probs.argmax(1)
    return np.mean([(pred[y == c] == c).mean() for c in np.unique(y)])


def search_weights(probs_list, y, step=0.05):
    n = len(probs_list)
    best = (-1, None)
    ticks = np.round(np.arange(0, 1 + 1e-9, step), 4)
    for w in itertools.product(ticks, repeat=n - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array(list(w) + [1 - sum(w)])
        p = sum(wi * pi for wi, pi in zip(w, probs_list))
        score = bal_acc(p, y)
        if score > best[0] + 1e-9:
            best = (score, w)
    return best[1], best[0]


def referral_threshold(probs, y, target=0.95):
    p_mal = probs[:, MAL_IDX].sum(1)
    y_mal = np.isin(y, MAL_IDX)
    # largest threshold that still reaches the target sensitivity
    cands = np.sort(np.unique(p_mal[y_mal]))[::-1]
    for t in cands:
        if (p_mal[y_mal] >= t).mean() >= target:
            return float(t)
    return 0.0


def evaluate(probs, y, threshold, external=False):
    if external:
        p = restrict(probs, PAD_CLASSES)
        yy = np.array([PAD_CLASSES.index(CLASSES[i]) for i in y])
        cl = PAD_CLASSES
    else:
        p, yy, cl = probs, y, CLASSES
    m = classification_metrics(p, yy, cl)
    m.update(malignant_metrics(p, yy, cl, threshold))
    m.update(bootstrap_ci(p, yy))
    m["n"] = int(len(yy))
    return m


def main():
    available = [k for k in KEYS if load(k, "val") is not None]
    if len(available) < len(KEYS):
        print(f"Warning: only {available} have predictions; ensemble uses these.")
    y_val, y_test, y_ext = labels("val"), labels("test"), labels("external")

    temps, probs = {}, {}
    for k in available:
        temps[k] = fit_temperature(load(k, "val"), y_val)
        probs[k] = {s: softmax_np(load(k, s), temps[k]) for s in ["val", "test", "external"]
                    if load(k, s) is not None}

    weights, val_ba = search_weights([probs[k]["val"] for k in available], y_val)
    ens = {s: sum(w * probs[k][s] for w, k in zip(weights, available))
           for s in probs[available[0]]}
    equal = {s: sum(probs[k][s] for k in available) / len(available) for s in ens}

    systems = {k: probs[k] for k in available}
    systems["ensemble_equal"] = equal
    systems["ensemble"] = ens

    results = {"weights": dict(zip(available, map(float, weights))),
               "temperatures": temps, "val_balanced_accuracy_of_weights": float(val_ba),
               "systems": {}}
    for name, pr in systems.items():
        thr = referral_threshold(pr["val"], y_val)
        entry = {"validation": evaluate(pr["val"], y_val, thr),
                 "internal_test": evaluate(pr["test"], y_test, thr)}
        if "external" in pr and y_ext is not None:
            entry["external_test"] = evaluate(pr["external"], y_ext, thr, external=True)
        results["systems"][name] = entry

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(results, RESULTS / "metrics.json")
    np.save(RESULTS / "ensemble_probs_test.npy", ens["test"])
    if "external" in ens:
        np.save(RESULTS / "ensemble_probs_external.npy", ens["external"])

    for split in ["internal_test", "external_test"]:
        rows = []
        for name, e in results["systems"].items():
            if split in e:
                rows.append({"system": name, **e[split]["per_class_recall"]})
        if rows:
            pd.DataFrame(rows).to_csv(RESULTS / f"per_class_recall_{split}.csv", index=False)

    print(f"Ensemble weights: {results['weights']}")
    hdr = f"{'system':16s} {'split':14s} {'Acc':>6s} {'BA':>6s} {'F1':>6s} {'AUC':>6s} {'Sens':>6s} {'Spec':>6s}"
    print(hdr)
    for name, e in results["systems"].items():
        for split in ["internal_test", "external_test"]:
            if split in e:
                m = e[split]
                print(f"{name:16s} {split:14s} {100*m['accuracy']:6.2f} {100*m['balanced_accuracy']:6.2f} "
                      f"{100*m['macro_f1']:6.2f} {m['macro_auc']:6.3f} "
                      f"{100*m['malignant_sensitivity']:6.2f} {100*m['malignant_specificity']:6.2f}")


if __name__ == "__main__":
    main()
