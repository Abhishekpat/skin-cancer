"""Shared definitions for the skin cancer ensemble project.

Classes, label maps, dataset, transforms, model factory, focal loss and
evaluation metrics used by prepare_data.py, train.py and ensemble_eval.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
# Folders can be redirected with environment variables (used for smoke tests).
DATA = Path(os.environ.get("SC_DATA", ROOT / "data"))
CACHE = DATA / "cache"          # resized images (shorter side = CACHE_SIZE)
RUNS = Path(os.environ.get("SC_RUNS", ROOT / "runs"))        # checkpoints, histories, predictions
RESULTS = Path(os.environ.get("SC_RESULTS", ROOT / "results"))  # metrics, figures, paper assets
CACHE_SIZE = 256

# ISIC 2019 training classes (the UNK column is empty in the training set).
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = {
    "MEL": "Melanoma", "NV": "Nevus", "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis", "BKL": "Benign keratosis",
    "DF": "Dermatofibroma", "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}
# Lesions that need referral: malignant or pre-malignant (as in the paper).
MALIGNANT = ["MEL", "BCC", "SCC", "AK"]
MAL_IDX = [CLASSES.index(c) for c in MALIGNANT]

# PAD-UFES-20 diagnosis -> ISIC 2019 class (BOD is already merged into SCC).
PAD_MAP = {"BCC": "BCC", "SCC": "SCC", "MEL": "MEL", "ACK": "AK",
           "NEV": "NV", "SEK": "BKL"}
PAD_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]   # 6 shared classes

# Backbones: two CNNs and one transformer (timm names, ImageNet-21k pretrained).
MODELS = {
    "effnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_t": "convnext_tiny.fb_in22k_ft_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
}
MODEL_LABELS = {"effnetv2_s": "EfficientNetV2-S", "convnext_t": "ConvNeXt-T",
                "swin_t": "Swin-T", "ensemble_equal": "Ensemble (equal weights)",
                "ensemble": "Ensemble (learned weights)"}
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(45)], p=0.5),
        T.ColorJitter(0.2, 0.2, 0.15, 0.03),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class LesionDataset(Dataset):
    """Reads cached images listed in a dataframe with columns path, label."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


class ThreadLoader:
    """Mini-batch loader that decodes images in a thread pool.

    Used instead of torch's multi-process DataLoader: on Windows every
    worker process re-imports torch/sklearn/scipy, which exhausts the page
    file on a 16 GB laptop. PIL decoding and torchvision transforms release
    the GIL, so threads give similar throughput without extra processes.
    """

    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False,
                 threads=8, prefetch=3, seed=0):
        self.ds, self.bs, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last
        self.threads, self.prefetch = threads, prefetch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.ds)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        idx = self.rng.permutation(len(self.ds)) if self.shuffle else np.arange(len(self.ds))
        batches = [idx[i:i + self.bs] for i in range(0, len(idx), self.bs)]
        if self.drop_last and batches and len(batches[-1]) < self.bs:
            batches = batches[:-1]
        with ThreadPoolExecutor(self.threads) as ex:
            queue = deque()
            for b in batches[:self.prefetch]:
                queue.append([ex.submit(self.ds.__getitem__, int(i)) for i in b])
            nxt = self.prefetch
            while queue:
                items = [f.result() for f in queue.popleft()]
                if nxt < len(batches):
                    queue.append([ex.submit(self.ds.__getitem__, int(i)) for i in batches[nxt]])
                    nxt += 1
                yield (torch.stack([x for x, _ in items]),
                       torch.tensor([y for _, y in items], dtype=torch.long))


def load_split(name: str) -> pd.DataFrame:
    """name in {train, val, test, external}."""
    df = pd.read_csv(DATA / f"{name}.csv")
    df["path"] = df["path"].map(lambda p: str(DATA / p))
    return df


# ----------------------------------------------------------------------
# Model and loss
# ----------------------------------------------------------------------
def build_model(key: str, pretrained: bool = True) -> nn.Module:
    import timm
    return timm.create_model(MODELS[key], pretrained=pretrained,
                             num_classes=len(CLASSES), drop_path_rate=0.1)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with per-class alpha."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_alpha(labels: np.ndarray) -> torch.Tensor:
    """Inverse square-root frequency weights, normalized to mean 1."""
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def restrict(probs: np.ndarray, classes: list[str]) -> np.ndarray:
    """Keep only the given classes and renormalize (for PAD-UFES-20)."""
    idx = [CLASSES.index(c) for c in classes]
    p = probs[:, idx]
    return p / p.sum(axis=1, keepdims=True)


def ece_score(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def classification_metrics(probs: np.ndarray, y: np.ndarray,
                           class_list: list[str]) -> dict:
    """probs: (N, len(class_list)); y: indices into class_list."""
    pred = probs.argmax(1)
    labels = list(range(len(class_list)))
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "ece": ece_score(probs, y),
    }
    present = [c for c in labels if (y == c).any()]
    try:
        out["macro_auc"] = roc_auc_score(
            np.eye(len(class_list))[y][:, present], probs[:, present],
            average="macro")
    except ValueError:
        out["macro_auc"] = float("nan")
    cm = confusion_matrix(y, pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(cm) / cm.sum(1)
    out["per_class_recall"] = {c: float(r) for c, r in zip(class_list, recall)}
    out["confusion_matrix"] = cm.tolist()
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in out.items()}


def malignant_metrics(probs: np.ndarray, y: np.ndarray, class_list: list[str],
                      threshold: float) -> dict:
    """Binary referral task: malignant/pre-malignant vs benign."""
    mal = [class_list.index(c) for c in MALIGNANT if c in class_list]
    p_mal = probs[:, mal].sum(1)
    y_mal = np.isin(y, mal).astype(int)
    pred = (p_mal >= threshold).astype(int)
    tp = int(((pred == 1) & (y_mal == 1)).sum())
    tn = int(((pred == 0) & (y_mal == 0)).sum())
    fp = int(((pred == 1) & (y_mal == 0)).sum())
    fn = int(((pred == 0) & (y_mal == 1)).sum())
    return {
        "malignant_auc": float(roc_auc_score(y_mal, p_mal)) if 0 < y_mal.sum() < len(y_mal) else float("nan"),
        "malignant_sensitivity": tp / max(tp + fn, 1),
        "malignant_specificity": tn / max(tn + fp, 1),
        "threshold": float(threshold),
    }


def bootstrap_ci(probs: np.ndarray, y: np.ndarray, n: int = 1000,
                 seed: int = 0) -> dict:
    """95% bootstrap confidence intervals for accuracy and balanced accuracy."""
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    accs, bas = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[i], pred[i]))
        bas.append(balanced_accuracy_score(y[i], pred[i]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_ci": q(accs), "balanced_accuracy_ci": q(bas)}


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
