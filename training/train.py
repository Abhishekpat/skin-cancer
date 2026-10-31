"""Step 2: fine-tune one backbone and save its predictions.

    python train.py --model effnetv2_s
    python train.py --model convnext_t
    python train.py --model swin_t

Saves to runs/<model>/:
    best.pt          best checkpoint (highest validation balanced accuracy)
    history.csv      per-epoch training curves (for the paper's graphs)
    logits_{val,test,external}.npy   test-time-augmented logits
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from common import (CLASSES, MODELS, RUNS, FocalLoss, LesionDataset, ThreadLoader,
                    build_model, class_alpha, eval_transform, load_split,
                    save_json, train_transform)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate")
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6, help="image decoding threads")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    return ap.parse_args()


def make_loader(df, transform, bs, shuffle, workers, seed=0):
    """workers = number of decoding threads (no extra processes; see ThreadLoader)."""
    return ThreadLoader(LesionDataset(df, transform), bs, shuffle=shuffle,
                        drop_last=shuffle, threads=max(workers, 1), seed=seed)


@torch.no_grad()
def predict(model, df, bs, workers, device, tta=True):
    """Average logits over 4 flips (test-time augmentation)."""
    model.eval()
    loader = make_loader(df, eval_transform(), bs, False, workers)
    outs = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)] if tta else [x]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = torch.stack([model(v).float() for v in views]).mean(0)
        outs.append(logits.cpu().numpy())
    return np.concatenate(outs)


def param_groups(model, lr, head_mult, wd):
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    decay, no_decay, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        elif p.ndim <= 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
            {"params": head_params, "lr": lr * head_mult, "weight_decay": wd}]


def keep_awake():
    """Stop Windows from sleeping while training runs (released on exit)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def main():
    args = get_args()
    keep_awake()
    # One intra-op CPU thread: the GPU does the maths, and extra CPU threads
    # only compete with the image-decoding threads (measured: 63 -> 221 img/s).
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = RUNS / args.model
    out.mkdir(parents=True, exist_ok=True)

    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    try:
        ex = load_split("external")
    except FileNotFoundError:
        ex = None
    if args.limit:
        tr, va, te = (d.groupby("label", group_keys=False).head(max(args.limit // len(CLASSES), 2))
                      for d in (tr, va, te))
        ex = ex.head(args.limit) if ex is not None else None
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  external {0 if ex is None else len(ex)}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    model = model.to(memory_format=torch.channels_last)
    criterion = FocalLoss(class_alpha(tr["label"].values), args.gamma).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr_mult, args.weight_decay))
    base_lrs = [g["lr"] for g in opt.param_groups]

    train_loader = make_loader(tr, train_transform(), args.batch_size, True, args.workers, args.seed)
    steps_per_epoch = math.ceil(len(train_loader) / args.accum)
    total = steps_per_epoch * args.epochs
    warmup = steps_per_epoch  # one epoch of linear warm-up, then cosine decay

    def set_lr(step):
        f = step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    history, best_ba, bad, step = [], -1.0, 0, 0
    y_val = va["label"].values
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = criterion(logits, y)
            (loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                set_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        val_logits = predict(model, va, args.batch_size, args.workers, device, tta=False)
        with torch.no_grad():
            val_loss = criterion(torch.tensor(val_logits, device=device),
                                 torch.tensor(y_val, device=device)).item()
        val_pred = val_logits.argmax(1)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": float((val_pred == y_val).mean()),
            "val_balanced_acc": balanced_accuracy_score(y_val, val_pred),
            "val_macro_f1": f1_score(y_val, val_pred, average="macro"),
            "lr": opt.param_groups[0]["lr"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"[{args.model}] epoch {epoch:2d}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"train_acc {row['train_acc']:.3f}  val_acc {row['val_acc']:.3f}  "
              f"val_BA {row['val_balanced_acc']:.3f}  ({row['seconds']:.0f}s)", flush=True)
        if row["val_balanced_acc"] > best_ba:
            best_ba, bad = row["val_balanced_acc"], 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for name, df in [("val", va), ("test", te), ("external", ex)]:
        if df is not None and len(df):
            np.save(out / f"logits_{name}.npy", predict(model, df, args.batch_size, args.workers, device))
            np.save(out / f"labels_{name}.npy", df["label"].values)
    save_json({"model": args.model, "timm_name": MODELS[args.model], "best_val_balanced_acc": best_ba,
               "epochs_run": len(history), "args": vars(args),
               "train_minutes": sum(h["seconds"] for h in history) / 60,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
              out / "run_info.json")
    print(f"Done. Best validation balanced accuracy: {best_ba:.4f}")


if __name__ == "__main__":
    main()
