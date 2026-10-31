"""Step 1: download and prepare the two datasets.

  * ISIC 2019  -> train / val / internal test (lesion-grouped, stratified)
  * PAD-UFES-20 -> external test (smartphone images, never used in training)

Images are resized directly from the zip files (shorter side = 256 px) into
data/cache, so the 10 GB zip never has to be extracted.

Usage:
    python prepare_data.py               # download + prepare everything
    python prepare_data.py --delete-zips # also delete the zips afterwards
"""
from __future__ import annotations

import torch  # noqa: F401  (load torch DLLs before sklearn/PIL on Windows)
import argparse
import io
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, CACHE_SIZE, CLASSES, DATA, PAD_MAP, save_json

RAW = DATA / "raw"
ISIC_BASE = "https://isic-challenge-data.s3.amazonaws.com/2019/"
ISIC_FILES = ["ISIC_2019_Training_GroundTruth.csv",
              "ISIC_2019_Training_Metadata.csv",
              "ISIC_2019_Training_Input.zip"]
PAD_URL = "https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1"
PAD_ZIP = "PAD-UFES-20.zip"
SEED = 42


# Expected sizes of the large zips (bytes), used to skip finished downloads.
EXPECTED = {"ISIC_2019_Training_Input.zip": 9_771_618_190, PAD_ZIP: 3_593_344_681}


def download(url: str, dest: Path):
    """Resumable download with curl (bundled with Windows 10/11)."""
    if dest.exists() and dest.stat().st_size > 0:
        want = EXPECTED.get(dest.name)
        if want is None or dest.stat().st_size == want:
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name} ...")
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "10",
                    "-o", str(dest), url], check=True)


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------
def resize_bytes(data: bytes, out: Path) -> int:
    """Resize so the shorter side is CACHE_SIZE; save JPEG; return dHash."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = CACHE_SIZE / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=95)
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _isic_worker(args):
    zip_path, entries = args
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for image_id, entry in entries:
            dest = CACHE / "isic2019" / f"{image_id}.jpg"
            out[image_id] = resize_bytes(z.read(entry), dest)
    return out


# ----------------------------------------------------------------------
# ISIC 2019
# ----------------------------------------------------------------------
def prepare_isic(workers: int) -> pd.DataFrame:
    for f in ISIC_FILES:
        download(ISIC_BASE + f, RAW / f)
    gt = pd.read_csv(RAW / ISIC_FILES[0])
    meta = pd.read_csv(RAW / ISIC_FILES[1])
    gt["label"] = gt[CLASSES].values.argmax(1)
    df = gt[["image", "label"]].merge(meta, on="image", how="left")

    zip_path = RAW / ISIC_FILES[2]
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
    missing = [i for i in df["image"] if i not in names]
    if missing:
        raise SystemExit(f"{len(missing)} ISIC images missing from zip, e.g. {missing[:3]}")

    todo = [(i, names[i]) for i in df["image"]]
    chunks = [todo[k::workers * 4] for k in range(workers * 4)]
    hashes = {}
    print(f"Resizing {len(todo)} ISIC 2019 images with {workers} workers ...")
    with ThreadPoolExecutor(workers) as ex:  # PIL releases the GIL while decoding/resizing
        for n, part in enumerate(ex.map(_isic_worker, [(zip_path, c) for c in chunks]), 1):
            hashes.update(part)
            print(f"  chunk {n}/{len(chunks)}", end="\r")
    print()
    df["dhash"] = df["image"].map(hashes)
    df["path"] = df["image"].map(lambda i: f"cache/isic2019/{i}.jpg")
    df["source"] = "ISIC2019"
    return df


def lesion_groups(df: pd.DataFrame) -> np.ndarray:
    """Group images by lesion_id and merge exact-duplicate image hashes."""
    key = df["lesion_id"].fillna(df["image"]).tolist()
    parent = {k: k for k in key}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    first_by_hash = {}
    n_dup = 0
    for k, h in zip(key, df["dhash"]):
        if h in first_by_hash:
            ra, rb = find(first_by_hash[h]), find(k)
            if ra != rb:
                parent[rb] = ra
                n_dup += 1
        else:
            first_by_hash[h] = k
    print(f"Merged {n_dup} groups that shared a near-identical image hash.")
    roots = [find(k) for k in key]
    codes, _ = pd.factorize(pd.Series(roots))
    return codes


def split_isic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = lesion_groups(df)
    df["split"] = "train"
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    _, test_idx = next(sgkf.split(df, df["label"], df["group"]))
    df.loc[df.index[test_idx], "split"] = "test"
    rest = df[df["split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
    _, val_idx = next(sgkf.split(rest, rest["label"], rest["group"]))
    df.loc[rest.index[val_idx], "split"] = "val"
    # sanity: no lesion group may appear in two splits
    assert df.groupby("group")["split"].nunique().max() == 1
    return df


# ----------------------------------------------------------------------
# PAD-UFES-20 (external test)
# ----------------------------------------------------------------------
def _walk_zip(z: zipfile.ZipFile, wanted: set, found: dict, tmp: Path):
    for info in z.infolist():
        name = info.filename
        base = Path(name).name
        if base.lower().endswith(".zip"):
            nested = tmp / base
            with z.open(info) as src, open(nested, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with zipfile.ZipFile(nested) as zz:
                _walk_zip(zz, wanted, found, tmp)
            nested.unlink()
        elif base in wanted and base not in found:
            found[base] = resize_bytes(z.read(info), CACHE / "pad_ufes20" / (Path(base).stem + ".jpg"))
        elif base == "metadata.csv" and "metadata.csv" not in found:
            found["metadata.csv"] = z.read(info)


def prepare_pad() -> pd.DataFrame:
    zip_path = RAW / PAD_ZIP
    download(PAD_URL, zip_path)
    found = {}
    with tempfile.TemporaryDirectory(dir=RAW) as tmp, zipfile.ZipFile(zip_path) as z:
        # first pass: metadata
        for info in z.infolist():
            if Path(info.filename).name == "metadata.csv":
                found["metadata.csv"] = z.read(info)
                break
        meta = pd.read_csv(io.BytesIO(found["metadata.csv"]))
        print(f"Resizing {len(meta)} PAD-UFES-20 images (nested zips) ...")
        _walk_zip(z, set(meta["img_id"]), found, Path(tmp))
    meta = meta[meta["img_id"].isin(found)].copy()
    print(f"  found {len(meta)} PAD-UFES-20 images")
    meta["label"] = meta["diagnostic"].map(PAD_MAP).map(CLASSES.index)
    meta["path"] = meta["img_id"].map(lambda i: f"cache/pad_ufes20/{Path(i).stem}.jpg")
    meta["source"] = "PAD-UFES-20"
    meta["split"] = "external"
    return meta


# ----------------------------------------------------------------------
def summarize(isic: pd.DataFrame, pad: pd.DataFrame):
    summary = {}
    for name, d in [("train", isic[isic.split == "train"]),
                    ("val", isic[isic.split == "val"]),
                    ("test", isic[isic.split == "test"]),
                    ("external", pad)]:
        counts = d["label"].value_counts().reindex(range(len(CLASSES)), fill_value=0)
        summary[name] = {"n": int(len(d)),
                         "classes": {c: int(counts[i]) for i, c in enumerate(CLASSES)}}
    save_json(summary, DATA / "dataset_summary.json")
    for k, v in summary.items():
        print(f"{k:9s} n={v['n']:6d}  " + " ".join(f"{c}:{n}" for c, n in v["classes"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delete-zips", action="store_true")
    ap.add_argument("--skip-pad", action="store_true")
    args = ap.parse_args()

    isic = split_isic(prepare_isic(args.workers))
    cols = ["path", "label", "group", "image", "lesion_id", "age_approx",
            "sex", "anatom_site_general", "source"]
    for s in ["train", "val", "test"]:
        isic.loc[isic.split == s, cols].to_csv(DATA / f"{s}.csv", index=False)
    if args.delete_zips:
        (RAW / ISIC_FILES[2]).unlink(missing_ok=True)

    if args.skip_pad:
        pad = pd.DataFrame(columns=["label"])
    else:
        pad = prepare_pad()
        keep = [c for c in ["path", "label", "img_id", "patient_id", "diagnostic",
                            "fitspatrick", "biopsed", "age", "gender", "region",
                            "source"] if c in pad.columns]
        pad[keep].to_csv(DATA / "external.csv", index=False)
        if args.delete_zips:
            (RAW / PAD_ZIP).unlink(missing_ok=True)
    summarize(isic, pad)


if __name__ == "__main__":
    main()
