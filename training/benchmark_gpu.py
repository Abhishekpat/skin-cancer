"""Measure training throughput (images/second) of each backbone on this GPU.

    python benchmark_gpu.py --batch-size 32
"""
import argparse
import time

import torch

from common import CLASSES, IMG_SIZE, MODELS, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--steps", type=int, default=20)
args = ap.parse_args()
dev = torch.device("cuda")
for key in MODELS:
    model = build_model(key).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, len(CLASSES), (args.batch_size,), device=dev)
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(x).float(), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    ips = args.steps * args.batch_size / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 2**30
    print(f"{key:12s} {ips:6.1f} img/s   peak GPU memory {mem:.2f} GB", flush=True)
    del model, opt
    torch.cuda.empty_cache()
