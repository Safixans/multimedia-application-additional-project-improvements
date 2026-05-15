"""
Benchmark a single training epoch end-to-end.

Reports wall-clock time, average step time, and peak GPU memory.
Use this to A/B baseline vs. optimized variants WITHOUT running full convergence.

Example:
    python scripts/benchmark.py \
        --datasets LUNG --feature_encoder uni_v1_official \
        --source_dataroot ~/STFlow/dataset/ \
        --embed_dataroot ~/STFlow/dataset/embed_dataroot \
        --epochs 1 --num_workers 4 --amp bf16 --use_compile
"""
import argparse
import json
import os
import time

import torch

from stflow.utils import set_random_seed
from stflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher
from stflow.data.normalize_utils import get_normalize_method
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--datasets', nargs='+', default=["LUNG"])
    p.add_argument('--source_dataroot', required=True)
    p.add_argument('--embed_dataroot', required=True)
    p.add_argument('--gene_list', type=str, default='var_50genes.json')
    p.add_argument('--feature_encoder', type=str, default='uni_v1_official')
    p.add_argument('--normalize_method', type=str, default='log1p')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--prefetch_factor', type=int, default=4)
    p.add_argument('--sample_times', type=int, default=10)
    p.add_argument('--patch_distribution', type=str, default='uniform')

    p.add_argument('--amp', type=str, default='off', choices=['off', 'bf16', 'fp16'])
    p.add_argument('--use_compile', action='store_true')
    p.add_argument('--compile_mode', type=str, default='default')

    p.add_argument('--n_genes', type=int, default=50)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--pairwise_hidden_dim', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--attn_dropout', type=float, default=0.2)
    p.add_argument('--n_neighbors', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--norm', type=str, default='layer')
    p.add_argument('--activation', type=str, default='swiglu')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--clip_norm', type=float, default=1.)

    p.add_argument('--prior_sampler', type=str, default='zinb')
    p.add_argument('--zinb_logits', type=float, default=0.1)
    p.add_argument('--zinb_total_count', type=float, default=1)
    p.add_argument('--zinb_zi_logits', type=float, default=0.)

    p.add_argument('--label', type=str, default='unnamed', help='Tag for the output report.')
    p.add_argument('--out_json', type=str, default=None)
    return p.parse_args()


def build_loader(args, dataset_name):
    normalize_method = get_normalize_method(args.normalize_method)
    split_dir = os.path.join(args.source_dataroot, dataset_name, 'splits')
    import pandas as pd
    train_df = pd.read_csv(os.path.join(split_dir, 'train_0.csv'))
    sample_ids = train_df['sample_id'].tolist()

    paths = [
        HESTDatasetPath(
            name=sid,
            h5_path=os.path.join(args.embed_dataroot, dataset_name, args.feature_encoder, f"fp32/{sid}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, dataset_name, f"adata/{sid}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, dataset_name, args.gene_list),
        ) for sid in sample_ids
    ]
    dataset = MultiHESTDataset(paths, distribution=args.patch_distribution,
                               normalize_method=normalize_method, sample_times=args.sample_times)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=padding_batcher(),
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0 and torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    return loader


def main():
    args = parse_args()
    set_random_seed(args.seed)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "ciga": 512}[args.feature_encoder]

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[bench] device={device} amp={args.amp} compile={args.use_compile}")

    loader = build_loader(args, args.datasets[0])
    print(f"[bench] dataloader steps/epoch = {len(loader)}")

    model = Denoiser(args).to(device)
    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)

    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    amp_enabled = amp_dtype is not None and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # --- warmup step (excluded from timing; pays compile cost here) ---
    warmup_batch = next(iter(loader))
    warmup_batch = [x.to(device, non_blocking=True) for x in warmup_batch]
    img_features, coords, gene_exp = warmup_batch
    noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        _, loss = model(exp=noisy_exp, img_features=img_features, coords=coords,
                        labels=gene_exp, t_steps=t_steps)
    if scaler.is_enabled():
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[bench] warmup done")

    # --- timed loop ---
    step_times = []
    epoch_start = time.perf_counter()
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            t0 = time.perf_counter()
            batch = [x.to(device, non_blocking=True) for x in batch]
            img_features, coords, gene_exp = batch

            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                _, loss = model(exp=noisy_exp, img_features=img_features, coords=coords,
                                labels=gene_exp, t_steps=t_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
    epoch_time = time.perf_counter() - epoch_start

    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0.0

    report = {
        "label": args.label,
        "amp": args.amp,
        "use_compile": args.use_compile,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_total": len(step_times),
        "wall_clock_s": round(epoch_time, 3),
        "step_time_mean_ms": round(1000 * sum(step_times) / len(step_times), 3),
        "step_time_median_ms": round(1000 * sorted(step_times)[len(step_times) // 2], 3),
        "step_time_p90_ms": round(1000 * sorted(step_times)[int(0.9 * len(step_times))], 3),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1),
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[bench] wrote {args.out_json}")


if __name__ == "__main__":
    main()
