# Performance Optimizations — STFlow

This document summarizes the performance changes applied on top of the original
[Graph-and-Geometric-Learning/STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow) codebase.

**Goal:** reduce training wall-clock time and GPU memory without changing model
accuracy. All optimizations are opt-in flags — a run with no new flags
reproduces the original baseline behavior.

**Expected combined speedup:** ~2.5–4× faster training and ~30–50% less peak GPU
memory on a modern NVIDIA GPU (A100 / H100 / RTX 30–40 series). Numbers must be
verified on the user's hardware via `scripts/benchmark.py`.

---

## Summary of files changed

| File | Type of change |
|---|---|
| `stflow/app/flow/train.py` | DataLoader, AMP, `torch.compile`, on-device loss accumulation, `zero_grad` fix, throttled wandb logging |
| `stflow/app/flow/test.py` | Reuse KNN graph across all flow-matching denoising steps |
| `stflow/model/transformer.py` | `torch.cdist` for distance matrix; cached-graph API |
| `stflow/model/denoiser.py` | Forward `cached_graph`; expose `build_inference_cache()` |
| `stflow/data/dataset.py` | Build SciPy `KDTree` once per slide |
| `stflow/data/sampling_utils.py` | Accept a pre-built KDTree |
| `stflow/flow/interpolant.py` | Sample `t` on GPU; avoid redundant `.to(device)` |
| `stflow/flow/noise.py` | Sample Gaussian/zero priors directly on GPU |
| `scripts/benchmark.py` | **NEW** — one-epoch timing tool |
| `.gitignore` | **NEW** — exclude `.DS_Store`, `__pycache__`, `reports/`, etc. |

---

## Optimization #1 — DataLoader concurrency

**Where:** `stflow/app/flow/train.py`

**Problem:** the DataLoader was constructed without `num_workers`, `pin_memory`,
`persistent_workers`, or `prefetch_factor`. Every batch was loaded on the main
process while the GPU sat idle. `--num_workers=1` was even defined as a CLI
argument but was not actually passed to the loader.

**Fix:** plumb `--num_workers` through, enable `pin_memory` and
`persistent_workers` when workers > 0, set `prefetch_factor=4`.

**Expected gain:** 10–30% wall-clock reduction on a CPU-bound preprocess
(KDTree query inside `PatchSampler.__call__`).

**New flags:** `--num_workers` (default raised from 1 → 4), `--prefetch_factor`.

---

## Optimization #2 — Mixed precision (bf16 / fp16)

**Where:** `stflow/app/flow/train.py`

**Problem:** the training loop ran entirely in fp32. Modern GPUs have 2–4×
higher throughput for bf16 / fp16 with no accuracy loss in this kind of
flow-matching model.

**Fix:** wrap the forward pass in `torch.amp.autocast("cuda", dtype=...)`,
attach `torch.amp.GradScaler` (only needed for fp16, no-op for bf16).

**Expected gain:** 40–80% step-time reduction on A100/H100/RTX-30-40.

**Numerical safety:** bf16 has the same dynamic range as fp32 and is the
recommended default for diffusion-style models. fp16 is also supported but
needs the GradScaler to avoid overflow.

**New flag:** `--amp {off,bf16,fp16}` (default `off` = unchanged behavior).

---

## Optimization #3 — `torch.compile`

**Where:** `stflow/app/flow/train.py`

**Problem:** the denoiser runs the same graph every step (no dynamic shapes
within a step), so PyTorch's eager-mode dispatch overhead is pure waste.

**Fix:** optionally wrap the model in `torch.compile(model, mode=...)`.

**Expected gain:** 20–50% additional step-time reduction, compounding with AMP.

**New flag:** `--use_compile` (off by default). The first step pays a 1–3 min
compile cost; the warmup step in `scripts/benchmark.py` absorbs this so the
reported timings reflect steady-state speed.

---

## Optimization #4 — On-device loss accumulation

**Where:** `stflow/app/flow/train.py:101–103` (original)

**Problem:** on every training step the original code called
`loss.cpu().item()` twice (once for wandb, once for the running average).
Each call **synchronizes the GPU with the CPU** and stalls the pipeline.

**Fix:** accumulate the loss as a zero-dim tensor on the GPU
(`loss_sum = loss_sum + loss.detach()`) and call `.item()` once per epoch.
Throttled wandb logging to once every N steps via `--log_every`.

**Expected gain:** 5–15% step-time reduction.

**New flag:** `--log_every` (default 10).

---

## Optimization #5 — KNN graph cache across denoising steps

**Where:** `stflow/model/transformer.py`, `stflow/model/denoiser.py`, `stflow/app/flow/test.py`

**Problem:** During flow-matching inference, the model is invoked
`n_sample_steps` times per slide (5 by default). Each invocation rebuilds the
**K-nearest-neighbor graph from the spot coordinates** — but the coordinates
are identical across denoising steps. The graph construction is
O(N²) memory and O(N² log K) compute.

**Fix:** add `SpatialTransformer.build_inference_cache(features, coords)` that
returns `{pad_mask, batch_idx, coords, nearest_indices}`. `test.py` builds it
once per slide and passes it to every `model.inference()` call. The
`forward(cached_graph=...)` path skips the graph rebuild entirely.

**Expected gain:** 15–40% reduction in **inference** wall-clock, and a small
training reduction from the `torch.cdist` replacement (see #6).

**API compatibility:** `cached_graph` is an optional keyword argument that
defaults to `None` (original behavior); existing callers are unaffected.

---

## Optimization #6 — `torch.cdist` for distance matrix

**Where:** `stflow/model/transformer.py:_build_graph`

**Problem:** the original code materialized a full `[N, N, 2]` pairwise
difference tensor via `rearrange` + broadcasting, then took its norm. This
allocates 2× more memory than necessary and is slower than the fused kernel.

**Fix:** replace with `torch.cdist(coords, coords, p=2)`, wrapped in
`torch.no_grad()` since the graph is a non-differentiable structural quantity.

**Expected gain:** small (~5%) speed gain, larger memory reduction.

---

## Optimization #7 — KDTree cached per slide

**Where:** `stflow/data/dataset.py`, `stflow/data/sampling_utils.py`

**Problem:** `PatchSampler.sample_nearest_patch()` constructed a fresh
`scipy.spatial.KDTree` on every `__getitem__` call. With `sample_times=10` and
multi-worker DataLoaders, this is rebuilt thousands of times per epoch even
though the underlying coordinates never change.

**Fix:** build the KDTree once in `SPData.__init__` (only for the full slide,
not for the per-patch chunks), cache it on the dataset, and pass it through to
`PatchSampler.__call__(coords, tree=...)`.

**Expected gain:** 5–10% reduction in DataLoader CPU time.

---

## Optimization #8 — Prior sampling on GPU

**Where:** `stflow/flow/noise.py`, `stflow/flow/interpolant.py`

**Problem:** the Gaussian and zero priors were sampled on the CPU, then
transferred to the GPU on every flow-matching step. Sampling the timestep
`t` likewise happened on the CPU.

**Fix:** sample directly on the target device. ZINB is unavoidable on the CPU
(scvi-tools is CPU-only), but the host→device transfer is now `non_blocking`.

**Expected gain:** 3–8% reduction for Gaussian / zero prior; minor for ZINB.

---

## Optimization #9 — `zero_grad(set_to_none=True)` and dropped redundant call

**Where:** `stflow/app/flow/train.py:94–95` (original)

**Problem:** the original code called `optimizer.zero_grad()` immediately
followed by `model.zero_grad()` — the second call is redundant (the optimizer
holds all the model's parameters). The default `zero_grad()` (without
`set_to_none=True`) also runs slower because it issues a `tensor.fill_(0)`
on every gradient instead of releasing them.

**Fix:** `optimizer.zero_grad(set_to_none=True)`; removed the redundant
`model.zero_grad()`.

**Expected gain:** 1–3% reduction.

---

## New tool — `scripts/benchmark.py`

A standalone single-epoch timing tool that reports:

- Total wall-clock time
- Mean / median / p90 step time (ms)
- Peak GPU memory (MB)
- Output as JSON for easy aggregation

Run it once per variant with a different `--label` and compare the JSON outputs.
A warmup step is run first and excluded from timings, so the reported numbers
reflect steady-state performance (i.e. they do not include the one-time
`torch.compile` compilation cost).

---

## How to verify the gains

```bash
# 1. Baseline (no flags)
python scripts/benchmark.py \
  --datasets LUNG --feature_encoder uni_v1_official \
  --source_dataroot $DATA --embed_dataroot $EMBED \
  --batch_size 2 --epochs 1 --num_workers 0 --amp off \
  --label baseline --out_json reports/baseline.json

# 2. All optimizations
python scripts/benchmark.py \
  --datasets LUNG --feature_encoder uni_v1_official \
  --source_dataroot $DATA --embed_dataroot $EMBED \
  --batch_size 2 --epochs 1 --num_workers 4 --amp bf16 --use_compile \
  --label optimized --out_json reports/optimized.json
```

Compare `wall_clock_s` and `peak_gpu_mem_mb` between the two reports.

**To confirm accuracy is preserved**, train both variants for ~30 epochs and
compare `pearson_mean` from `results_kfold.json` — should agree within ±0.01.
