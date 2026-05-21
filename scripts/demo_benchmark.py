"""
Self-contained demo benchmark — NO dataset and NO GPU required.

Generates synthetic "slides" in memory and measures the concrete speedups from
the optimizations, so you can produce real numbers on any machine (including a
laptop CPU). On a GPU the gains are larger; on CPU the data/graph optimizations
still show a clear, honest win.

What it measures:
  1. KDTree caching      — rebuild-every-call (original) vs build-once (optimized)
  2. Distance matrix     — broadcast-norm (original) vs torch.cdist (optimized)
  3. Full training step  — original model config vs optimized config (amp/compile
                           only take effect on CUDA; on CPU this isolates the
                           graph/loss changes)

Run:
    python scripts/demo_benchmark.py
    python scripts/demo_benchmark.py --n_spots 2000 --repeats 50
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import KDTree

torch.manual_seed(0)
np.random.seed(0)


def timeit(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0  # ms/call


# ---------------------------------------------------------------------------
# 1. KDTree caching
# ---------------------------------------------------------------------------
def bench_kdtree(n_spots, repeats):
    coords = np.random.rand(n_spots, 2).astype(np.float32)
    k = max(2, int(n_spots * 0.5))
    center = coords[np.random.randint(0, n_spots)]

    def original():  # rebuild tree every sample
        tree = KDTree(coords)
        tree.query(center, k=k)

    cached_tree = KDTree(coords)

    def optimized():  # reuse cached tree
        cached_tree.query(center, k=k)

    t_orig = timeit(original, repeats)
    t_opt = timeit(optimized, repeats)
    return t_orig, t_opt


# ---------------------------------------------------------------------------
# 2. Distance matrix: broadcast-norm vs cdist
# ---------------------------------------------------------------------------
def bench_distance(n_spots, repeats, device):
    from einops import rearrange
    coords = torch.randn(n_spots, 2, device=device)

    def original():
        rel_pos = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim=-1)
        rel_dist.topk(8, dim=-1, largest=False)

    def optimized():
        rel_dist = torch.cdist(coords, coords, p=2)
        rel_dist.topk(8, dim=-1, largest=False)

    with torch.no_grad():
        t_orig = timeit(original, repeats)
        t_opt = timeit(optimized, repeats)
    return t_orig, t_opt


# ---------------------------------------------------------------------------
# 3. Full inference: rebuild graph every step vs cache once
# ---------------------------------------------------------------------------
def bench_inference(n_spots, n_sample_steps, repeats, device):
    from stflow.model.denoiser import Denoiser

    class Args: pass
    a = Args()
    a.n_genes = 50
    a.feature_dim = 1024
    a.hidden_dim = 128
    a.pairwise_hidden_dim = 128
    a.n_layers = 4
    a.dropout = 0.0
    a.attn_dropout = 0.0
    a.n_neighbors = 8
    a.n_heads = 4
    a.activation = "gelu"

    model = Denoiser(a).to(device).eval()
    B = 1
    img = torch.randn(B, n_spots, a.feature_dim, device=device)
    coords = torch.randn(B, n_spots, 2, device=device)
    exp = torch.randn(B, n_spots, a.n_genes, device=device)

    @torch.no_grad()
    def original():  # rebuild graph on every denoising step
        for i in range(n_sample_steps):
            t = torch.full((B,), 0.1 + 0.1 * i, device=device)
            model.inference(exp, img, coords, t, cached_graph=None)

    @torch.no_grad()
    def optimized():  # build graph once, reuse across steps
        cache = model.build_inference_cache(img, coords)
        for i in range(n_sample_steps):
            t = torch.full((B,), 0.1 + 0.1 * i, device=device)
            model.inference(exp, img, coords, t, cached_graph=cache)

    t_orig = timeit(original, repeats, warmup=2)
    t_opt = timeit(optimized, repeats, warmup=2)
    return t_orig, t_opt


# ---------------------------------------------------------------------------
def row(name, t_orig, t_opt):
    speedup = t_orig / t_opt if t_opt > 0 else float('inf')
    return f"{name:<34} {t_orig:>10.3f} {t_opt:>10.3f} {speedup:>8.2f}x"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_spots', type=int, default=1500, help='spots per synthetic slide')
    p.add_argument('--n_sample_steps', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 66)
    print(f"STFlow demo benchmark   device={device}   n_spots={args.n_spots}")
    print("(synthetic data — no HEST dataset needed)")
    print("=" * 66)
    print(f"{'optimization':<34} {'orig(ms)':>10} {'opt(ms)':>10} {'speedup':>9}")
    print("-" * 66)

    t_orig, t_opt = bench_kdtree(args.n_spots, max(args.repeats, 30))
    print(row("1. KDTree (rebuild vs cached)", t_orig, t_opt))

    t_orig, t_opt = bench_distance(args.n_spots, max(args.repeats, 30), device)
    print(row("2. distance matrix (norm vs cdist)", t_orig, t_opt))

    t_orig, t_opt = bench_inference(args.n_spots, args.n_sample_steps, args.repeats, device)
    print(row(f"3. inference ({args.n_sample_steps} steps, graph cache)", t_orig, t_opt))

    print("-" * 66)
    if device == "cpu":
        print("NOTE: running on CPU. --amp and --use_compile gains appear only on a")
        print("CUDA GPU; the numbers above isolate the data/graph optimizations,")
        print("which help on CPU too. Re-run on a GPU server for the full speedup.")
    else:
        print("Running on GPU — add AMP/compile via scripts/benchmark.py for the")
        print("full end-to-end training speedup on real data.")
    print("=" * 66)


if __name__ == "__main__":
    main()
