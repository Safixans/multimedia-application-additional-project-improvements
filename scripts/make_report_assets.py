"""
Run the measured micro-benchmarks and generate chart PNGs for the PDF report.

Outputs into reports/:
  - bench_results.json   (raw numbers)
  - chart_speedup.png    (overall speedup per optimization)
  - chart_kdtree.png     (before/after ms)
  - chart_distance.png   (before/after ms)

Only measured (CPU-verifiable) optimizations are charted.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import KDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS, exist_ok=True)

BLUE = "#9aa7b5"      # "before" (muted)
GREEN = "#2e7d32"     # "after" (highlight)


def timeit(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000.0


def bench_kdtree(n_spots, repeats):
    coords = np.random.rand(n_spots, 2).astype(np.float32)
    k = max(2, int(n_spots * 0.5))
    center = coords[np.random.randint(0, n_spots)]
    cached = KDTree(coords)
    t_orig = timeit(lambda: (KDTree(coords).query(center, k=k)), repeats)
    t_opt = timeit(lambda: cached.query(center, k=k), repeats)
    return t_orig, t_opt


def bench_distance(n_spots, repeats):
    from einops import rearrange
    coords = torch.randn(n_spots, 2)

    def original():
        rel = rearrange(coords, 'n d -> n 1 d') - rearrange(coords, 'n d -> 1 n d')
        rel.norm(dim=-1).topk(8, dim=-1, largest=False)

    def optimized():
        torch.cdist(coords, coords, p=2).topk(8, dim=-1, largest=False)

    with torch.no_grad():
        return timeit(original, repeats), timeit(optimized, repeats)


def bench_inference(n_spots, n_steps, repeats):
    from stflow.model.denoiser import Denoiser

    class A: pass
    a = A()
    a.n_genes, a.feature_dim, a.hidden_dim, a.pairwise_hidden_dim = 50, 1024, 128, 128
    a.n_layers, a.dropout, a.attn_dropout, a.n_neighbors, a.n_heads = 4, 0.0, 0.0, 8, 4
    a.activation = "gelu"
    model = Denoiser(a).eval()
    img = torch.randn(1, n_spots, a.feature_dim)
    coords = torch.randn(1, n_spots, 2)
    exp = torch.randn(1, n_spots, a.n_genes)

    @torch.no_grad()
    def original():
        for i in range(n_steps):
            model.inference(exp, img, coords, torch.full((1,), 0.1 + 0.1 * i), cached_graph=None)

    @torch.no_grad()
    def optimized():
        cache = model.build_inference_cache(img, coords)
        for i in range(n_steps):
            model.inference(exp, img, coords, torch.full((1,), 0.1 + 0.1 * i), cached_graph=cache)

    return timeit(original, repeats, warmup=2), timeit(optimized, repeats, warmup=2)


def bar_before_after(path, title, before, after, ylabel="time per call (ms)"):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(["Before", "After"], [before, after], color=[BLUE, GREEN], width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, [before, after]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def speedup_chart(path, labels, speedups):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bars = ax.barh(labels, speedups, color=GREEN)
    ax.axvline(1.0, color="#b00020", linestyle="--", linewidth=1, label="baseline (1.0x)")
    ax.set_xlabel("speedup factor  (higher = faster)", fontsize=9)
    ax.set_title("Measured speedup per optimization (CPU)", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, speedups):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.2f}x", va="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    n_spots, repeats = 2000, 30
    print(f"Benchmarking (n_spots={n_spots}) ...")

    kt_o, kt_a = bench_kdtree(n_spots, max(repeats, 50))
    di_o, di_a = bench_distance(n_spots, max(repeats, 50))
    inf_o, inf_a = bench_inference(n_spots, 5, 12)

    results = {
        "n_spots": n_spots,
        "kdtree": {"before_ms": kt_o, "after_ms": kt_a, "speedup": kt_o / kt_a},
        "distance": {"before_ms": di_o, "after_ms": di_a, "speedup": di_o / di_a},
        "inference": {"before_ms": inf_o, "after_ms": inf_a, "speedup": inf_o / inf_a},
    }
    with open(os.path.join(REPORTS, "bench_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    bar_before_after(os.path.join(REPORTS, "chart_kdtree.png"),
                     "KDTree build (per patch sample)", kt_o, kt_a)
    bar_before_after(os.path.join(REPORTS, "chart_distance.png"),
                     "Neighbor distance matrix", di_o, di_a)
    speedup_chart(os.path.join(REPORTS, "chart_speedup.png"),
                  ["KDTree cached\nper slide",
                   "Distance via\ntorch.cdist",
                   "KNN graph cached\nacross 5 steps"],
                  [results["kdtree"]["speedup"],
                   results["distance"]["speedup"],
                   results["inference"]["speedup"]])
    print("Charts written to reports/")


if __name__ == "__main__":
    main()
