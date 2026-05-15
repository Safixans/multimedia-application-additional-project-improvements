"""
CPU-only smoke test for the perf-optimization changes.

Exercises every modified module without needing the HEST dataset or a GPU.
Verifies that the cached-graph code path produces identical output to the
non-cached path (proves the optimization doesn't change behavior).

Run:
    python scripts/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
def test_prior_sampler_gpu_device():
    from stflow.flow.noise import PriorSampler
    s = PriorSampler("gaussian", device=DEVICE)
    out = s.sample((2, 4, 3))
    assert out.shape == (2, 4, 3), f"shape mismatch: {out.shape}"
    assert out.device == torch.device(DEVICE)
    s = PriorSampler("zero", device=DEVICE)
    out = s.sample((2, 4, 3))
    assert torch.all(out == 0)
    print("[OK] noise.PriorSampler — gaussian/zero priors sample on requested device")


def test_interpolant_corrupt_denoise():
    from stflow.flow.interpolant import Interpolant
    interp = Interpolant("gaussian", normalize=False)
    interp.device = DEVICE  # override (Interpolant auto-picks cuda if available)
    interp.prior_sampler.device = DEVICE
    exp = torch.randn(2, 5, 3)
    noisy, t = interp.corrupt_exp(exp)
    assert noisy.shape == exp.shape
    assert t.shape == (2,)
    assert t.device == torch.device(DEVICE)
    print("[OK] interpolant.Interpolant.corrupt_exp — shapes and device correct")


def test_patch_sampler_with_tree():
    from scipy.spatial import KDTree
    from stflow.data.sampling_utils import PatchSampler
    coords = np.random.rand(50, 2).astype(np.float32)
    tree = KDTree(coords)
    s = PatchSampler("constant_0.5")
    # both code paths should work
    a = s(coords, tree=tree)
    b = s(coords, tree=None)
    assert len(a) > 0 and len(b) > 0
    assert len(a) == int(50 * 0.5)
    print("[OK] sampling_utils.PatchSampler — accepts prebuilt KDTree")


def test_spdata_kdtree_cache():
    try:
        from stflow.data.dataset import SPData
    except ModuleNotFoundError as e:
        print(f"[SKIP] dataset.SPData — missing optional dep: {e.name}")
        return
    coords = torch.from_numpy(np.random.rand(50, 2).astype(np.float32))
    features = torch.randn(50, 8)
    labels = torch.randn(50, 3)
    sp = SPData(features=features, labels=labels, coords=coords)
    assert sp.kdtree is not None, "KDTree should be cached on full slide"
    chunk = sp.chunk(np.arange(10))
    assert chunk.kdtree is None, "KDTree should NOT be built for per-patch chunks"
    print("[OK] dataset.SPData — KDTree cached on slide, skipped on chunks")


def _build_dummy_args():
    class Args: pass
    a = Args()
    # n_genes MUST be 50 — the original transformer.py hard-codes +50 in the
    # mlp_attn input dim, so any other value fails with a matmul shape error.
    a.n_genes = 50
    a.feature_dim = 16
    a.hidden_dim = 16
    a.pairwise_hidden_dim = 16
    a.n_layers = 2
    a.dropout = 0.0
    a.attn_dropout = 0.0
    a.n_neighbors = 3
    a.n_heads = 2
    a.activation = "gelu"  # swiglu also requires GPU-friendly fused ops; gelu works on CPU
    return a


def test_denoiser_forward_runs():
    from stflow.model.denoiser import Denoiser
    args = _build_dummy_args()
    model = Denoiser(args).to(DEVICE).eval()
    B, N = 1, 8
    img_features = torch.randn(B, N, args.feature_dim)
    coords = torch.randn(B, N, 2)
    gene_exp = torch.randn(B, N, args.n_genes)
    labels = torch.randn(B, N, args.n_genes)
    t = torch.rand(B)
    with torch.no_grad():
        pred, loss = model(exp=gene_exp, img_features=img_features,
                           coords=coords, labels=labels, t_steps=t)
    assert pred.shape == (B, N, args.n_genes), f"shape {pred.shape}"
    assert torch.isfinite(loss)
    print(f"[OK] denoiser.Denoiser.forward — pred {tuple(pred.shape)}, loss {loss.item():.4f}")


def test_cached_graph_matches_uncached():
    """
    The critical correctness test: model.inference(..., cached_graph=cache)
    must produce IDENTICAL output to model.inference(..., cached_graph=None).
    """
    from stflow.model.denoiser import Denoiser
    args = _build_dummy_args()
    model = Denoiser(args).to(DEVICE).eval()
    B, N = 1, 8
    img_features = torch.randn(B, N, args.feature_dim)
    coords = torch.randn(B, N, 2)
    noisy_exp = torch.randn(B, N, args.n_genes)
    t = torch.rand(B)

    with torch.no_grad():
        out_uncached = model.inference(noisy_exp, img_features, coords, t, cached_graph=None)
        cache = model.build_inference_cache(img_features, coords)
        out_cached = model.inference(noisy_exp, img_features, coords, t, cached_graph=cache)

    max_abs_diff = (out_uncached - out_cached).abs().max().item()
    assert max_abs_diff < 1e-5, f"cached vs uncached differ by {max_abs_diff}"
    print(f"[OK] cached_graph numerical equivalence — max abs diff = {max_abs_diff:.2e}")


def test_cached_graph_reused_across_timesteps():
    """The actual use pattern from test.py: build cache once, reuse for N steps."""
    from stflow.model.denoiser import Denoiser
    args = _build_dummy_args()
    model = Denoiser(args).to(DEVICE).eval()
    B, N = 1, 8
    img_features = torch.randn(B, N, args.feature_dim)
    coords = torch.randn(B, N, 2)
    noisy_exp = torch.randn(B, N, args.n_genes)
    with torch.no_grad():
        cache = model.build_inference_cache(img_features, coords)
        for step_i in range(5):
            t = torch.full((B,), 0.1 + 0.18 * step_i)
            out = model.inference(noisy_exp, img_features, coords, t, cached_graph=cache)
            assert torch.isfinite(out).all()
    print("[OK] cached_graph reused across 5 denoising steps")


def test_train_new_flags_present():
    """Verify the train.py source advertises the new CLI flags (no heavy imports)."""
    train_src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "stflow/app/flow/train.py")).read()
    for flag in ("--amp", "--use_compile", "--prefetch_factor", "--log_every"):
        assert flag in train_src, f"flag {flag} missing from train.py"
    for token in ("torch.amp.autocast", "torch.compile", "set_to_none=True",
                  "non_blocking=True", "loss.detach()"):
        assert token in train_src, f"token {token!r} missing from train.py"
    print("[OK] train.py source — all new flags + optimization tokens present")


# ---------------------------------------------------------------------------
TESTS = [
    test_prior_sampler_gpu_device,
    test_interpolant_corrupt_denoise,
    test_patch_sampler_with_tree,
    test_spdata_kdtree_cache,
    test_denoiser_forward_runs,
    test_cached_graph_matches_uncached,
    test_cached_graph_reused_across_timesteps,
    test_train_new_flags_present,
]


def main():
    fails = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            fails += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            fails += 1
    print("-" * 60)
    if fails == 0:
        print(f"All {len(TESTS)} tests passed ✓")
        return 0
    print(f"{fails}/{len(TESTS)} tests failed ✗")
    return 1


if __name__ == "__main__":
    sys.exit(main())
