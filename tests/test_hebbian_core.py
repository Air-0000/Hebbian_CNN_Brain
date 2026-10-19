"""Unit and Mathematical Verification Tests for Hebbian Core Layers."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.hebbian_layer import HebbianConv2d


def test_hebbian_spatial_bcm_update():
    """Verify BCM spatial sliding threshold updates and weight boundedness."""
    layer = HebbianConv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, learning_rate=0.01)
    layer.train()

    # Initial weight norm
    initial_norm = layer.get_weight_norm().mean().item()
    initial_thresh = layer.bcm_threshold.clone()

    # Simulate 10 batches of visual input
    for _ in range(10):
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        layer.hebbian_update(mode="spatial_bcm")

    # BCM threshold must adaptively adjust
    final_thresh = layer.bcm_threshold
    assert not torch.equal(initial_thresh, final_thresh), "BCM sliding threshold failed to update"

    # Weight norm must remain finite and bounded
    final_norm = layer.get_weight_norm().mean().item()
    assert final_norm < 100.0, f"Weight norm diverged: {final_norm}"
    print(f"  [PASS] test_hebbian_spatial_bcm_update: norm={final_norm:.4f}, thresh_mean={final_thresh.mean():.4f}")


def test_cortical_lateral_inhibition():
    """Verify lateral inhibition enhances output sparsity across cortical channels."""
    layer = HebbianConv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1, use_wta=True)
    layer.train()

    x = torch.randn(2, 4, 16, 16)
    out_with_inhibition = layer(x)

    # Compare with pure conv + relu without lateral inhibition
    out_raw = F.relu(F.conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding))
    sparsity_raw = (out_raw <= 0).float().mean().item()
    sparsity_inhibited = (out_with_inhibition <= 0).float().mean().item()

    assert sparsity_inhibited >= sparsity_raw, f"Lateral inhibition should increase sparsity (got {sparsity_inhibited} vs {sparsity_raw})"
    print(f"  [PASS] test_cortical_lateral_inhibition: raw_sparsity={sparsity_raw:.3f}, inhibited_sparsity={sparsity_inhibited:.3f}")


def test_oja_convergence():
    """Verify Oja's rule prevents runaway weight growth on repetitive signals."""
    layer = HebbianConv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, learning_rate=0.05, oja_beta=0.1)
    layer.train()

    # Strong constant signal
    x = torch.ones(2, 1, 16, 16) * 5.0
    for _ in range(25):
        _ = layer(x)
        layer.hebbian_update(mode="oja")

    max_norm = layer.get_weight_norm().max().item()
    assert max_norm <= 5.0, f"Oja normalization failed, weight norm grew to {max_norm}"
    print(f"  [PASS] test_oja_convergence: bounded max_norm={max_norm:.4f}")


if __name__ == "__main__":
    print("Running Hebbian Brain Core Mathematical Verification Suite...")
    test_hebbian_spatial_bcm_update()
    test_cortical_lateral_inhibition()
    test_oja_convergence()
    print("\nALL HEBBIAN BRAIN CORE TESTS PASSED SUCCESSFULLY!")
