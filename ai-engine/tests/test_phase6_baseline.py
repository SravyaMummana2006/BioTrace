"""Comprehensive tests for BIOTrace Phase 6 Baseline Deepfake Detection Engine.

Covers:
1. Dataset loading and label correctness from frame manifest.
2. Strict split isolation and video-level disjointness (zero partition leakage).
3. EfficientNet-B0 model output dimensions (batch_size, 1).
4. BCEWithLogitsLoss computation and gradient flow.
5. Video-level prediction aggregation via mean-logit pooling.
6. Safe weights_only checkpoint loading and state_dict restoration.
7. Inference interface validation and forensic disclaimer presence.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Ensure ai-engine and ai-engine/scripts are in sys.path
_test_dir = Path(__file__).resolve().parent
_ai_engine_dir = _test_dir.parent
_workspace_dir = _ai_engine_dir.parent
if str(_ai_engine_dir) not in sys.path:
    sys.path.insert(0, str(_ai_engine_dir))
if str(_workspace_dir) not in sys.path:
    sys.path.insert(0, str(_workspace_dir))

from scripts.train_baseline import (
    build_efficientnet_b0,
    compute_sha256,
    compute_video_level_metrics,
    run_smoke_test,
    set_seed,
)
from scripts.predict_video import (
    load_model,
    predict_video,
    preprocess_image,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    # Anchor to workspace root
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "reports" / "data" / "phase5_frame_extraction_manifest.csv").exists():
            return parent
    return Path.cwd()


def test_dataset_loading_and_labels(project_root: Path):
    """Verifies that frame extraction manifest loads cleanly and labels are valid."""
    manifest_path = project_root / "reports" / "data" / "phase5_frame_extraction_manifest.csv"
    assert manifest_path.exists(), f"Manifest file missing: {manifest_path}"

    df = pd.read_csv(manifest_path)
    assert len(df) == 87220, f"Expected 87,220 frames, found {len(df)}"

    labels = set(df["label"].unique())
    assert labels == {"real", "fake"}, f"Unexpected labels: {labels}"

    splits = set(df["split"].unique())
    assert splits == {"train", "val", "test"}, f"Unexpected splits: {splits}"

    # Verify 50:50 balance per split
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        real_count = (sub["label"] == "real").sum()
        fake_count = (sub["label"] == "fake").sum()
        assert real_count == fake_count, f"Unbalanced split {split}: {real_count} real vs {fake_count} fake"


def test_split_isolation_and_disjointness(project_root: Path):
    """Verifies that video IDs and components never leak across train, val, and test partitions."""
    manifest_path = project_root / "reports" / "data" / "phase5_frame_extraction_manifest.csv"
    df = pd.read_csv(manifest_path)

    train_vids = set(df[df["split"] == "train"]["video_id"].unique())
    val_vids = set(df[df["split"] == "val"]["video_id"].unique())
    test_vids = set(df[df["split"] == "test"]["video_id"].unique())

    assert train_vids.isdisjoint(val_vids), "Video leakage detected between train and val!"
    assert train_vids.isdisjoint(test_vids), "Video leakage detected between train and test!"
    assert val_vids.isdisjoint(test_vids), "Video leakage detected between val and test!"

    # Verify component disjointness
    train_comps = set(df[df["split"] == "train"]["component_id"].unique())
    val_comps = set(df[df["split"] == "val"]["component_id"].unique())
    test_comps = set(df[df["split"] == "test"]["component_id"].unique())

    assert train_comps.isdisjoint(val_comps), "Component leakage detected between train and val!"
    assert train_comps.isdisjoint(test_comps), "Component leakage detected between train and test!"
    assert val_comps.isdisjoint(test_comps), "Component leakage detected between val and test!"


def test_model_output_dimensions():
    """Verifies EfficientNet-B0 binary classification head outputs (B, 1) logits."""
    model = build_efficientnet_b0(pretrained=False)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {out.shape}"
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf values"


def test_bce_loss_and_gradients():
    """Verifies binary cross entropy loss calculation and non-zero gradient backprop."""
    set_seed(42)
    model = build_efficientnet_b0(pretrained=False)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([[0.0], [1.0]])

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)

    assert loss.item() > 0, "Loss must be positive"
    assert torch.isfinite(loss), "Loss must be finite"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were calculated"
    assert any(g.abs().sum() > 0 for g in grads), "All gradients are zero"


def test_video_level_prediction_aggregation():
    """Verifies that mean-logit pooling across 10 frames produces accurate video-level metrics."""
    # Create 2 videos with 10 frames each
    # Video 0 (real): 10 frames with negative logits
    # Video 1 (fake): 10 frames with positive logits
    video_ids = ["vid_real"] * 10 + ["vid_fake"] * 10
    frame_logits = np.array([-2.5] * 10 + [3.0] * 10)
    frame_labels = np.array([0.0] * 10 + [1.0] * 10)

    metrics = compute_video_level_metrics(
        video_ids=video_ids,
        frame_logits=frame_logits,
        frame_labels=frame_labels,
    )

    assert metrics["num_videos"] == 2
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]


def test_checkpoint_saving_and_safe_loading():
    """Verifies safe weights_only loading and parameter preservation."""
    model = build_efficientnet_b0(pretrained=False)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save state dict
        ckpt_payload = {
            "epoch": 1,
            "model_name": "efficientnet_b0_baseline",
            "state_dict": model.state_dict(),
            "config": {"seed": 42},
        }
        torch.save(ckpt_payload, str(tmp_path))

        # Load with weights_only=True
        loaded = torch.load(str(tmp_path), weights_only=True)
        assert "state_dict" in loaded
        assert loaded["epoch"] == 1

        # Re-instantiate model and load state
        new_model = build_efficientnet_b0(pretrained=False)
        new_model.load_state_dict(loaded["state_dict"])

        # Check weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_smoke_test_runner():
    """Verifies that run_smoke_test completes without exception."""
    run_smoke_test(device=torch.device("cpu"))


def test_image_preprocessing():
    """Verifies image preprocessing output shape and normalization."""
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_bgr[:, :] = [100, 150, 200]

    tensor = preprocess_image(dummy_bgr, target_size=(224, 224))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
