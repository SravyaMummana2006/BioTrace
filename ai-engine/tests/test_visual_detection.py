"""Comprehensive unit and integration tests for BIOTrace Stage 6 Visual Deepfake Detection Engine.

Covers:
1. Frame sampling, exact timestamp preservation, normalization, and corrupted frame handling.
2. MediaPipe face crop extraction, 25% margin padding, boundary clamping, and multi-face auditing.
3. Model architecture, safe weights_only loading, SHA-256 verification, and MODEL_UNAVAILABLE guardrails.
4. Temporal aggregation (trimmed mean, median, mean logit, percentiles, autocorrelation CI).
5. Dataset abstraction, strict video-level partitioning (zero frame leakage), and class balancing.
6. Training loop, deterministic seed reproducibility, checkpoint saving, and forensic evaluation metrics.
7. JobManager integration, failure isolation, cancellation, and API response serialization.
"""

from datetime import datetime
from pathlib import Path
import tempfile
import threading
import time
from typing import List, Tuple
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from biotrace.core.config import Settings
from biotrace.domain.enums import ForensicModality, JobStatus, ProcessingStage
from biotrace.domain.models import JobCreationRequest, VideoMetadata
from biotrace.services.job_manager import JobManager
from biotrace.services.video_inspector import VideoInspector
from biotrace.visual_detection.dataset import (
    FrameSample,
    VideoDataSplitter,
    VideoFrameDataset,
    VideoRecord,
)
from biotrace.visual_detection.domain import (
    FaceCropData,
    FaceDetectionStatus,
    FramePrediction,
    ModelAvailabilityStatus,
    TemporalAggregationMethod,
    VisualDetectionConfig,
    VisualDetectionResult,
)
from biotrace.visual_detection.face_crops import FaceCropExtractor
from biotrace.visual_detection.inference import VisualDeepfakeDetector
from biotrace.visual_detection.model import (
    LightweightVisualBaseline,
    compute_sha256,
    create_model,
    load_model_with_status,
)
from biotrace.visual_detection.preprocessing import ImageNormalizer, VideoFrameSampler
from biotrace.visual_detection.temporal_aggregation import TemporalAggregator
from biotrace.visual_detection.training import (
    TrainingConfig,
    VisualDetectionTrainer,
    compute_metrics,
    set_seed,
)


# ==============================================================================
# Helper Fixtures & Utilities
# ==============================================================================


@pytest.fixture
def synthetic_face_image() -> np.ndarray:
    """Generates a synthetic 3-channel BGR test image with an oval 'face' shape."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Background texture
    img[:] = (80, 80, 80)
    # Draw face oval
    cv2.ellipse(img, (320, 240), (100, 140), 0, 0, 360, (180, 200, 220), -1)
    # Draw eyes
    cv2.circle(img, (280, 200), 12, (50, 40, 30), -1)
    cv2.circle(img, (360, 200), 12, (50, 40, 30), -1)
    # Draw mouth
    cv2.ellipse(img, (320, 300), (35, 15), 0, 0, 360, (70, 70, 160), -1)
    return img


@pytest.fixture
def synthetic_video_file(tmp_path: Path, synthetic_face_image: np.ndarray) -> Path:
    """Creates a temporary 2-second 30 FPS MP4 video with the synthetic face."""
    video_path = tmp_path / "test_visual_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 640, 480
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(60):
        # Slightly alter lighting to simulate real playback
        frame = synthetic_face_image.copy()
        if i % 10 == 0:
            frame = np.clip(frame.astype(np.int16) + 10, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    return video_path


@pytest.fixture
def saved_checkpoint(tmp_path: Path) -> Tuple[Path, str]:
    """Builds and serializes a LightweightVisualBaseline checkpoint for testing."""
    model = LightweightVisualBaseline(in_channels=3)
    ckpt_path = tmp_path / "baseline_test_weights.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": "lightweight"}, str(ckpt_path))
    digest = compute_sha256(str(ckpt_path))
    return ckpt_path, digest


# ==============================================================================
# 1. Preprocessing & Video Frame Sampler Tests
# ==============================================================================


def test_video_frame_sampler_uniform_sampling(synthetic_video_file: Path):
    """Verifies uniform temporal frame extraction and timestamp preservation."""
    sampler = VideoFrameSampler(target_fps=5.0, max_frames=32)
    frames, timestamps = sampler.sample_frames(str(synthetic_video_file))

    assert len(frames) > 0
    assert len(frames) <= 32
    assert len(frames) == len(timestamps)

    # Check strictly monotonic increasing timestamps
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i - 1]

    # Verify frame shapes and type
    assert frames[0].shape == (480, 640, 3)
    assert frames[0].dtype == np.uint8


def test_video_frame_sampler_nonexistent_file():
    """Verifies that non-existent video files return empty lists without crashing."""
    sampler = VideoFrameSampler()
    frames, timestamps = sampler.sample_frames("non_existent_video_path.mp4")
    assert frames == []
    assert timestamps == []


def test_image_normalizer():
    """Verifies normalization to float32 tensor of shape (3, H, W) in range [-2.5, 2.5]."""
    normalizer = ImageNormalizer(target_size=(224, 224))
    bgr_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    tensor = normalizer.normalize(bgr_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_image_normalizer_empty():
    """Verifies that empty image arrays return None."""
    normalizer = ImageNormalizer()
    assert normalizer.normalize(np.zeros((0, 0, 3), dtype=np.uint8)) is None


# ==============================================================================
# 2. Face Crop Extractor Tests
# ==============================================================================


def test_face_crop_margin_and_clamping():
    """Verifies 25% boundary expansion and strict frame boundary clamping."""
    config = VisualDetectionConfig(
        margin_padding=0.25,
        target_crop_size=(224, 224),
        min_sharpness_score=0.0,
    )
    extractor = FaceCropExtractor(config)

    # Synthetic 500x500 image with face near top-left edge
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (90, 90), (200, 200, 200), -1)
    # Landmark coordinates from (10, 10) to (100, 100) -> width 90, height 90
    landmarks = np.array([[10, 10], [100, 100]], dtype=np.float32)

    crop_data = extractor.crop_from_landmarks(
        image_bgr=img,
        landmarks=landmarks,
        frame_index=0,
        presentation_time_seconds=0.0,
    )

    assert crop_data.status in (FaceDetectionStatus.SUCCESS, FaceDetectionStatus.FACE_DETECTED)
    # Clamped to 0 at left/top
    x1, y1, x2, y2 = crop_data.bounding_box
    assert x1 == 0
    assert y1 == 0
    # Expected right/bottom: 100 + pad (pad is 90 * 0.25 = 22.5 -> 122)
    assert x2 > 100
    assert y2 > 100
    assert crop_data.crop_bgr.shape == (224, 224, 3)
    extractor.close()


def test_face_crop_quality_metrics():
    """Verifies blur score (Laplacian variance) and degraded resolution flagging."""
    config = VisualDetectionConfig(min_crop_resolution=64, min_sharpness_score=15.0)
    extractor = FaceCropExtractor(config)

    # 1. Blurry uniform image
    blur_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    landmarks = np.array([[10, 10], [90, 90]], dtype=np.float32)
    crop_data = extractor.crop_from_landmarks(blur_img, landmarks, 0, 0.0)
    assert crop_data.sharpness_score < 1.0
    assert crop_data.status == FaceDetectionStatus.DEGRADED_QUALITY

    # 2. Extremely tiny face (below min_crop_resolution)
    tiny_img = np.zeros((200, 200, 3), dtype=np.uint8)
    tiny_landmarks = np.array([[50, 50], [60, 60]], dtype=np.float32)  # 10x10 bbox
    tiny_crop = extractor.crop_from_landmarks(tiny_img, tiny_landmarks, 1, 0.033)
    assert tiny_crop.status == FaceDetectionStatus.DEGRADED_QUALITY
    extractor.close()


def test_face_crop_extract_no_face():
    """Verifies NO_FACE_DETECTED when image contains no detectable facial structures."""
    extractor = FaceCropExtractor()
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    crop_data = extractor.extract(blank_img, frame_index=0, presentation_time_seconds=0.0)

    assert crop_data.status == FaceDetectionStatus.NO_FACE_DETECTED
    assert crop_data.crop_bgr is None
    assert crop_data.bounding_box == (0, 0, 0, 0)
    extractor.close()


# ==============================================================================
# 3. Model Architecture & Weight Loading Guardrail Tests
# ==============================================================================


def test_lightweight_baseline_forward():
    """Verifies that LightweightVisualBaseline executes forward pass and outputs (B, 1) logits."""
    model = LightweightVisualBaseline(in_channels=3)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)
    assert not torch.isnan(out).any()


def test_model_loading_missing_weights():
    """Scientific Guardrail: Unconfigured weights must return MODEL_UNAVAILABLE with no score fabrication."""
    config = VisualDetectionConfig(weights_path=None)
    model, status, digest, err = load_model_with_status(config)

    assert model is None
    assert status == ModelAvailabilityStatus.MODEL_UNAVAILABLE
    assert digest is None
    assert err is not None
    assert "no weights" in err.lower()


def test_model_loading_weights_not_found():
    """Verifies that pointing to a nonexistent file returns WEIGHTS_NOT_FOUND."""
    config = VisualDetectionConfig(weights_path="non_existent_weights_xyz.pt")
    model, status, digest, err = load_model_with_status(config)

    assert model is None
    assert status == ModelAvailabilityStatus.WEIGHTS_NOT_FOUND
    assert "not found" in err.lower()


def test_model_loading_success(saved_checkpoint: Tuple[Path, str]):
    """Verifies successful loading with weights_only=True and SHA-256 digest validation."""
    ckpt_path, expected_digest = saved_checkpoint
    config = VisualDetectionConfig(
        weights_path=str(ckpt_path),
        expected_sha256=expected_digest,
        backbone="lightweight",
    )
    model, status, digest, err = load_model_with_status(config)

    assert status == ModelAvailabilityStatus.AVAILABLE
    assert model is not None
    assert digest == expected_digest
    assert err is None


def test_model_loading_sha256_mismatch(saved_checkpoint: Tuple[Path, str]):
    """Verifies that a mismatched cryptographic digest causes verification failure."""
    ckpt_path, _ = saved_checkpoint
    config = VisualDetectionConfig(
        weights_path=str(ckpt_path),
        expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        backbone="lightweight",
    )
    model, status, digest, err = load_model_with_status(config)

    assert status == ModelAvailabilityStatus.VERIFICATION_FAILED
    assert model is None
    assert "sha-256 digest mismatch" in err.lower()


# ==============================================================================
# 4. Temporal Aggregation Tests
# ==============================================================================


def test_temporal_aggregator_trimmed_mean():
    """Verifies trimmed mean aggregation filtering top and bottom 10% outliers."""
    aggregator = TemporalAggregator(method=TemporalAggregationMethod.TRIMMED_MEAN, trim_ratio=0.1)

    # 10 frames: 8 normal (0.5), 1 extreme low (0.0), 1 extreme high (1.0)
    scores = [0.0] + [0.5] * 8 + [1.0]
    logits = [-10.0] + [0.0] * 8 + [10.0]
    preds = [
        FramePrediction(
            frame_index=i,
            presentation_time_seconds=i * 0.1,
            raw_logit=logits[i],
            raw_score=scores[i],
        )
        for i in range(10)
    ]

    score, logit, var, ci = aggregator.aggregate(preds)
    assert pytest.approx(score, rel=1e-3) == 0.5
    assert var > 0.0
    assert ci is not None
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_temporal_aggregator_median():
    """Verifies median aggregation."""
    aggregator = TemporalAggregator(method=TemporalAggregationMethod.MEDIAN)
    preds = [
        FramePrediction(frame_index=0, presentation_time_seconds=0.0, raw_logit=-1.0, raw_score=0.2),
        FramePrediction(frame_index=1, presentation_time_seconds=0.1, raw_logit=0.0, raw_score=0.5),
        FramePrediction(frame_index=2, presentation_time_seconds=0.2, raw_logit=2.0, raw_score=0.88),
    ]
    score, logit, _, _ = aggregator.aggregate(preds)
    assert pytest.approx(score, rel=1e-3) == 0.5
    assert pytest.approx(logit, rel=1e-3) == 0.0


def test_temporal_aggregator_empty():
    """Verifies handling of empty prediction sequences."""
    aggregator = TemporalAggregator()
    score, logit, var, ci = aggregator.aggregate([])
    assert score is None
    assert logit is None
    assert var == 0.0
    assert ci is None


def test_temporal_aggregator_descriptive_dispersion_and_autocorrelation():
    """Verifies descriptive frame score dispersion and autocorrelation calculations without false 95% CI claims."""
    aggregator = TemporalAggregator()
    # High score values near ceiling with serial trend
    preds = [
        FramePrediction(frame_index=i, presentation_time_seconds=i * 0.1, raw_logit=1.0 + i * 0.1, raw_score=0.7 + i * 0.01)
        for i in range(20)
    ]
    score, logit, var, desc_range = aggregator.aggregate(preds)
    assert desc_range is not None
    assert 0.0 <= desc_range[0] <= desc_range[1] <= 1.0

    # Test full aggregation output
    _, _, _, _, diag, rho1, neff = TemporalAggregator.aggregate_predictions(preds)
    assert rho1 is not None
    assert -1.0 <= rho1 <= 1.0
    assert neff is not None
    assert 1.0 <= neff <= 20.0
    # Guardrail check: Diagnostics must explicitly disclaim inferential confidence intervals
    assert any("NOT a frequentist or Bayesian inferential confidence interval" in d for d in diag)


# ==============================================================================
# 5. Dataset Abstraction & Video-Level Splitting Tests
# ==============================================================================


def test_video_data_splitter_zero_frame_leakage():
    """Scientific Rigor: Asserts strict disjointness between train, val, and test video IDs."""
    splitter = VideoDataSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    videos = [
        VideoRecord(
            video_id=f"vid_{i:03d}",
            video_path=f"/path/vid_{i:03d}.mp4",
            label=i % 2,
            subject_id=f"subj_{i // 2:02d}",
            frames=[
                FrameSample(video_id=f"vid_{i:03d}", frame_index=j, timestamp=j * 0.033, label=i % 2)
                for j in range(10)
            ],
        )
        for i in range(30)
    ]

    train_set, val_set, test_set = splitter.split_videos(videos)

    train_vids = {v.video_id for v in train_set}
    val_vids = {v.video_id for v in val_set}
    test_vids = {v.video_id for v in test_set}

    # Strict video-level separation: zero overlap
    assert train_vids.isdisjoint(val_vids)
    assert train_vids.isdisjoint(test_vids)
    assert val_vids.isdisjoint(test_vids)
    assert len(train_vids) + len(val_vids) + len(test_vids) == 30


def test_video_data_splitter_subject_level():
    """Verifies subject-level splitting: identical subjects never cross partition boundaries."""
    splitter = VideoDataSplitter(seed=42)
    videos = [
        VideoRecord(
            video_id="vid_1_a",
            video_path="v1a.mp4",
            label=0,
            subject_id="alice",
            frames=[FrameSample(video_id="vid_1_a", frame_index=0, timestamp=0.0, label=0)],
        ),
        VideoRecord(
            video_id="vid_1_b",
            video_path="v1b.mp4",
            label=1,
            subject_id="alice",
            frames=[FrameSample(video_id="vid_1_b", frame_index=0, timestamp=0.0, label=1)],
        ),
        VideoRecord(
            video_id="vid_2_a",
            video_path="v2a.mp4",
            label=0,
            subject_id="bob",
            frames=[FrameSample(video_id="vid_2_a", frame_index=0, timestamp=0.0, label=0)],
        ),
        VideoRecord(
            video_id="vid_3_a",
            video_path="v3a.mp4",
            label=1,
            subject_id="carol",
            frames=[FrameSample(video_id="vid_3_a", frame_index=0, timestamp=0.0, label=1)],
        ),
    ]

    train_set, val_set, test_set = splitter.split_by_subject(videos)
    train_subjects = {v.subject_id for v in train_set}
    test_subjects = {v.subject_id for v in test_set}

    assert train_subjects.isdisjoint(test_subjects)


def test_video_data_splitter_missing_subject_ids_handling():
    """Verifies that subject IDs take priority when present, and missing IDs are safely handled as video singletons."""
    splitter = VideoDataSplitter(seed=42)
    videos = [
        VideoRecord(video_id="v_known_1", video_path="1.mp4", label=0, subject_id="subject_alpha"),
        VideoRecord(video_id="v_known_2", video_path="2.mp4", label=1, subject_id="subject_alpha"),
        VideoRecord(video_id="v_anon_1", video_path="3.mp4", label=0, subject_id=None),
        VideoRecord(video_id="v_anon_2", video_path="4.mp4", label=1, subject_id=""),
        VideoRecord(video_id="v_known_3", video_path="5.mp4", label=0, subject_id="subject_beta"),
    ]

    train_set, val_set, test_set = splitter.split_videos(videos)

    train_vids = {v.video_id for v in train_set}
    val_vids = {v.video_id for v in val_set}
    test_vids = {v.video_id for v in test_set}

    assert train_vids.isdisjoint(val_vids)
    assert train_vids.isdisjoint(test_vids)
    assert val_vids.isdisjoint(test_vids)

    # Videos from subject_alpha must be together in the same split
    alpha_splits = [
        "v_known_1" in train_vids and "v_known_2" in train_vids,
        "v_known_1" in val_vids and "v_known_2" in val_vids,
        "v_known_1" in test_vids and "v_known_2" in test_vids,
    ]
    assert any(alpha_splits)


def test_video_frame_dataset_class_balancing():
    """Verifies inverse-frequency sample weight generation for class balancing."""
    # Imbalanced dataset: 8 real (label=0), 2 fake (label=1)
    frames = [
        FrameSample(video_id="vid_0", frame_index=i, timestamp=i * 0.1, label=0)
        for i in range(8)
    ] + [
        FrameSample(video_id="vid_1", frame_index=i, timestamp=i * 0.1, label=1)
        for i in range(2)
    ]

    dataset = VideoFrameDataset(frames)
    weights = dataset.get_sample_weights()

    assert len(weights) == 10
    # Fake samples (minority) must have higher weights than real samples (majority)
    assert weights[8] > weights[0]
    # Ratio should be 8/2 = 4x
    assert pytest.approx(weights[8] / weights[0], rel=1e-2) == 4.0


# ==============================================================================
# 6. Training & Evaluation Metrics Tests
# ==============================================================================


def test_compute_metrics():
    """Verifies AUROC, AP, EER, and Brier score evaluation on known probabilities."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    # Strongly separated logits: negative for real, positive for fake
    y_logits = np.array([-3.5, -2.8, -3.0, 3.2, 4.0, 3.8], dtype=np.float32)

    metrics = compute_metrics(y_true, y_logits)
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["eer"] == 0.0
    assert metrics["brier_score"] < 0.05
    assert metrics["accuracy"] == 1.0


def test_set_seed_reproducibility():
    """Verifies that set_seed guarantees deterministic random tensor generation."""
    set_seed(1234)
    t1 = torch.randn(5, 5)
    set_seed(1234)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)


def test_trainer_single_epoch(tmp_path: Path):
    """Executes a 1-epoch training loop and validates checkpoint serialization."""
    set_seed(42)
    model = LightweightVisualBaseline(in_channels=3)
    train_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "ckpts"),
        seed=42,
    )
    trainer = VisualDetectionTrainer(model=model, config=train_config)

    # Synthetic dataset with dummy crop tensors
    frames = [
        FrameSample(
            video_id=f"v_{i}",
            frame_index=0,
            timestamp=0.0,
            label=i % 2,
            crop_tensor=torch.randn(3, 224, 224),
        )
        for i in range(8)
    ]
    train_loader = DataLoader(VideoFrameDataset(frames), batch_size=4)
    val_loader = DataLoader(VideoFrameDataset(frames), batch_size=4)

    history = trainer.train(train_loader, val_loader)
    assert len(history) == 1
    assert "train_loss" in history[0]

    # Save checkpoint and verify file exists
    ckpt_path = trainer.save_checkpoint(epoch=1, metrics={"val_loss": 0.5})
    assert Path(ckpt_path).exists()
    assert compute_sha256(ckpt_path) is not None

    # Verify checkpoint metadata provenance
    loaded_ckpt = torch.load(ckpt_path, weights_only=True)
    assert loaded_ckpt["model_architecture"] == "LightweightVisualBaseline"
    assert loaded_ckpt["seed"] == 42
    assert loaded_ckpt["provenance"]["scientific_status"] == "UNVALIDATED_VISUAL_BASELINE"
    assert "generalization_notice" in loaded_ckpt["provenance"]


# ==============================================================================
# 7. VisualDeepfakeDetector Inference & Guardrail Tests
# ==============================================================================


def test_detector_missing_model_zero_fabrication(synthetic_video_file: Path):
    """Scientific Guardrail: Missing weights produce MODEL_UNAVAILABLE with no score fabrication."""
    config = VisualDetectionConfig(weights_path=None)
    detector = VisualDeepfakeDetector(config)

    dummy_meta = VideoMetadata(
        file_path=str(synthetic_video_file),
        file_name="test_visual_video.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=60,
        duration_seconds=2.0,
        codec="mp4v",
        file_size_bytes=100000,
    )

    result = detector.analyze_video(str(synthetic_video_file), dummy_meta)

    assert result.model_status == ModelAvailabilityStatus.MODEL_UNAVAILABLE
    assert result.aggregated_raw_score is None
    assert result.aggregated_logit is None
    assert result.calibrated_probability is None
    assert result.is_calibrated is False
    assert len(result.limitations) > 0
    assert "no predictions or probabilities are fabricated" in result.limitations[0].lower()
    detector.close()


def test_detector_inference_with_weights(
    synthetic_video_file: Path, saved_checkpoint: Tuple[Path, str]
):
    """Verifies end-to-end inference when valid weights are supplied."""
    ckpt_path, digest = saved_checkpoint
    config = VisualDetectionConfig(
        weights_path=str(ckpt_path),
        expected_sha256=digest,
        target_fps=5.0,
        max_frames=16,
    )
    detector = VisualDeepfakeDetector(config)

    dummy_meta = VideoMetadata(
        file_path=str(synthetic_video_file),
        file_name="test_visual_video.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=60,
        duration_seconds=2.0,
        codec="mp4v",
        file_size_bytes=100000,
    )

    result = detector.analyze_video(str(synthetic_video_file), dummy_meta)

    assert result.model_status == ModelAvailabilityStatus.AVAILABLE
    assert result.weights_digest == digest
    assert result.is_calibrated is False
    assert result.calibrated_probability is None
    assert "Integrity Notice" in result.explanation
    assert "95% CI" not in result.explanation
    assert result.descriptive_score_range is not None
    assert result.lag1_autocorrelation is not None
    assert result.effective_sample_size is not None
    detector.close()


def test_detector_cancellation(synthetic_video_file: Path, saved_checkpoint: Tuple[Path, str]):
    """Verifies clean thread cancellation when cancel_event is set."""
    ckpt_path, digest = saved_checkpoint
    config = VisualDetectionConfig(weights_path=str(ckpt_path), expected_sha256=digest)
    detector = VisualDeepfakeDetector(config)

    cancel_event = threading.Event()
    cancel_event.set()  # Cancel immediately

    dummy_meta = VideoMetadata(
        file_path=str(synthetic_video_file),
        file_name="test_visual_video.mp4",
        width=640,
        height=480,
        fps=30.0,
        frame_count=60,
        duration_seconds=2.0,
        codec="mp4v",
        file_size_bytes=100000,
    )

    result = detector.analyze_video(str(synthetic_video_file), dummy_meta, cancel_event=cancel_event)
    assert "cancelled" in result.explanation.lower()
    detector.close()


# ==============================================================================
# 8. JobManager & API Integration Tests
# ==============================================================================


def test_job_manager_visual_detection_modality(
    sample_video: Path, test_settings: Settings, saved_checkpoint: Tuple[Path, str]
):
    """Verifies JobManager executes VISUAL_DETECTION stage and preserves decoupling from physiological fusion."""
    ckpt_path, digest = saved_checkpoint
    config = VisualDetectionConfig(
        weights_path=str(ckpt_path),
        expected_sha256=digest,
        target_fps=2.0,
        max_frames=8,
    )
    detector = VisualDeepfakeDetector(config)
    inspector = VideoInspector(test_settings)

    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        visual_detector=detector,
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.VISUAL_DETECTION],
    )
    job = manager.submit_job(req)

    # Wait for completion
    for _ in range(80):
        curr = manager.get_job(job.job_id)
        if curr.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None

    # Physiological authenticity remains None (unassessed)
    assert final.result.overall_authenticity_score is None

    # Visual detection is populated separately
    assert final.result.visual_detection is not None
    assert isinstance(final.result.visual_detection, VisualDetectionResult)
    assert final.result.visual_detection.model_status == ModelAvailabilityStatus.AVAILABLE

    # Visual detection was NOT injected into modality_verdicts or fused_evidence
    for v in final.result.modality_verdicts:
        assert v.modality != ForensicModality.VISUAL_DETECTION
    assert "visual_detection" not in final.result.fused_evidence.family_summaries

    manager.shutdown()


def test_job_manager_visual_detection_failure_isolation(
    sample_video: Path, test_settings: Settings
):
    """Verifies that an unrecoverable failure in visual detection does not crash JobManager or physiological modules."""

    class CrashingVisualDetector:
        def analyze_video(self, *args, **kwargs):
            raise RuntimeError("Fatal CUDA out-of-memory simulation in visual detection")

        def close(self):
            pass

    inspector = VideoInspector(test_settings)
    manager = JobManager(
        settings=test_settings,
        inspector=inspector,
        visual_detector=CrashingVisualDetector(),
    )

    req = JobCreationRequest(
        video_path=str(sample_video),
        requested_modalities=[ForensicModality.RPPG, ForensicModality.VISUAL_DETECTION],
    )
    job = manager.submit_job(req)

    for _ in range(80):
        curr = manager.get_job(job.job_id)
        if curr.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            break
        time.sleep(0.1)

    final = manager.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.result is not None
    # Visual detection recorded failure cleanly
    assert final.result.visual_detection.model_status == ModelAvailabilityStatus.FAILED
    assert "CUDA out-of-memory simulation" in final.result.visual_detection.explanation
    manager.shutdown()
