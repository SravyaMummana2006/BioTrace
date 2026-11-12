"""Layout-preserving preprocessing and evidence-based verdict tests."""

import numpy as np
import torch

from biotrace.multimodal.multimodal_service import (
    MultimodalFusionResult,
    PhysiologicalAnalysisResult,
    derive_overall_assessment,
)
from biotrace.vision.image_preprocessing import letterbox_bgr, preprocess_letterboxed
from biotrace.visual_detection.baseline_service import (
    BaselineInferenceResult,
    BaselineInferenceService,
)


def _visual(prediction: str, fake_score: float = 0.5, status: str = "COMPLETED",
            error: str | None = None, non_square: bool = False) -> BaselineInferenceResult:
    return BaselineInferenceResult(
        video_path="clip.mp4",
        non_square_input=non_square,
        status=status,
        error_message=error,
        prediction=prediction,
        predicted_class=prediction,
        real_score=1.0 - fake_score,
        fake_score=fake_score,
        raw_sigmoid_score=fake_score,
        mean_logit_score=0.0,
        aggregated_video_score=fake_score,
        sampled_frames_count=10 if status == "COMPLETED" else 0,
        number_of_frames_evaluated=10 if status == "COMPLETED" else 0,
        checkpoint_sha256="0" * 64,
        inter_frame_variance=0.0,
    )


def _phys(prediction: str = "UNASSESSED", fake_score: float | None = None,
          status: str = "UNAVAILABLE") -> PhysiologicalAnalysisResult:
    return PhysiologicalAnalysisResult(
        status=status,
        prediction=prediction,
        fake_score=fake_score,
        real_score=None if fake_score is None else 1.0 - fake_score,
    )


def _fusion(prediction: str = "UNAVAILABLE", score: float | None = None,
            status: str = "UNAVAILABLE") -> MultimodalFusionResult:
    return MultimodalFusionResult(
        status=status,
        prediction=prediction,
        fusion_score=score,
        operating_threshold=0.5,
        uncertainty_level="DEFINITIVE" if status == "COMPLETED" else "UNASSESSED",
    )


def test_letterbox_portrait_preserves_aspect_ratio():
    image = np.zeros((1920, 864, 3), dtype=np.uint8)
    boxed, meta = letterbox_bgr(image, target_size=(224, 224))
    assert boxed.shape == (224, 224, 3)
    assert meta["content_width"] == 101
    assert meta["content_height"] == 224
    assert meta["pad_x"] == 61
    assert meta["pad_y"] == 0
    assert abs(meta["content_width"] / 864 - meta["content_height"] / 1920) < 0.01


def test_letterbox_landscape_preserves_aspect_ratio():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxed, meta = letterbox_bgr(image, target_size=(224, 224))
    assert boxed.shape == (224, 224, 3)
    assert meta["content_width"] == 224
    assert meta["content_height"] == 126
    assert meta["pad_x"] == 0
    assert meta["pad_y"] == 49
    assert abs(meta["content_width"] / 1920 - meta["content_height"] / 1080) < 0.01


def test_letterbox_square_is_unchanged():
    image = np.zeros((1080, 1080, 3), dtype=np.uint8)
    _, meta = letterbox_bgr(image, target_size=(224, 224))
    assert meta["content_width"] == 224
    assert meta["content_height"] == 224
    assert meta["pad_x"] == 0
    assert meta["pad_y"] == 0


def test_preprocess_letterboxed_output_contract():
    image = np.full((1920, 864, 3), 127, dtype=np.uint8)
    chw, meta = preprocess_letterboxed(image, target_size=(224, 224))
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    assert np.isfinite(chw).all()
    tensor = BaselineInferenceService.preprocess_frame(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert meta["pad_x"] > 0


def test_fusion_verdict_wins_when_completed():
    verdict, confidence, basis, reason = derive_overall_assessment(
        _visual("REAL", 0.08),
        _phys("REAL", 0.2, "COMPLETED"),
        _fusion("FAKE", 0.91, "COMPLETED"),
    )
    assert verdict == "FAKE"
    assert basis == "fusion"
    assert confidence > 0.5
    assert "Fusion" in reason


def test_agreeing_visual_and_physiological_are_assessed_without_fusion():
    verdict, _, basis, reason = derive_overall_assessment(
        _visual("REAL", 0.1), _phys("REAL", 0.2, "COMPLETED"), _fusion()
    )
    assert verdict == "REAL"
    assert basis == "visual_phys_agreement"
    assert "agree" in reason.lower()


def test_conflicting_models_are_inconclusive():
    verdict, confidence, basis, reason = derive_overall_assessment(
        _visual("FAKE", 0.9), _phys("REAL", 0.2, "COMPLETED"), _fusion()
    )
    assert verdict == "INCONCLUSIVE"
    assert basis == "inconclusive"
    assert confidence == 0.0
    assert "conflict" in reason.lower()


def test_visual_only_result_is_assessed():
    verdict, _, basis, reason = derive_overall_assessment(
        _visual("FAKE", 0.9), _phys(), _fusion()
    )
    assert verdict == "FAKE"
    assert basis == "visual_only"
    assert "visual-only" in reason.lower()


def test_aspect_ratio_is_not_a_verdict_gate():
    verdict, _, basis, _ = derive_overall_assessment(
        _visual("REAL", 0.088, non_square=True),
        _phys("REAL", 0.2, "COMPLETED"),
        _fusion("REAL", 0.006, "COMPLETED"),
    )
    assert verdict == "REAL"
    assert basis == "fusion"


def test_failed_visual_model_yields_concrete_unassessed_reason():
    verdict, confidence, basis, reason = derive_overall_assessment(
        _visual("UNAVAILABLE", 0.0, status="FAILED", error="Video frames could not be decoded"),
        _phys(),
        _fusion(),
    )
    assert verdict == "UNASSESSED"
    assert basis == "unassessed"
    assert confidence == 0.0
    assert "could not be decoded" in reason
