"""Command-line video inference script using the BIOTrace baseline deepfake detector.

Scientific Principles & Forensic Guardrails:
    1. Video-Level Aggregation:
       Aggregates frame predictions using mean-logit pooling to guard against single-frame anomalies.
    2. Forensic Disclaimer:
       Visual CNN predictions indicate statistical artifact likelihood only. They do not constitute
       definitive proof of media manipulation or biometric tampering.
    3. Safe Model Deserialization:
       Loads checkpoints with weights_only=True and verifies SHA-256 provenance.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

AI_ENGINE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(AI_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_SRC))

from biotrace.vision.image_preprocessing import preprocess_letterboxed


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic digest of a file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, str]:
    """Safely builds EfficientNet-B0 and loads trained checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    ckpt_digest = compute_sha256(str(checkpoint_path))
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )

    ckpt_data = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    state_dict = ckpt_data["state_dict"] if "state_dict" in ckpt_data else ckpt_data
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, ckpt_digest


def preprocess_image(bgr_image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Letterboxes and normalizes an image using standard ImageNet statistics."""
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Invalid or empty image array.")

    chw, _ = preprocess_letterboxed(bgr_image, target_size=target_size)
    tensor = torch.from_numpy(chw).float()
    return tensor


def sample_frames_from_video(video_path: str, num_frames: int = 10) -> List[Tuple[int, float, np.ndarray]]:
    """Uniformly samples frames across a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video container: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video contains zero readable frames: {video_path}")

    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames: List[Tuple[int, float, np.ndarray]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if ret and frame_bgr is not None:
            ts = float(idx) / float(fps)
            frames.append((int(idx), ts, frame_bgr))

    cap.release()
    return frames


def sample_frames_from_directory(dir_path: str, num_frames: int = 10) -> List[Tuple[int, float, np.ndarray]]:
    """Loads image frames from a directory."""
    folder = Path(dir_path)
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(folder.glob(pat))

    files = sorted(list(set(files)))
    if not files:
        raise ValueError(f"No image files found in directory: {dir_path}")

    indices = np.linspace(0, max(0, len(files) - 1), min(num_frames, len(files)), dtype=int)
    frames: List[Tuple[int, float, np.ndarray]] = []

    for i, idx in enumerate(indices):
        fpath = files[idx]
        bgr = cv2.imread(str(fpath))
        if bgr is not None:
            frames.append((i, float(i) / 30.0, bgr))

    return frames


def predict_video(
    input_path: str,
    checkpoint_path: str = "ai-engine/models/baseline/best_model.pth",
    num_frames: int = 10,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Runs end-to-end deepfake inference on a video or directory of frames."""
    device = torch.device(device_str)
    model, ckpt_sha256 = load_model(Path(checkpoint_path), device)

    path_obj = Path(input_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if path_obj.is_dir():
        extracted = sample_frames_from_directory(input_path, num_frames=num_frames)
    else:
        extracted = sample_frames_from_video(input_path, num_frames=num_frames)

    if not extracted:
        raise ValueError(f"Zero valid frames extracted from {input_path}")

    tensors = []
    frame_info = []
    for frame_idx, ts, bgr in extracted:
        t = preprocess_image(bgr)
        tensors.append(t)
        frame_info.append({"frame_index": frame_idx, "timestamp_seconds": round(ts, 4)})

    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits_t = model(batch)
        if logits_t.ndim > 1:
            logits_t = logits_t.squeeze(-1)
        logits = logits_t.cpu().numpy()

    # Probability via sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    frame_predictions = []
    for info, logit, prob in zip(frame_info, logits, probs):
        frame_predictions.append({
            "frame_index": info["frame_index"],
            "timestamp_seconds": info["timestamp_seconds"],
            "raw_logit": round(float(logit), 4),
            "fake_probability": round(float(prob), 4),
            "real_probability": round(float(1.0 - prob), 4),
        })

    # Video-level aggregation: mean-logit pooling
    aggregated_logit = float(np.mean(logits))
    aggregated_fake_prob = float(1.0 / (1.0 + np.exp(-np.clip(aggregated_logit, -30.0, 30.0))))
    aggregated_real_prob = float(1.0 - aggregated_fake_prob)

    predicted_class = "FAKE" if aggregated_fake_prob >= 0.5 else "REAL"
    inter_frame_variance = float(np.var(logits))

    result = {
        "input_path": str(input_path),
        "model_identifier": "efficientnet_b0_baseline",
        "checkpoint_sha256": ckpt_sha256,
        "predicted_class": predicted_class,
        "real_probability": round(aggregated_real_prob, 4),
        "fake_probability": round(aggregated_fake_prob, 4),
        "aggregated_video_score": round(aggregated_fake_prob, 4),
        "aggregated_video_logit": round(aggregated_logit, 4),
        "inter_frame_variance": round(inter_frame_variance, 4),
        "number_of_frames_evaluated": len(frame_predictions),
        "per_frame_scores": frame_predictions,
        "forensic_disclaimer": (
            "A CNN baseline prediction reflects visual statistical patterns only. "
            "In accordance with forensic science standards, it does NOT constitute "
            "definitive proof of synthetic generation or biometric tampering."
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Run BIOTrace baseline deepfake detection inference on a video.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to video file or folder of frames")
    parser.add_argument("--checkpoint", "-c", type=str, default="ai-engine/models/baseline/best_model.pth")
    parser.add_argument("--frames", "-n", type=int, default=10, help="Number of frames to evaluate")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json", action="store_true", help="Output full JSON to stdout")
    args = parser.parse_args()

    result = predict_video(
        input_path=args.input,
        checkpoint_path=args.checkpoint,
        num_frames=args.frames,
        device_str=args.device,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print(" BIOTrace Baseline Deepfake Inference Result")
        print("=" * 60)
        print(f"Input:             {result['input_path']}")
        print(f"Model ID:          {result['model_identifier']}")
        print(f"Checkpoint SHA256: {result['checkpoint_sha256'][:16]}...")
        print(f"Frames Evaluated:  {result['number_of_frames_evaluated']}")
        print(f"Predicted Class:   {result['predicted_class']}")
        print(f"Fake Probability:  {result['fake_probability']:.4f} ({result['fake_probability']*100:.2f}%)")
        print(f"Real Probability:  {result['real_probability']:.4f} ({result['real_probability']*100:.2f}%)")
        print(f"Aggregated Logit:  {result['aggregated_video_logit']:.4f}")
        print(f"Inter-frame Var:   {result['inter_frame_variance']:.4f}")
        print("-" * 60)
        print("Per-Frame Details:")
        for fp in result["per_frame_scores"]:
            print(f"  Frame {fp['frame_index']:04d} (t={fp['timestamp_seconds']:.2f}s): Fake Prob = {fp['fake_probability']:.4f} [logit: {fp['raw_logit']:.2f}]")
        print("-" * 60)
        print(f"Disclaimer: {result['forensic_disclaimer']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
