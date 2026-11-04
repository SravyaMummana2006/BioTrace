"""BIOTrace Phase 8 — Physiological Feature Feasibility Extraction Script.

Extracts rPPG, eye-blink dynamics, facial kinematics, and spatial biological consistency
features across a balanced sample of 100 real and 100 fake videos from the HiDF
training partition under Protocol C.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Add ai-engine/src to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ENGINE_SRC = PROJECT_ROOT / "ai-engine" / "src"
if str(AI_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_SRC))

from biotrace.core.config import Settings
from biotrace.domain.models import VideoMetadata
from biotrace.facial_dynamics.ear import calculate_ear_velocity
from biotrace.facial_dynamics.extractor import FacialDynamicsExtractor
from biotrace.facial_dynamics.landmarks import EyeLandmarkTracker
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import FilterConfig, QualityConfig, WindowConfig
from biotrace.rppg.face_tracker import LandmarkFaceTracker
from biotrace.rppg.quality import assess_signal_quality
from biotrace.rppg.signal_extractor import extract_raw_rgb_signals
from biotrace.services.video_inspector import VideoInspector
from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    SpatialConsistencyConfig,
)
from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor
from biotrace.spatial_consistency.quality import assess_regional_signal
from biotrace.spatial_consistency.spatial_analyzer import (
    compute_lagged_cross_correlation,
    compute_spectral_overlap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase8Feasibility")

FROZEN_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


def verify_manifest_hash(manifest_path: str) -> None:
    """Verifies that protocol_c_split_manifest.csv is completely untampered."""
    with open(manifest_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA-256 mismatch! Expected {FROZEN_MANIFEST_SHA256}, got {actual_hash}"
        )
    logger.info("Frozen Protocol C split manifest SHA-256 verified OK: %s", actual_hash[:16])


def select_sample_videos(
    manifest_path: str,
    real_dir: str,
    fake_dir: str,
    num_pairs: int = 100,
) -> List[Dict[str, Any]]:
    """Selects a balanced sample of real and fake videos strictly from the train split."""
    verify_manifest_hash(manifest_path)
    df = pd.read_csv(manifest_path)

    # STRICT CONSTRAINTS: Only train partition, never validation or test
    train_df = df[df["split"] == "train"]
    if len(train_df) < num_pairs:
        raise ValueError(
            f"Requested {num_pairs} pairs, but train split only has {len(train_df)} rows"
        )

    sampled_pairs = train_df.iloc[:num_pairs]
    samples: List[Dict[str, Any]] = []

    # 1. Collect real videos (100)
    for _, row in sampled_pairs.iterrows():
        b_id = str(row["base_id"])
        real_file = os.path.join(real_dir, f"{b_id}.mp4")
        samples.append({
            "video_id": b_id,
            "base_id": b_id,
            "target_id": "",
            "file_path": real_file,
            "split": "train",
            "label": "real",
            "ground_truth_class": 0,
            "component_id": str(row["component_id"]),
            "race": str(row["race"]),
            "gender": str(row["gender"]),
            "age": str(row["age"]),
        })

    # 2. Collect fake videos (100)
    for _, row in sampled_pairs.iterrows():
        b_id = str(row["base_id"])
        t_id = str(row["target_id"])
        fake_file = os.path.join(fake_dir, f"{b_id}_{t_id}.mp4")
        samples.append({
            "video_id": f"{b_id}_{t_id}",
            "base_id": b_id,
            "target_id": t_id,
            "file_path": fake_file,
            "split": "train",
            "label": "fake",
            "ground_truth_class": 1,
            "component_id": str(row["component_id"]),
            "race": str(row["race"]),
            "gender": str(row["gender"]),
            "age": str(row["age"]),
        })

    # Pre-verification: all 200 files must exist and be readable
    for s in samples:
        p = s["file_path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Selected sample video not found: {p}")
        if os.path.getsize(p) == 0:
            raise ValueError(f"Selected sample video is empty: {p}")

    logger.info(
        "Successfully selected %d videos (%d real, %d fake) from train partition. All exist on disk.",
        len(samples),
        num_pairs,
        num_pairs,
    )
    return samples


def extract_features_single_video(
    sample_info: Dict[str, Any],
    inspector: VideoInspector,
    rppg_window_cfg: WindowConfig,
    rppg_filter_cfg: FilterConfig,
    rppg_quality_cfg: QualityConfig,
    spatial_cfg: SpatialConsistencyConfig,
) -> Dict[str, Any]:
    """Runs rPPG, facial dynamics, and spatial consistency extractors on a single video."""
    video_path = sample_info["file_path"]
    t_start = time.perf_counter()

    row: Dict[str, Any] = {
        # Identifiers & Split Metadata
        "video_id": sample_info["video_id"],
        "base_id": sample_info["base_id"],
        "target_id": sample_info["target_id"],
        "file_path": video_path,
        "split": sample_info["split"],
        "label": sample_info["label"],
        "ground_truth_class": sample_info["ground_truth_class"],
        "component_id": sample_info["component_id"],
        "race": sample_info["race"],
        "gender": sample_info["gender"],
        "age": sample_info["age"],
        # Defaults for container
        "video_duration_s": np.nan,
        "video_fps": np.nan,
        "video_frame_count": np.nan,
        "video_width": np.nan,
        "video_height": np.nan,
        "file_size_bytes": np.nan,
        # Defaults for rPPG
        "rppg_extractor_success": 0,
        "rppg_is_valid_pulse": 0,
        "rppg_confidence": np.nan,
        "rppg_quality_grade": "UNPROCESSED",
        "rppg_quality_flags": "",
        "rppg_valid_frame_ratio": np.nan,
        "rppg_dropped_frames": np.nan,
        "rppg_duplicate_frames": np.nan,
        "rppg_pos_snr_db": np.nan,
        "rppg_pos_hr_bpm": np.nan,
        "rppg_pos_peak_freq_hz": np.nan,
        "rppg_pos_peak_prominence": np.nan,
        "rppg_chrom_snr_db": np.nan,
        "rppg_chrom_hr_bpm": np.nan,
        "rppg_chrom_peak_freq_hz": np.nan,
        "rppg_chrom_peak_prominence": np.nan,
        "rppg_pos_chrom_corr": np.nan,
        "rppg_exec_time_s": np.nan,
        # Defaults for Facial Dynamics & Blink
        "facial_extractor_success": 0,
        "face_detected_ratio": np.nan,
        "landmark_valid_ratio": np.nan,
        "ear_left_mean": np.nan,
        "ear_left_std": np.nan,
        "ear_left_min": np.nan,
        "ear_left_max": np.nan,
        "ear_right_mean": np.nan,
        "ear_right_std": np.nan,
        "ear_right_min": np.nan,
        "ear_right_max": np.nan,
        "ear_bino_mean": np.nan,
        "ear_bino_std": np.nan,
        "ear_bino_min": np.nan,
        "ear_bino_max": np.nan,
        "ear_velocity_mean": np.nan,
        "ear_velocity_max": np.nan,
        "ear_closing_vel_max": np.nan,
        "ear_opening_vel_max": np.nan,
        "blink_asymmetry_mean": np.nan,
        "blink_asymmetry_std": np.nan,
        "head_pose_yaw_mean": np.nan,
        "head_pose_yaw_std": np.nan,
        "head_pose_pitch_mean": np.nan,
        "head_pose_pitch_std": np.nan,
        "head_pose_roll_mean": np.nan,
        "head_pose_roll_std": np.nan,
        "head_pose_acceptable_ratio": np.nan,
        "blink_calibration_state": "UNCALIBRATED",
        "blink_baseline_ear": np.nan,
        "blink_closure_threshold": np.nan,
        "blink_reopening_threshold": np.nan,
        "blink_total_count": np.nan,
        "blink_bilateral_count": np.nan,
        "blink_rate_bpm": np.nan,
        "blink_mean_duration_ms": np.nan,
        "blink_median_duration_ms": np.nan,
        "blink_duration_iqr_ms": np.nan,
        "blink_short_uncertain_count": np.nan,
        "blink_low_res_candidate_count": np.nan,
        "blink_prolonged_closure_count": np.nan,
        "blink_rejected_artifact_count": np.nan,
        "blink_tracking_quality": np.nan,
        "facial_exec_time_s": np.nan,
        # Defaults for Spatial Consistency
        "spatial_extractor_success": 0,
        "spatial_is_measurable": 0,
        "spatial_global_state": "UNPROCESSED",
        "spatial_confidence": np.nan,
        "spatial_snr_forehead": np.nan,
        "spatial_snr_left_cheek": np.nan,
        "spatial_snr_right_cheek": np.nan,
        "spatial_bpm_forehead": np.nan,
        "spatial_bpm_left_cheek": np.nan,
        "spatial_bpm_right_cheek": np.nan,
        "spatial_cheek_zero_lag_corr": np.nan,
        "spatial_cheek_max_cross_corr": np.nan,
        "spatial_cheek_lag_seconds": np.nan,
        "spatial_cheek_spectral_overlap": np.nan,
        "spatial_forehead_left_corr": np.nan,
        "spatial_forehead_right_corr": np.nan,
        "spatial_forehead_left_spectral_overlap": np.nan,
        "spatial_forehead_right_spectral_overlap": np.nan,
        "spatial_cheek_bpm_diff": np.nan,
        "spatial_exec_time_s": np.nan,
        # Defaults for Overall Execution
        "total_wall_time_s": np.nan,
        "error_message": "",
    }

    try:
        # Container Inspection
        meta = inspector.inspect_video(video_path)
        row["video_duration_s"] = round(float(meta.duration_seconds), 3)
        row["video_fps"] = round(float(meta.fps), 2)
        row["video_frame_count"] = int(meta.frame_count)
        row["video_width"] = int(meta.width)
        row["video_height"] = int(meta.height)
        row["file_size_bytes"] = int(meta.file_size_bytes)
    except Exception as e:
        row["error_message"] = f"Container error: {e}"
        row["total_wall_time_s"] = round(time.perf_counter() - t_start, 3)
        return row

    # 1. rPPG Extraction
    t0_rppg = time.perf_counter()
    face_tracker = LandmarkFaceTracker()
    try:
        tracking_result = face_tracker.track_video_rois(video_path)
        row["rppg_valid_frame_ratio"] = round(float(tracking_result.valid_frame_ratio), 4)

        if tracking_result.is_landmark_tracking_reliable:
            raw_signals = extract_raw_rgb_signals(
                video_path, tracking_result, target_fps=meta.fps
            )
            if raw_signals is not None and len(raw_signals.combined_rgb) >= 10:
                fs = raw_signals.resampled_fps
                rgb_trace = raw_signals.combined_rgb
                row["rppg_dropped_frames"] = int(raw_signals.dropped_frames_detected)
                row["rppg_duplicate_frames"] = int(raw_signals.duplicate_frames_detected)

                # POS extraction
                pos_signal = extract_pos_rppg(
                    rgb_trace,
                    sampling_rate_hz=fs,
                    window_config=rppg_window_cfg,
                    filter_config=rppg_filter_cfg,
                )
                q_pos, p_pos = assess_signal_quality(
                    signal=pos_signal,
                    sampling_rate_hz=fs,
                    total_frames=tracking_result.total_frames,
                    valid_frames=tracking_result.valid_frames,
                    duration_seconds=meta.duration_seconds,
                    config=rppg_quality_cfg,
                )

                # CHROM extraction
                chrom_signal = extract_chrom_rppg(
                    rgb_trace,
                    sampling_rate_hz=fs,
                    window_config=rppg_window_cfg,
                    filter_config=rppg_filter_cfg,
                )
                q_chrom, p_chrom = assess_signal_quality(
                    signal=chrom_signal,
                    sampling_rate_hz=fs,
                    total_frames=tracking_result.total_frames,
                    valid_frames=tracking_result.valid_frames,
                    duration_seconds=meta.duration_seconds,
                    config=rppg_quality_cfg,
                )

                row["rppg_extractor_success"] = 1
                row["rppg_is_valid_pulse"] = 1 if p_pos.is_valid else 0
                row["rppg_confidence"] = round(float(p_pos.confidence), 4)
                row["rppg_quality_grade"] = str(q_pos.grade.value)
                row["rppg_quality_flags"] = "; ".join(f.value for f in q_pos.flags)
                row["rppg_pos_snr_db"] = round(float(q_pos.snr_db), 3)
                row["rppg_pos_peak_freq_hz"] = (
                    round(float(p_pos.peak_frequency_hz), 3)
                    if p_pos.peak_frequency_hz is not None
                    else np.nan
                )
                row["rppg_pos_hr_bpm"] = (
                    round(float(p_pos.heart_rate_bpm), 1)
                    if p_pos.heart_rate_bpm is not None
                    else (
                        round(float(p_pos.peak_frequency_hz * 60.0), 1)
                        if p_pos.peak_frequency_hz is not None
                        else np.nan
                    )
                )
                row["rppg_pos_peak_prominence"] = (
                    round(float(p_pos.peak_prominence), 4)
                    if p_pos.peak_prominence is not None
                    else np.nan
                )

                row["rppg_chrom_snr_db"] = round(float(q_chrom.snr_db), 3)
                row["rppg_chrom_peak_freq_hz"] = (
                    round(float(p_chrom.peak_frequency_hz), 3)
                    if p_chrom.peak_frequency_hz is not None
                    else np.nan
                )
                row["rppg_chrom_hr_bpm"] = (
                    round(float(p_chrom.heart_rate_bpm), 1)
                    if p_chrom.heart_rate_bpm is not None
                    else (
                        round(float(p_chrom.peak_frequency_hz * 60.0), 1)
                        if p_chrom.peak_frequency_hz is not None
                        else np.nan
                    )
                )
                row["rppg_chrom_peak_prominence"] = (
                    round(float(p_chrom.peak_prominence), 4)
                    if p_chrom.peak_prominence is not None
                    else np.nan
                )

                if (
                    len(pos_signal) > 5
                    and len(chrom_signal) == len(pos_signal)
                    and np.std(pos_signal) > 1e-7
                    and np.std(chrom_signal) > 1e-7
                ):
                    corr = float(np.corrcoef(pos_signal, chrom_signal)[0, 1])
                    row["rppg_pos_chrom_corr"] = round(corr, 4)
            else:
                row["rppg_quality_grade"] = "INSUFFICIENT_SIGNAL"
                row["rppg_quality_flags"] = "INSUFFICIENT_SKIN_PIXELS"
        else:
            row["rppg_quality_grade"] = "INSUFFICIENT_SIGNAL"
            row["rppg_quality_flags"] = "TRACKING_DEGRADED"
    except Exception as e:
        row["error_message"] += f" [rPPG err: {e}]"
        logger.warning("rPPG failed on %s: %s", video_path, e)
    finally:
        face_tracker.close()
        row["rppg_exec_time_s"] = round(time.perf_counter() - t0_rppg, 3)

    # 2. Facial Dynamics & Eye-Blink Extraction
    t0_facial = time.perf_counter()
    eye_tracker = EyeLandmarkTracker()
    try:
        samples = eye_tracker.track_video_stream(video_path)
        n_samples = len(samples)
        if n_samples > 0:
            face_detected_count = sum(1 for s in samples if s.is_face_detected)
            valid_lm_count = sum(1 for s in samples if s.is_landmark_valid)
            row["face_detected_ratio"] = round(face_detected_count / n_samples, 4)
            row["landmark_valid_ratio"] = round(valid_lm_count / n_samples, 4)

            # EAR statistics
            left_ears = [s.ear_left for s in samples if s.ear_left is not None and np.isfinite(s.ear_left)]
            right_ears = [s.ear_right for s in samples if s.ear_right is not None and np.isfinite(s.ear_right)]
            bino_ears = [s.ear_bino for s in samples if s.ear_bino is not None and np.isfinite(s.ear_bino)]

            if left_ears:
                row["ear_left_mean"] = round(float(np.mean(left_ears)), 4)
                row["ear_left_std"] = round(float(np.std(left_ears)), 4)
                row["ear_left_min"] = round(float(np.min(left_ears)), 4)
                row["ear_left_max"] = round(float(np.max(left_ears)), 4)

            if right_ears:
                row["ear_right_mean"] = round(float(np.mean(right_ears)), 4)
                row["ear_right_std"] = round(float(np.std(right_ears)), 4)
                row["ear_right_min"] = round(float(np.min(right_ears)), 4)
                row["ear_right_max"] = round(float(np.max(right_ears)), 4)

            if bino_ears:
                row["ear_bino_mean"] = round(float(np.mean(bino_ears)), 4)
                row["ear_bino_std"] = round(float(np.std(bino_ears)), 4)
                row["ear_bino_min"] = round(float(np.min(bino_ears)), 4)
                row["ear_bino_max"] = round(float(np.max(bino_ears)), 4)

            # Eyelid velocity & Asymmetry
            valid_samples = [s for s in samples if s.is_landmark_valid and s.ear_bino is not None]
            if len(valid_samples) > 2:
                dt_list = []
                d_ear_list = []
                asym_list = []
                for i in range(1, len(valid_samples)):
                    s_curr = valid_samples[i]
                    s_prev = valid_samples[i - 1]
                    dt = s_curr.timestamp_seconds - s_prev.timestamp_seconds
                    if 0.001 < dt < 0.2:
                        de = s_curr.ear_bino - s_prev.ear_bino
                        dt_list.append(dt)
                        d_ear_list.append(de / dt)
                    if s_curr.ear_left is not None and s_curr.ear_right is not None:
                        denom = s_curr.ear_left + s_curr.ear_right
                        if denom > 1e-4:
                            asym = abs(s_curr.ear_left - s_curr.ear_right) / denom
                            asym_list.append(asym)

                if d_ear_list:
                    abs_vel = np.abs(d_ear_list)
                    row["ear_velocity_mean"] = round(float(np.mean(abs_vel)), 4)
                    row["ear_velocity_max"] = round(float(np.max(abs_vel)), 4)
                    neg_vel = [-v for v in d_ear_list if v < 0]
                    pos_vel = [v for v in d_ear_list if v > 0]
                    row["ear_closing_vel_max"] = round(float(np.max(neg_vel)), 4) if neg_vel else np.nan
                    row["ear_opening_vel_max"] = round(float(np.max(pos_vel)), 4) if pos_vel else np.nan

                if asym_list:
                    row["blink_asymmetry_mean"] = round(float(np.mean(asym_list)), 4)
                    row["blink_asymmetry_std"] = round(float(np.std(asym_list)), 4)

            # Head Pose
            poses = [s.head_pose for s in samples if s.head_pose is not None]
            if poses:
                yaws = [p.yaw_deg for p in poses]
                pitches = [p.pitch_deg for p in poses]
                rolls = [p.roll_deg for p in poses]
                acc_poses = sum(1 for p in poses if p.is_pose_acceptable)
                row["head_pose_yaw_mean"] = round(float(np.mean(yaws)), 2)
                row["head_pose_yaw_std"] = round(float(np.std(yaws)), 2)
                row["head_pose_pitch_mean"] = round(float(np.mean(pitches)), 2)
                row["head_pose_pitch_std"] = round(float(np.std(pitches)), 2)
                row["head_pose_roll_mean"] = round(float(np.mean(rolls)), 2)
                row["head_pose_roll_std"] = round(float(np.std(rolls)), 2)
                row["head_pose_acceptable_ratio"] = round(acc_poses / len(poses), 4)

            # Standard Facial Dynamics Extractor execution
            facial_extractor = FacialDynamicsExtractor()
            verdict_facial = facial_extractor.extract_from_samples(samples, meta)
            row["facial_extractor_success"] = 1
            if verdict_facial and verdict_facial.raw_signal and verdict_facial.raw_signal.metadata:
                f_meta = verdict_facial.raw_signal.metadata
                row["blink_calibration_state"] = str(f_meta.get("calibration_state", "UNCALIBRATED"))
                row["blink_baseline_ear"] = (
                    round(float(f_meta["baseline_bino"]), 4)
                    if f_meta.get("baseline_bino") is not None
                    else np.nan
                )
                row["blink_closure_threshold"] = (
                    round(float(f_meta["closure_threshold_bino"]), 4)
                    if f_meta.get("closure_threshold_bino") is not None
                    else np.nan
                )
                row["blink_reopening_threshold"] = (
                    round(float(f_meta["reopening_threshold_bino"]), 4)
                    if f_meta.get("reopening_threshold_bino") is not None
                    else np.nan
                )
                row["blink_total_count"] = (
                    float(f_meta["total_complete_blinks"])
                    if f_meta.get("total_complete_blinks") is not None
                    else np.nan
                )
                row["blink_bilateral_count"] = (
                    float(f_meta["complete_bilateral_blinks"])
                    if f_meta.get("complete_bilateral_blinks") is not None
                    else np.nan
                )
                row["blink_rate_bpm"] = (
                    round(float(f_meta["blinks_per_minute"]), 2)
                    if f_meta.get("blinks_per_minute") is not None
                    else np.nan
                )
                row["blink_mean_duration_ms"] = (
                    round(float(f_meta["mean_complete_duration_ms"]), 2)
                    if f_meta.get("mean_complete_duration_ms") is not None
                    else np.nan
                )
                row["blink_median_duration_ms"] = (
                    round(float(f_meta["median_complete_duration_ms"]), 2)
                    if f_meta.get("median_complete_duration_ms") is not None
                    else np.nan
                )
                row["blink_duration_iqr_ms"] = (
                    round(float(f_meta["complete_duration_iqr_ms"]), 2)
                    if f_meta.get("complete_duration_iqr_ms") is not None
                    else np.nan
                )
                row["blink_short_uncertain_count"] = (
                    float(f_meta["short_duration_uncertain"])
                    if f_meta.get("short_duration_uncertain") is not None
                    else np.nan
                )
                row["blink_low_res_candidate_count"] = (
                    float(f_meta["low_resolution_candidates"])
                    if f_meta.get("low_resolution_candidates") is not None
                    else np.nan
                )
                row["blink_prolonged_closure_count"] = (
                    float(f_meta["prolonged_closures"])
                    if f_meta.get("prolonged_closures") is not None
                    else np.nan
                )
                row["blink_rejected_artifact_count"] = (
                    float(f_meta["rejected_artifacts"])
                    if f_meta.get("rejected_artifacts") is not None
                    else np.nan
                )
                row["blink_tracking_quality"] = (
                    round(float(f_meta["provisional_tracking_quality"]), 4)
                    if f_meta.get("provisional_tracking_quality") is not None
                    else np.nan
                )
    except Exception as e:
        row["error_message"] += f" [Facial err: {e}]"
        logger.warning("Facial dynamics failed on %s: %s", video_path, e)
    finally:
        eye_tracker.close()
        row["facial_exec_time_s"] = round(time.perf_counter() - t0_facial, 3)

    # 3. Spatial Biological Consistency Extraction
    t0_spatial = time.perf_counter()
    spatial_extractor = SpatialConsistencyExtractor(config=spatial_cfg)
    try:
        reg_samples, timing = spatial_extractor.tracker.track_video_regions(video_path)
        row["spatial_extractor_success"] = 1

        # Regional signal conditioning
        regional_summaries = {}
        regional_pulses = {}
        regional_timestamps = {}
        for r_id in FacialRegionId:
            samples_r = reg_samples.get(r_id, [])
            summary, ts_r, pulse_r = assess_regional_signal(
                samples=samples_r,
                region_id=r_id,
                config=spatial_cfg,
                reported_fps=meta.fps,
            )
            regional_summaries[r_id] = summary
            regional_pulses[r_id] = pulse_r
            regional_timestamps[r_id] = ts_r

        # Populate individual regional SNRs and BPMs
        s_fore = regional_summaries[FacialRegionId.FOREHEAD]
        s_left = regional_summaries[FacialRegionId.LEFT_CHEEK]
        s_right = regional_summaries[FacialRegionId.RIGHT_CHEEK]

        row["spatial_snr_forehead"] = round(float(s_fore.snr_db), 3)
        row["spatial_snr_left_cheek"] = round(float(s_left.snr_db), 3)
        row["spatial_snr_right_cheek"] = round(float(s_right.snr_db), 3)

        row["spatial_bpm_forehead"] = (
            round(float(s_fore.dominant_bpm), 1) if s_fore.dominant_bpm is not None else np.nan
        )
        row["spatial_bpm_left_cheek"] = (
            round(float(s_left.dominant_bpm), 1) if s_left.dominant_bpm is not None else np.nan
        )
        row["spatial_bpm_right_cheek"] = (
            round(float(s_right.dominant_bpm), 1) if s_right.dominant_bpm is not None else np.nan
        )

        p_fore = regional_pulses[FacialRegionId.FOREHEAD]
        p_left = regional_pulses[FacialRegionId.LEFT_CHEEK]
        p_right = regional_pulses[FacialRegionId.RIGHT_CHEEK]

        # Bilateral Cheek Comparisons
        if p_left is not None and p_right is not None and len(p_left) == len(p_right) and len(p_left) >= 10:
            z_corr, max_corr, lag_sec, _ = compute_lagged_cross_correlation(
                p_left, p_right, meta.fps, spatial_cfg.max_physiological_lag_seconds
            )
            row["spatial_cheek_zero_lag_corr"] = round(float(z_corr), 4)
            row["spatial_cheek_max_cross_corr"] = round(float(max_corr), 4)
            row["spatial_cheek_lag_seconds"] = round(float(lag_sec), 4)
            spec_overlap = compute_spectral_overlap(
                getattr(s_left, "psd", np.array([])),
                getattr(s_right, "psd", np.array([])),
            )
            row["spatial_cheek_spectral_overlap"] = round(float(spec_overlap), 4)
            if s_left.dominant_bpm is not None and s_right.dominant_bpm is not None:
                row["spatial_cheek_bpm_diff"] = round(
                    abs(float(s_left.dominant_bpm) - float(s_right.dominant_bpm)), 2
                )

        # Forehead vs. Left Cheek Comparisons
        if p_fore is not None and p_left is not None and len(p_fore) == len(p_left) and len(p_fore) >= 10:
            z_corr_fl, _, _, _ = compute_lagged_cross_correlation(
                p_fore, p_left, meta.fps, spatial_cfg.max_physiological_lag_seconds
            )
            row["spatial_forehead_left_corr"] = round(float(z_corr_fl), 4)
            spec_overlap_fl = compute_spectral_overlap(
                getattr(s_fore, "psd", np.array([])),
                getattr(s_left, "psd", np.array([])),
            )
            row["spatial_forehead_left_spectral_overlap"] = round(float(spec_overlap_fl), 4)

        # Forehead vs. Right Cheek Comparisons
        if p_fore is not None and p_right is not None and len(p_fore) == len(p_right) and len(p_fore) >= 10:
            z_corr_fr, _, _, _ = compute_lagged_cross_correlation(
                p_fore, p_right, meta.fps, spatial_cfg.max_physiological_lag_seconds
            )
            row["spatial_forehead_right_corr"] = round(float(z_corr_fr), 4)
            spec_overlap_fr = compute_spectral_overlap(
                getattr(s_fore, "psd", np.array([])),
                getattr(s_right, "psd", np.array([])),
            )
            row["spatial_forehead_right_spectral_overlap"] = round(float(spec_overlap_fr), 4)

        # Standard Spatial Consistency Analysis
        verdict_spatial = spatial_extractor.extract_from_samples(
            reg_samples, timing, meta
        )
        if verdict_spatial and verdict_spatial.raw_signal and verdict_spatial.raw_signal.metadata:
            s_meta = verdict_spatial.raw_signal.metadata
            row["spatial_global_state"] = str(s_meta.get("global_state", "UNPROCESSED"))
            row["spatial_confidence"] = round(float(verdict_spatial.confidence), 4)
            row["spatial_is_measurable"] = (
                1 if s_meta.get("global_state") not in ("insufficient_evidence", "UNPROCESSED") else 0
            )
        else:
            row["spatial_global_state"] = "INSUFFICIENT_EVIDENCE"
            row["spatial_confidence"] = 0.0
            row["spatial_is_measurable"] = 0
    except Exception as e:
        row["error_message"] += f" [Spatial err: {e}]"
        logger.warning("Spatial consistency failed on %s: %s", video_path, e)
    finally:
        spatial_extractor.close()
        row["spatial_exec_time_s"] = round(time.perf_counter() - t0_spatial, 3)

    row["total_wall_time_s"] = round(time.perf_counter() - t_start, 3)
    return row


def run_feasibility_extraction(
    manifest_path: str = "protocol_c_split_manifest.csv",
    real_dir: str = "data/extracted/hidf/real",
    fake_dir: str = "data/extracted/hidf/fake",
    output_csv: str = "phase8_sample_features.csv",
    num_pairs: int = 100,
) -> pd.DataFrame:
    """Executes the full Phase 8 feasibility extraction over the sampled dataset."""
    logger.info("Starting Phase 8 Feasibility Feature Extraction...")
    samples = select_sample_videos(
        manifest_path=manifest_path,
        real_dir=real_dir,
        fake_dir=fake_dir,
        num_pairs=num_pairs,
    )

    settings = Settings()
    inspector = VideoInspector(settings)
    rppg_window_cfg = WindowConfig()
    rppg_filter_cfg = FilterConfig()
    rppg_quality_cfg = QualityConfig()
    spatial_cfg = SpatialConsistencyConfig()

    rows: List[Dict[str, Any]] = []
    total_samples = len(samples)
    t_start_all = time.perf_counter()

    for idx, sample in enumerate(samples, 1):
        v_id = sample["video_id"]
        lbl = sample["label"]
        t0 = time.perf_counter()
        row = extract_features_single_video(
            sample_info=sample,
            inspector=inspector,
            rppg_window_cfg=rppg_window_cfg,
            rppg_filter_cfg=rppg_filter_cfg,
            rppg_quality_cfg=rppg_quality_cfg,
            spatial_cfg=spatial_cfg,
        )
        rows.append(row)
        dt = time.perf_counter() - t0

        if idx % 10 == 0 or idx == total_samples:
            elapsed = time.perf_counter() - t_start_all
            avg_per_vid = elapsed / idx
            eta = avg_per_vid * (total_samples - idx)
            logger.info(
                "[%3d/%3d] (%s) %s -> total_time: %.2fs | Avg: %.2fs/vid | Elapsed: %.1fs | ETA: %.1fs",
                idx,
                total_samples,
                lbl.upper(),
                v_id,
                dt,
                avg_per_vid,
                elapsed,
                eta,
            )

            # Periodically checkpoint progress to disk
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(output_csv, index=False)

    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_csv, index=False)
    total_elapsed = time.perf_counter() - t_start_all
    logger.info(
        "Feasibility extraction complete! Processed %d videos in %.2f seconds (%.2fs/video). Saved to %s",
        len(df_results),
        total_elapsed,
        total_elapsed / len(df_results),
        output_csv,
    )
    return df_results


def main():
    parser = argparse.ArgumentParser(description="BIOTrace Phase 8 Feasibility Extraction")
    parser.add_argument(
        "--manifest",
        type=str,
        default="protocol_c_split_manifest.csv",
        help="Path to Protocol C manifest",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default="data/extracted/hidf/real",
        help="Path to real videos directory",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        default="data/extracted/hidf/fake",
        help="Path to fake videos directory",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of training pairs to sample (each produces 1 real and 1 fake = 2x videos)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="phase8_sample_features.csv",
        help="Output CSV path for extracted features",
    )
    args = parser.parse_args()

    run_feasibility_extraction(
        manifest_path=args.manifest,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_csv=args.output_csv,
        num_pairs=args.num_pairs,
    )


if __name__ == "__main__":
    main()
