"""BIOTrace Phase 9 — Full Physiological Feature Extraction Script.

Extracts the approved 10-feature physiological and facial kinematic vector across all
8,722 videos in the HiDF dataset (train, validation, and test partitions) under Protocol C.
"""

import argparse
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

# Add ai-engine/src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI_ENGINE_SRC = PROJECT_ROOT / "ai-engine" / "src"
if str(AI_ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE_SRC))

from biotrace.facial_dynamics.domain import LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.face_tracker import (
    FOREHEAD_LANDMARKS,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
)
from biotrace.rppg.filtering import compute_welch_psd, find_dominant_cardiac_peak
from biotrace.rppg.quality import calculate_de_haan_snr
from biotrace.vision.mediapipe_compat import create_face_mesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase9Extraction")

FROZEN_MANIFEST_SHA256 = "c15300c15ca78fb58ab16d8f01549534dffb05fb9d75664ac14b2d12c0af06b7"


def verify_manifest(manifest_path: str) -> None:
    """Ensures protocol_c_split_manifest.csv is intact."""
    with open(manifest_path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    if h != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"Manifest SHA-256 mismatch! Expected {FROZEN_MANIFEST_SHA256}, got {h}"
        )
    logger.info("Manifest SHA-256 verified OK: %s...", h[:16])


def calc_ear_from_pts(pts: np.ndarray) -> float:
    """Calculates Soukupová & Čech (2016) Eye Aspect Ratio."""
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return float((v1 + v2) / (2.0 * max(h, 1e-6)))


def poly_mean_rgb(frame_bgr: np.ndarray, poly_pts: np.ndarray) -> Optional[np.ndarray]:
    """Computes mean [R, G, B] values within facial skin polygon mask."""
    if len(poly_pts) < 3:
        return None
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_pts.reshape((-1, 1, 2))], 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    # Return RGB
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


def extract_video_features(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function processing a single video to extract the 10 approved features."""
    video_path = task["file_path"]
    row: Dict[str, Any] = {
        "video_id": task["video_id"],
        "base_id": task["base_id"],
        "target_id": task["target_id"],
        "label": task["label"],
        "ground_truth_class": task["ground_truth_class"],
        "split": task["split"],
        "extraction_success": 0,
        "error_message": "",
        # 4 Facial Kinematics
        "ear_right_std": np.nan,
        "ear_bino_std": np.nan,
        "ear_velocity_mean": np.nan,
        "ear_opening_vel_max": np.nan,
        # 4 Exploratory rPPG Chromatic Consistency
        "rppg_pos_chrom_corr": np.nan,
        "rppg_pos_snr_db": np.nan,
        "rppg_pos_peak_freq_hz": np.nan,
        "rppg_pos_peak_prominence": np.nan,
        # 2 Exploratory Facial Asymmetry
        "blink_asymmetry_mean": np.nan,
        "blink_asymmetry_std": np.nan,
    }

    if not os.path.exists(video_path):
        row["error_message"] = "File not found"
        return row

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        row["error_message"] = "OpenCV failed to open video"
        return row

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 24.0

    ear_right_list: List[float] = []
    ear_bino_list: List[float] = []
    asym_list: List[float] = []
    rgb_traces: List[np.ndarray] = []

    # Use MediaPipe face landmarks in single-pass mode
    face_mesh = create_face_mesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                pts_px = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
                pts_int = np.array([[int(round(lm.x * w)), int(round(lm.y * h))] for lm in lms], dtype=np.int32)

                # 1. Eyes & Kinematics
                ear_l = calc_ear_from_pts(pts_px[LEFT_EYE_LANDMARKS])
                ear_r = calc_ear_from_pts(pts_px[RIGHT_EYE_LANDMARKS])
                ear_b = (ear_l + ear_r) / 2.0

                ear_right_list.append(ear_r)
                ear_bino_list.append(ear_b)

                denom = ear_l + ear_r
                if denom > 1e-4:
                    asym = abs(ear_l - ear_r) / denom
                    asym_list.append(asym)

                # 2. Skin patches for rPPG (forehead, left cheek, right cheek)
                fh_rgb = poly_mean_rgb(frame, pts_int[FOREHEAD_LANDMARKS])
                lc_rgb = poly_mean_rgb(frame, pts_int[LEFT_CHEEK_LANDMARKS])
                rc_rgb = poly_mean_rgb(frame, pts_int[RIGHT_CHEEK_LANDMARKS])

                valid_patches = [p for p in [fh_rgb, lc_rgb, rc_rgb] if p is not None]
                if valid_patches:
                    comb_rgb = np.mean(valid_patches, axis=0)
                    rgb_traces.append(comb_rgb)
    except Exception as e:
        row["error_message"] = f"Extraction loop error: {e}"
        return row
    finally:
        cap.release()
        face_mesh.close()

    # Calculate facial kinematic features
    if len(ear_bino_list) >= 10:
        row["ear_right_std"] = round(float(np.std(ear_right_list)), 6)
        row["ear_bino_std"] = round(float(np.std(ear_bino_list)), 6)

        dt = 1.0 / fps
        ear_vels = np.diff(ear_bino_list) / dt
        row["ear_velocity_mean"] = round(float(np.mean(np.abs(ear_vels))), 6)

        opening_vels = [v for v in ear_vels if v > 0]
        if opening_vels:
            row["ear_opening_vel_max"] = round(float(np.max(opening_vels)), 6)

    # Calculate asymmetry features
    if len(asym_list) >= 10:
        row["blink_asymmetry_mean"] = round(float(np.mean(asym_list)), 6)
        row["blink_asymmetry_std"] = round(float(np.std(asym_list)), 6)

    # Calculate rPPG chromatic features
    if len(rgb_traces) >= 30:
        try:
            rgb_arr = np.array(rgb_traces, dtype=np.float64)
            pos = extract_pos_rppg(rgb_arr, fps)
            chrom = extract_chrom_rppg(rgb_arr, fps)

            if len(pos) > 10 and len(chrom) == len(pos) and np.std(pos) > 1e-7 and np.std(chrom) > 1e-7:
                row["rppg_pos_chrom_corr"] = round(float(np.corrcoef(pos, chrom)[0, 1]), 6)

            freqs, psd = compute_welch_psd(pos, fps)
            peak_f, peak_p, prom = find_dominant_cardiac_peak(freqs, psd)
            snr = calculate_de_haan_snr(freqs, psd, peak_f)

            row["rppg_pos_snr_db"] = round(float(snr), 4)
            row["rppg_pos_peak_freq_hz"] = round(float(peak_f), 4)
            row["rppg_pos_peak_prominence"] = round(float(prom), 4)
        except Exception as e:
            row["error_message"] += f" [rPPG error: {e}]"

    # Mark success if at least facial or rPPG features were populated
    if not np.isnan(row["ear_bino_std"]) or not np.isnan(row["rppg_pos_snr_db"]):
        row["extraction_success"] = 1

    return row


def build_full_task_list(
    manifest_path: str,
    real_dir: str,
    fake_dir: str,
) -> List[Dict[str, Any]]:
    """Builds the comprehensive 8,722 video task list from the frozen manifest."""
    verify_manifest(manifest_path)
    df = pd.read_csv(manifest_path)

    tasks: List[Dict[str, Any]] = []
    # Every pair in the manifest defines 1 real video and 1 fake video
    for _, r in df.iterrows():
        b_id = str(r["base_id"])
        t_id = str(r["target_id"])
        split = str(r["split"])

        # Real video
        tasks.append({
            "video_id": b_id,
            "base_id": b_id,
            "target_id": "",
            "file_path": os.path.join(real_dir, f"{b_id}.mp4"),
            "label": "real",
            "ground_truth_class": 0,
            "split": split,
        })

        # Fake video
        tasks.append({
            "video_id": f"{b_id}_{t_id}",
            "base_id": b_id,
            "target_id": t_id,
            "file_path": os.path.join(fake_dir, f"{b_id}_{t_id}.mp4"),
            "label": "fake",
            "ground_truth_class": 1,
            "split": split,
        })

    logger.info("Compiled %d total video extraction tasks from manifest.", len(tasks))
    return tasks


def run_full_extraction(
    manifest_path: str = "protocol_c_split_manifest.csv",
    real_dir: str = "data/extracted/hidf/real",
    fake_dir: str = "data/extracted/hidf/fake",
    output_csv: str = "phase9_physiological_features.csv",
    max_workers: int = 8,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Executes full dataset physiological feature extraction with resume capability."""
    tasks = build_full_task_list(manifest_path, real_dir, fake_dir)
    if limit is not None:
        tasks = tasks[:limit]

    # Resumability check
    already_done: Set[str] = set()
    existing_rows: List[Dict[str, Any]] = []
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            already_done = set(prev_df["video_id"].astype(str))
            existing_rows = prev_df.to_dict(orient="records")
            logger.info("Resuming extraction: found %d videos already extracted.", len(already_done))
        except Exception as e:
            logger.warning("Could not parse existing output CSV (%s); starting fresh.", e)

    remaining_tasks = [t for t in tasks if t["video_id"] not in already_done]
    logger.info("Remaining tasks to process: %d / %d", len(remaining_tasks), len(tasks))

    if not remaining_tasks:
        logger.info("All tasks are already completed!")
        return pd.DataFrame(existing_rows)

    completed_rows = existing_rows
    t_start = time.perf_counter()
    save_interval = 100
    batch_counter = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_video_features, t): t for t in remaining_tasks}

        for i, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                completed_rows.append(res)
            except Exception as e:
                t_info = futures[future]
                logger.error("Future raised unhandled exception on %s: %s", t_info["video_id"], e)
                completed_rows.append({
                    "video_id": t_info["video_id"],
                    "base_id": t_info["base_id"],
                    "target_id": t_info["target_id"],
                    "label": t_info["label"],
                    "ground_truth_class": t_info["ground_truth_class"],
                    "split": t_info["split"],
                    "extraction_success": 0,
                    "error_message": str(e),
                    "ear_right_std": np.nan,
                    "ear_bino_std": np.nan,
                    "ear_velocity_mean": np.nan,
                    "ear_opening_vel_max": np.nan,
                    "rppg_pos_chrom_corr": np.nan,
                    "rppg_pos_snr_db": np.nan,
                    "rppg_pos_peak_freq_hz": np.nan,
                    "rppg_pos_peak_prominence": np.nan,
                    "blink_asymmetry_mean": np.nan,
                    "blink_asymmetry_std": np.nan,
                })

            batch_counter += 1
            if batch_counter >= save_interval or i == len(remaining_tasks):
                batch_counter = 0
                df_interim = pd.DataFrame(completed_rows)
                df_interim.to_csv(output_csv, index=False)
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(remaining_tasks) - i) / rate if rate > 0 else 0
                logger.info(
                    "Progress: [%5d/%5d remaining] (Total %5d/%5d) | %.1f vids/sec | ETA: %.1f min",
                    i,
                    len(remaining_tasks),
                    len(completed_rows),
                    len(tasks),
                    rate,
                    eta / 60.0,
                )

    df_final = pd.DataFrame(completed_rows)
    df_final.to_csv(output_csv, index=False)
    logger.info(
        "Full physiological extraction complete! Saved %d records to %s",
        len(df_final),
        output_csv,
    )
    return df_final


def main():
    parser = argparse.ArgumentParser(description="BIOTrace Phase 9 Full Physiological Extraction")
    parser.add_argument("--manifest", type=str, default="protocol_c_split_manifest.csv")
    parser.add_argument("--real-dir", type=str, default="data/extracted/hidf/real")
    parser.add_argument("--fake-dir", type=str, default="data/extracted/hidf/fake")
    parser.add_argument("--output-csv", type=str, default="phase9_physiological_features.csv")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional test limit")
    args = parser.parse_args()

    run_full_extraction(
        manifest_path=args.manifest,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_csv=args.output_csv,
        max_workers=args.max_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
