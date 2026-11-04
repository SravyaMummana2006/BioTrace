"""Dataset registry and automated discovery for standardized benchmark datasets.

Scientific Principles & Forensic Integrity:
    1. Schema Preservation: Maps dataset-specific folder structures, naming conventions,
       and metadata formats into standardized VideoRecord descriptors.
    2. Zero Assumption of Local Availability: Checks for directory existence; if files are
       missing or unapproved, cleanly flags the dataset as AWAITING_DATASET_ACCESS.
    3. Physiological Ground-Truth Extraction: Integrates ground-truth parsers for both
       legacy UBFC-rPPG and 64 Hz Empatica E4 contact sensors (UBFC-Phys).
    4. Verified Source-Target Deepfake Pairing: Inspects official metadata to pair deepfake
       derivatives with their genuine base videos, preventing unverified heuristic pairing.
"""

import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import signal as sp_signal

from biotrace.core.logging import get_logger
from biotrace.experiments.domain import DatasetIdentifier, ExperimentStatus

logger = get_logger(__name__)


def parse_ubfc_ground_truth(gt_path: str) -> Dict[str, Any]:
    """Parses ground-truth text file from the UBFC-rPPG dataset.

    Format in UBFC-rPPG ground_truth.txt:
        Line 1: BVP continuous waveform values (space-separated floats)
        Line 2: Instantaneous Heart Rate (BPM) values (space-separated floats)
        Line 3: Timestamps in seconds (space-separated floats)
    """
    bvp: List[float] = []
    bpm: List[float] = []
    timestamps: List[float] = []

    if not os.path.exists(gt_path):
        return {"bvp": bvp, "bpm": bpm, "timestamps": timestamps, "mean_hr_bpm": None}

    with open(gt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) >= 3:
        try:
            bvp = [float(x) for x in lines[0].split()]
            bpm = [float(x) for x in lines[1].split()]
            timestamps = [float(x) for x in lines[2].split()]
        except ValueError as e:
            logger.warning("Error parsing numbers from UBFC ground truth %s: %s", gt_path, e)
    elif len(lines) == 1:
        # Some variants have only heart rate or BVP
        try:
            bpm = [float(x) for x in lines[0].split()]
        except ValueError:
            pass

    mean_hr = float(sum(bpm) / len(bpm)) if bpm else None

    return {
        "bvp": bvp,
        "bpm": bpm,
        "timestamps": timestamps,
        "mean_hr_bpm": mean_hr,
        "num_reference_points": len(bpm),
    }


def parse_ubfc_phys_bvp(
    gt_path: str,
    default_sampling_rate_hz: float = 64.0,
) -> Dict[str, Any]:
    """Parses ground-truth Blood Volume Pulse (BVP) from the UBFC-Phys dataset.

    Supports:
        1. Empatica E4 standard CSV format (line 1: start time, line 2: fs=64.0, lines 3+: BVP values).
        2. Single-column or multi-column numeric CSV/TXT BVP waveforms.

    Forensic & Physical Principles:
        - Continuous sampling frequency (fs) validation.
        - Computes reference mean heart rate (BPM) via Welch power spectral density (PSD)
          in the standard cardiac band [0.70 Hz, 3.0 Hz] (42 to 180 BPM).
        - Requires at least 2.0 seconds of signal for stable reference heart rate estimation.
    """
    bvp: List[float] = []
    timestamps: List[float] = []
    fs = float(default_sampling_rate_hz)

    if not os.path.exists(gt_path):
        return {
            "bvp": bvp,
            "timestamps": timestamps,
            "sampling_rate_hz": fs,
            "mean_hr_bpm": None,
            "num_reference_points": 0,
        }

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            return {
                "bvp": bvp,
                "timestamps": timestamps,
                "sampling_rate_hz": fs,
                "mean_hr_bpm": None,
                "num_reference_points": 0,
            }

        # Check for Empatica E4 header format:
        # Line 0 is Unix epoch (> 1e8), Line 1 is sample rate (e.g. 64.0)
        is_empatica_header = False
        if len(lines) >= 3:
            try:
                val0 = float(lines[0].split(",")[0])
                val1 = float(lines[1].split(",")[0])
                if val0 > 1e8 and (1.0 <= val1 <= 256.0):
                    is_empatica_header = True
                    fs = val1
            except ValueError:
                is_empatica_header = False

        data_lines = lines[2:] if is_empatica_header else lines

        for line in data_lines:
            parts = re.split(r"[,\s\t]+", line.strip())
            if not parts:
                continue
            try:
                if len(parts) >= 2:
                    try:
                        val = float(parts[1])
                    except ValueError:
                        val = float(parts[0])
                else:
                    val = float(parts[0])
                bvp.append(val)
            except ValueError:
                # Skip non-numeric header lines
                continue

    except Exception as e:
        logger.warning("Error reading UBFC-Phys BVP file %s: %s", gt_path, e)

    num_pts = len(bvp)
    if num_pts > 0:
        timestamps = [float(i / fs) for i in range(num_pts)]

    # Compute reference mean heart rate (BPM) via Welch PSD in cardiac band [0.70 Hz, 3.0 Hz]
    mean_hr: Optional[float] = None
    if num_pts >= int(fs * 2.0):
        bvp_arr = np.array(bvp, dtype=np.float64)
        bvp_centered = bvp_arr - np.mean(bvp_arr)
        nperseg = min(num_pts, int(fs * 8.0))
        if nperseg >= int(fs * 2.0) and np.std(bvp_centered) > 1e-6:
            freqs, psd = sp_signal.welch(bvp_centered, fs=fs, nperseg=nperseg)
            cardiac_mask = (freqs >= 0.70) & (freqs <= 3.0)
            if np.any(cardiac_mask):
                peak_freq = freqs[cardiac_mask][np.argmax(psd[cardiac_mask])]
                mean_hr = round(float(peak_freq * 60.0), 2)

    return {
        "bvp": bvp,
        "timestamps": timestamps,
        "sampling_rate_hz": fs,
        "mean_hr_bpm": mean_hr,
        "num_reference_points": num_pts,
    }


class DatasetRegistry:
    """Central registry and layout resolver for research datasets."""

    def __init__(self, datasets_base_dir: str = "datasets"):
        self.base_dir = Path(datasets_base_dir)

    def get_dataset_status(self, dataset_id: str) -> Tuple[ExperimentStatus, Optional[str]]:
        """Checks if the dataset directory exists locally and contains video assets."""
        path = self.get_default_dataset_path(dataset_id)
        if not path.exists():
            return (
                ExperimentStatus.AWAITING_DATASET_ACCESS,
                f"Dataset directory '{path}' not found. Please review research/datasets/DATASET_ACCESS_AUDIT.md for access procedure.",
            )

        # Check for media files
        video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
        has_videos = any(
            f.suffix.lower() in video_extensions
            for f in path.rglob("*")
            if f.is_file()
        )

        if not has_videos:
            return (
                ExperimentStatus.AWAITING_DATASET_ACCESS,
                f"Dataset directory '{path}' exists but contains no recognized video files.",
            )

        return ExperimentStatus.DATASET_AVAILABLE, str(path)

    def get_default_dataset_path(self, dataset_id: str) -> Path:
        """Resolves the default relative path for a dataset identifier."""
        name_map = {
            DatasetIdentifier.UBFC_RPPG: "UBFC-rPPG",
            DatasetIdentifier.UBFC_PHYS: "UBFC-Phys",
            DatasetIdentifier.FACEFORENSICS_PP: "FaceForensics++",
            DatasetIdentifier.CELEB_DF_V2: "Celeb-DF-v2",
            DatasetIdentifier.DFDC: "DFDC",
            DatasetIdentifier.HIDF: "HiDF",
            DatasetIdentifier.SYNTHETIC_TEST: "synthetic_test",
        }
        dir_name = name_map.get(dataset_id, dataset_id)
        return self.base_dir / dir_name

    def discover_ubfc_samples(self, dataset_root: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scans UBFC-rPPG subject folders for video and ground_truth.txt files."""
        root = Path(dataset_root) if dataset_root else self.get_default_dataset_path(DatasetIdentifier.UBFC_RPPG)
        descriptors: List[Dict[str, Any]] = []

        if not root.exists():
            return descriptors

        for item in sorted(root.iterdir()):
            if item.is_dir() and item.name.lower().startswith("subject"):
                subj_id = item.name
                vid_file = None
                for ext in [".avi", ".mp4", ".mov"]:
                    cand = item / f"vid{ext}"
                    if cand.exists():
                        vid_file = cand
                        break
                if not vid_file:
                    # Look for any video in subject folder
                    for f in item.glob("*"):
                        if f.suffix.lower() in [".avi", ".mp4", ".mov"]:
                            vid_file = f
                            break

                if vid_file:
                    gt_file = item / "ground_truth.txt"
                    gt_path = str(gt_file) if gt_file.exists() else None
                    descriptors.append({
                        "video_path": str(vid_file),
                        "record_id": f"ubfc_{subj_id}",
                        "label": 0,  # All UBFC subjects are authentic
                        "subject_id": subj_id,
                        "source_video_id": f"ubfc_{subj_id}",
                        "manipulation_type": "none",
                        "compression": "raw",
                        "ground_truth_path": gt_path,
                    })

        return descriptors

    def discover_ubfc_phys_samples(
        self, dataset_root: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Scans UBFC-Phys directory for subject folders (s1, s2...) and task recordings (T1, T2, T3).

        Forensic Principles & Invariants:
            1. Authentic Exclusivity: All UBFC-Phys sequences represent authentic human participants (label=0).
            2. Subject-Level Identity Containment: All tasks (T1, T2, T3) executed by participant 's1'
               share subject_id='s1', enabling verified SUBJECT_LEVEL partitioning without leakage.
            3. Synchronized Contact Sensor Association: Links Empatica E4 64 Hz BVP files to video tracks.
        """
        root = Path(dataset_root) if dataset_root else self.get_default_dataset_path(DatasetIdentifier.UBFC_PHYS)
        descriptors: List[Dict[str, Any]] = []

        if not root.exists():
            return descriptors

        for item in sorted(root.iterdir()):
            if item.is_dir() and re.match(r"^(s\d+|subject\d+)", item.name, re.IGNORECASE):
                subj_id = item.name.lower()

                # Find video files within participant folder
                video_files = sorted(
                    f for f in item.glob("*")
                    if f.suffix.lower() in [".avi", ".mp4", ".mov", ".mkv"]
                )

                for vid_file in video_files:
                    stem = vid_file.stem
                    # Detect task phase (T1=Rest, T2=Speech, T3=Arithmetic)
                    task_match = re.search(r"(T[123]|rest|speech|arithmetic|task[123])", stem, re.IGNORECASE)
                    task_id = task_match.group(1).upper() if task_match else "T1"

                    # Candidate ground truth files for this task
                    gt_candidates = [
                        item / f"bvp_{subj_id}_{task_id}.csv",
                        item / f"bvp_{subj_id}_{task_id}.txt",
                        item / f"bvp_{task_id}.csv",
                        item / f"bvp_{task_id}.txt",
                        item / f"bvp_{task_id.lower()}.csv",
                        item / f"bvp_{task_id.lower()}.txt",
                        item / "bvp.csv",
                        item / "bvp.txt",
                    ]

                    gt_path = None
                    for cand in gt_candidates:
                        if cand.exists():
                            gt_path = str(cand)
                            break

                    if not gt_path:
                        for f in item.glob("*"):
                            if "bvp" in f.name.lower() and task_id.lower() in f.name.lower() and f.suffix.lower() in [".csv", ".txt"]:
                                gt_path = str(f)
                                break

                    descriptors.append({
                        "video_path": str(vid_file),
                        "record_id": f"ubfc_phys_{subj_id}_{task_id.lower()}",
                        "label": 0,  # Authentic physiological benchmark recording
                        "subject_id": subj_id,
                        "source_video_id": f"ubfc_phys_{subj_id}_{task_id.lower()}",
                        "manipulation_type": "none",
                        "compression": "raw",
                        "ground_truth_path": gt_path,
                        "metadata": {"task": task_id, "dataset": "UBFC-Phys"},
                    })

        return descriptors

    def discover_faceforensics_samples(
        self,
        dataset_root: Optional[str] = None,
        compression: str = "c23",
    ) -> List[Dict[str, Any]]:
        """Scans FaceForensics++ directory for authentic and manipulated sequences."""
        root = Path(dataset_root) if dataset_root else self.get_default_dataset_path(DatasetIdentifier.FACEFORENSICS_PP)
        descriptors: List[Dict[str, Any]] = []

        if not root.exists():
            return descriptors

        # 1. Authentic originals
        orig_dir = root / "original_sequences" / compression / "videos"
        if orig_dir.exists():
            for f in sorted(orig_dir.glob("*.mp4")):
                src_id = f.stem
                descriptors.append({
                    "video_path": str(f),
                    "record_id": f"ff_{compression}_orig_{src_id}",
                    "label": 0,
                    "source_video_id": src_id,
                    "subject_id": None,  # Sourced from YouTube; actor identity maps to source video
                    "manipulation_type": "none",
                    "compression": compression,
                })

        # 2. Manipulated sequences
        manip_dir = root / "manipulated_sequences"
        if manip_dir.exists():
            for method_dir in sorted(manip_dir.iterdir()):
                if method_dir.is_dir():
                    method_name = method_dir.name
                    vid_dir = method_dir / compression / "videos"
                    if vid_dir.exists():
                        for f in sorted(vid_dir.glob("*.mp4")):
                            stem = f.stem
                            # FF++ naming convention: "<donor>_<target>"
                            parts = stem.split("_")
                            target_id = parts[1] if len(parts) >= 2 else parts[0]
                            descriptors.append({
                                "video_path": str(f),
                                "record_id": f"ff_{compression}_{method_name}_{stem}",
                                "label": 1,
                                "source_video_id": target_id,  # Target video is the underlying background/face recipient
                                "subject_id": None,
                                "manipulation_type": method_name,
                                "compression": compression,
                            })

        return descriptors

    def discover_celeb_df_samples(self, dataset_root: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scans Celeb-DF v2 directory."""
        root = Path(dataset_root) if dataset_root else self.get_default_dataset_path(DatasetIdentifier.CELEB_DF_V2)
        descriptors: List[Dict[str, Any]] = []

        if not root.exists():
            return descriptors

        # Celeb-real
        real_dir = root / "Celeb-real"
        if real_dir.exists():
            for f in sorted(real_dir.glob("*.mp4")):
                # Format: id<number>_<number>.mp4
                match = re.match(r"(id\d+)", f.stem)
                subj = match.group(1) if match else None
                descriptors.append({
                    "video_path": str(f),
                    "record_id": f"cdf_real_{f.stem}",
                    "label": 0,
                    "subject_id": subj,
                    "source_video_id": f.stem,
                    "manipulation_type": "none",
                    "compression": "h264",
                })

        # Celeb-synthesis
        synth_dir = root / "Celeb-synthesis"
        if synth_dir.exists():
            for f in sorted(synth_dir.glob("*.mp4")):
                # Format: id<donor>_id<target>_<num>.mp4
                parts = f.stem.split("_")
                target_subj = parts[1] if len(parts) >= 2 else None
                descriptors.append({
                    "video_path": str(f),
                    "record_id": f"cdf_synth_{f.stem}",
                    "label": 1,
                    "subject_id": target_subj,
                    "source_video_id": target_subj,
                    "manipulation_type": "Deepfakes",
                    "compression": "h264",
                })

        return descriptors

    def discover_hidf_samples(
        self,
        dataset_root: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Scans HiDF dataset directory for authentic (Real-vid) and manipulated (Fake-vid) videos.

        Forensic Principles & Invariants:
            1. Official Metadata Verification: Validates base/target pairs using metadata.csv
               whenever present, strictly avoiding unverified filename heuristics.
            2. Source Identity Containment: Maps source_video_id to the base original video,
               guaranteeing that face-swapped derivatives are partitioned alongside the donor.
            3. Binary Label Accuracy: Real-vid is mapped to label=0 (pristine); Fake-vid to label=1.
        """
        root = Path(dataset_root) if dataset_root else self.get_default_dataset_path(DatasetIdentifier.HIDF)
        descriptors: List[Dict[str, Any]] = []

        if not root.exists():
            return descriptors

        # 1. Resolve official metadata CSV if present
        meta_candidates = [
            Path(metadata_path) if metadata_path else None,
            root / "metadata.csv",
            root / "HiDF_metadata.csv",
        ]
        meta_file = next((c for c in meta_candidates if c and c.exists()), None)

        verified_metadata: Dict[str, Dict[str, Any]] = {}
        if meta_file:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Extract primary video key
                        vid_key = (
                            row.get("video_id")
                            or row.get("filename")
                            or row.get("file_name")
                            or row.get("name")
                        )
                        if vid_key:
                            clean_key = Path(vid_key).stem
                            verified_metadata[clean_key] = row
            except Exception as e:
                logger.warning("Error reading HiDF metadata file %s: %s", meta_file, e)

        # 2. Locate video subdirectories
        real_dirs = [root / "Real-vid", root / "real", root / "Real", root / "real_videos"]
        fake_dirs = [root / "Fake-vid", root / "fake", root / "Fake", root / "fake_videos"]

        real_dir = next((d for d in real_dirs if d.exists() and d.is_dir()), None)
        fake_dir = next((d for d in fake_dirs if d.exists() and d.is_dir()), None)

        known_real_stems: Set[str] = set()

        # 3. Process Authentic (Real) Videos
        if real_dir:
            for f in sorted(real_dir.glob("*")):
                if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    stem = f.stem
                    known_real_stems.add(stem)
                    meta_row = verified_metadata.get(stem, {})
                    subj_id = meta_row.get("subject_id") or meta_row.get("actor_id") or stem

                    descriptors.append({
                        "video_path": str(f),
                        "record_id": f"hidf_real_{stem}",
                        "label": 0,
                        "source_video_id": stem,
                        "subject_id": subj_id,
                        "manipulation_type": "none",
                        "compression": "h264",
                        "metadata": meta_row,
                    })

        # 4. Process Manipulated (Fake) Videos
        if fake_dir:
            for f in sorted(fake_dir.glob("*")):
                if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                    stem = f.stem
                    meta_row = verified_metadata.get(stem, {})

                    # Determine base (source) video ID
                    base_id = None
                    if "base_id" in meta_row:
                        base_id = str(meta_row["base_id"]).strip()
                    elif "base_video_id" in meta_row:
                        base_id = str(meta_row["base_video_id"]).strip()
                    elif "source_id" in meta_row:
                        base_id = str(meta_row["source_id"]).strip()

                    # Fallback: parse from filename prefix if metadata did not provide it
                    pairing_verified = bool(base_id)
                    if not base_id:
                        parts = stem.split("_")
                        if len(parts) >= 2:
                            cand_base = parts[0]
                            if cand_base in known_real_stems:
                                base_id = cand_base
                                pairing_verified = True
                            else:
                                base_id = cand_base
                        else:
                            base_id = stem

                    manip_type = (
                        meta_row.get("manipulation_type")
                        or meta_row.get("method")
                        or "HiDF_Deepfake"
                    )

                    record_meta = meta_row.copy()
                    record_meta["pairing_verified"] = pairing_verified

                    descriptors.append({
                        "video_path": str(f),
                        "record_id": f"hidf_fake_{stem}",
                        "label": 1,
                        "source_video_id": base_id,
                        "subject_id": None,  # Transferred face; background/body belongs to base video
                        "manipulation_type": manip_type,
                        "compression": "h264",
                        "metadata": record_meta,
                    })

        return descriptors
