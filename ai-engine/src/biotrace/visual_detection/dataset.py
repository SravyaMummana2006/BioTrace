"""Dataset abstraction, video-level splitting, and class-balancing utilities.

Scientific Principles & Forensic Guardrails:
    1. Prevention of Frame Leakage:
       Adjacent video frames share background, lighting, and subject facial structure.
       Splitting at the frame level creates catastrophic data leakage, yielding artificially
       inflated test scores (e.g., 99%+ accuracy) that collapse on unseen videos.
       All data partitioning occurs STRICTLY at the video (or subject) level.
    2. Class Imbalance Mitigation:
       Deepfake datasets often have significant disparities between authentic and manipulated videos.
       Provides weighted random sampling and loss weighting to prevent majority-class bias.
    3. Zero Unverified Dataset Ingestion:
       Operates with verified local paths and manifest records. Does not download unverified or
       unlicensed external datasets.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from biotrace.core.logging import get_logger
from biotrace.visual_detection.preprocessing import ImageNormalizer

logger = get_logger(__name__)


@dataclass
class VideoRecord:
    """Metadata describing a video sample in a visual detection dataset."""

    video_id: str
    video_path: str
    label: int  # 0 for authentic/pristine, 1 for synthetic/manipulated
    subject_id: Optional[str] = None
    manipulation_type: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    frames: List["FrameSample"] = field(default_factory=list)


@dataclass
class FrameSample:
    """A single frame or face crop sample linked to its parent video record."""

    video_id: str
    frame_index: int
    image_path: Optional[str] = None
    image_array: Optional[np.ndarray] = None
    label: int = 0
    subject_id: Optional[str] = None
    timestamp: float = 0.0
    timestamp_seconds: float = 0.0
    crop_tensor: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.timestamp != 0.0 and self.timestamp_seconds == 0.0:
            self.timestamp_seconds = self.timestamp
        elif self.timestamp_seconds != 0.0 and self.timestamp == 0.0:
            self.timestamp = self.timestamp_seconds


class VideoDataSplitter:
    """Partitions video collections into disjoint train, validation, and test subsets.

    Scientific Principles & Leakage Prevention (Stage 6.1 Audit):
        1. Subject-Level Priority:
           When subject IDs (`subject_id`) are present, subject-level grouping takes priority.
           All videos depicting the same individual are allocated strictly to the same partition,
           guaranteeing that biometric identity or actor-specific lighting cannot cross between
           train, validation, and test sets.
        2. Handling Missing Subject IDs:
           When `subject_id` is None, missing, or empty, the partitioning unit defaults to the
           unique `video_id` (prefixed as `video_<id>`). This ensures that unassigned videos are
           treated as independent video units and cannot be inappropriately grouped together.
        3. Strict Zero-Leakage Invariants:
           The splitter asserts mathematical disjointness across both video IDs and subject IDs:
               train_ids.isdisjoint(val_ids)
               train_ids.isdisjoint(test_ids)
               val_ids.isdisjoint(test_ids)
               train_subjects.isdisjoint(val_subjects)
               train_subjects.isdisjoint(test_subjects)
               val_subjects.isdisjoint(test_subjects)
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_videos(
        self,
        videos: List[VideoRecord],
    ) -> Tuple[List[VideoRecord], List[VideoRecord], List[VideoRecord]]:
        """Splits videos, prioritizing subject-level grouping when subject IDs are available."""
        return self._split_videos_internal(
            videos=videos,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
        )

    def split_by_subject(
        self,
        videos: List[VideoRecord],
    ) -> Tuple[List[VideoRecord], List[VideoRecord], List[VideoRecord]]:
        """Explicit subject-level splitting alias."""
        return self.split_videos(videos)

    @classmethod
    def split_videos_static(
        cls,
        videos: List[VideoRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[VideoRecord], List[VideoRecord], List[VideoRecord]]:
        return cls._split_videos_internal(
            videos=videos,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    @classmethod
    def _split_videos_internal(
        cls,
        videos: List[VideoRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[VideoRecord], List[VideoRecord], List[VideoRecord]]:
        """Splits video records strictly at video or subject level to eliminate frame leakage."""
        if not videos:
            return [], [], []

        total_ratio = train_ratio + val_ratio + test_ratio
        if not math_close(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        rng = random.Random(seed)

        # Subject-level grouping takes priority when subject IDs are available.
        # Missing subject IDs fall back to unique video_id keys.
        groups: Dict[str, List[VideoRecord]] = {}
        for v in videos:
            if v.subject_id and v.subject_id.strip():
                key = f"subj_{v.subject_id.strip()}"
            else:
                key = f"video_{v.video_id}"
            groups.setdefault(key, []).append(v)

        group_keys = sorted(list(groups.keys()))
        rng.shuffle(group_keys)

        # Stratify by label if groups have homogeneous labels
        pos_keys = []
        neg_keys = []
        for k in group_keys:
            labels = [v.label for v in groups[k]]
            avg_label = sum(labels) / len(labels)
            if avg_label >= 0.5:
                pos_keys.append(k)
            else:
                neg_keys.append(k)

        def partition_list(keys: List[str]) -> Tuple[List[str], List[str], List[str]]:
            n = len(keys)
            n_train = int(round(train_ratio * n))
            n_val = int(round(val_ratio * n))
            train_k = keys[:n_train]
            val_k = keys[n_train : n_train + n_val]
            test_k = keys[n_train + n_val :]
            return train_k, val_k, test_k

        pos_train, pos_val, pos_test = partition_list(pos_keys)
        neg_train, neg_val, neg_test = partition_list(neg_keys)

        train_keys = pos_train + neg_train
        val_keys = pos_val + neg_val
        test_keys = pos_test + neg_test

        train_v = [v for k in train_keys for v in groups[k]]
        val_v = [v for k in val_keys for v in groups[k]]
        test_v = [v for k in test_keys for v in groups[k]]

        # Scientific audit assertion: verify zero leakage across videos and subjects
        train_ids = {v.video_id for v in train_v}
        val_ids = {v.video_id for v in val_v}
        test_ids = {v.video_id for v in test_v}

        assert train_ids.isdisjoint(val_ids), "Critical error: Video leakage detected between train and val!"
        assert train_ids.isdisjoint(test_ids), "Critical error: Video leakage detected between train and test!"
        assert val_ids.isdisjoint(test_ids), "Critical error: Video leakage detected between val and test!"

        train_subj = {v.subject_id for v in train_v if v.subject_id and v.subject_id.strip()}
        val_subj = {v.subject_id for v in val_v if v.subject_id and v.subject_id.strip()}
        test_subj = {v.subject_id for v in test_v if v.subject_id and v.subject_id.strip()}

        assert train_subj.isdisjoint(val_subj), "Critical error: Subject leakage detected between train and val!"
        assert train_subj.isdisjoint(test_subj), "Critical error: Subject leakage detected between train and test!"
        assert val_subj.isdisjoint(test_subj), "Critical error: Subject leakage detected between val and test!"

        logger.info(
            "Video/subject-level partition complete: %d train, %d val, %d test (Zero leakage verified).",
            len(train_v),
            len(val_v),
            len(test_v),
        )
        return train_v, val_v, test_v


def math_close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


class VideoFrameDataset(Dataset):
    """PyTorch Dataset for visual deepfake detection on extracted face crop frames."""

    def __init__(
        self,
        samples: List[FrameSample],
        target_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.samples = samples
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        sample = self.samples[idx]

        meta = {
            "video_id": sample.video_id,
            "frame_index": str(sample.frame_index),
            "subject_id": sample.subject_id or "",
        }
        tensor_label = torch.tensor(sample.label, dtype=torch.float32)

        if sample.crop_tensor is not None:
            return sample.crop_tensor, tensor_label, meta

        # Load image array from memory or disk
        if sample.image_array is not None:
            bgr = sample.image_array
        elif sample.image_path and Path(sample.image_path).exists():
            bgr = cv2.imread(sample.image_path)
        else:
            # Fallback zero image if path missing
            bgr = np.zeros((*self.target_size, 3), dtype=np.uint8)

        if self.transform:
            bgr = self.transform(bgr)

        chw = ImageNormalizer.preprocess_crop(
            crop_bgr=bgr,
            target_size=self.target_size,
            mean=self.mean,
            std=self.std,
        )

        tensor_img = torch.from_numpy(chw).float()
        return tensor_img, tensor_label, meta

    def get_sample_weights(self) -> List[float]:
        """Calculates per-sample inverse-frequency weights for class balancing."""
        labels = [s.label for s in self.samples]
        if not labels:
            return []

        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        if pos_count == 0 or neg_count == 0:
            return [1.0] * len(labels)

        # Inverse frequency: total / (2 * class_count)
        total = float(len(labels))
        weight_pos = total / (2.0 * pos_count)
        weight_neg = total / (2.0 * neg_count)
        return [weight_pos if lbl == 1 else weight_neg for lbl in labels]

    def get_class_balancer(self) -> WeightedRandomSampler:
        """Constructs a WeightedRandomSampler to balance authentic and manipulated classes."""
        weights = self.get_sample_weights()
        if not weights:
            return WeightedRandomSampler([1.0], num_samples=1)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
