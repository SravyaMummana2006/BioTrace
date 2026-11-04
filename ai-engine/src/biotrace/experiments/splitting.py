"""Leakage-safe, identity-aware dataset splitting for research experiments.

Scientific Principles & Forensic Guardrails:
    1. Identity-Aware Priority:
       Partitions datasets strictly at the highest available identity level:
           Priority 1: Subject identity (`subject_id` when verified).
           Priority 2: Source-video identity (`source_video_id` when present).
           Priority 3: Video-level (`record_id` when no grouping exists).
    2. Paired Manipulation Containment:
       All manipulated derivatives of a source video (e.g., Deepfakes, FaceSwap, NeuralTextures)
       share the same `source_video_id` and MUST be partitioned into the identical subset.
    3. Strict Zero-Leakage Invariants:
       Enforces mathematical disjointness across subject IDs, source video IDs, and SHA-256 hashes:
           train_subjects ∩ val_subjects = ∅
           train_sources ∩ val_sources = ∅
           train_hashes ∩ val_hashes = ∅
    4. Honest Labeling:
       If any samples lack subject identifiers, the split level is explicitly downgraded to
       SOURCE_VIDEO_LEVEL or VIDEO_LEVEL. Never falsely claims subject-level separation.
"""

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple

from biotrace.core.logging import get_logger
from biotrace.experiments.domain import (
    DatasetManifest,
    DatasetSplitManifest,
    SampleStatus,
    SplitLevel,
    VideoRecord,
)

logger = get_logger(__name__)


class UnionFind:
    """Disjoint Set Union (DSU) data structure for transitive identity grouping."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


class IdentityAwareSplitter:
    """Partitions dataset manifest records into disjoint train, val, and test splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not (0.99 <= (train_ratio + val_ratio + test_ratio) <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio:.3f})"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_manifest(
        self,
        manifest: DatasetManifest,
        split_id: Optional[str] = None,
        include_only_valid: bool = True,
    ) -> DatasetSplitManifest:
        """Deterministically partitions manifest records preventing subject and source leakage."""
        # 1. Filter candidate records
        candidates = [
            r for r in manifest.records
            if (not include_only_valid or r.status == SampleStatus.VALID)
        ]

        if not candidates:
            raise ValueError(f"No eligible records found in manifest {manifest.manifest_id}")

        n = len(candidates)
        uf = UnionFind(n)

        # 2. Transitive Identity Clustering via Inverted Indices
        subj_to_idx: Dict[str, List[int]] = defaultdict(list)
        src_to_idx: Dict[str, List[int]] = defaultdict(list)
        hash_to_idx: Dict[str, List[int]] = defaultdict(list)

        for i, r in enumerate(candidates):
            if r.subject_id and len(str(r.subject_id).strip()) > 0:
                subj_to_idx[str(r.subject_id).strip()].append(i)
            if r.source_video_id and len(str(r.source_video_id).strip()) > 0:
                src_to_idx[str(r.source_video_id).strip()].append(i)
            if r.sha256_hash and r.sha256_hash not in ("MISSING_FILE", "EMPTY_FILE", "READ_ERROR"):
                hash_to_idx[r.sha256_hash].append(i)

        # Merge connected indices sharing subject_id
        for indices in subj_to_idx.values():
            for k in range(1, len(indices)):
                uf.union(indices[0], indices[k])

        # Merge connected indices sharing source_video_id
        for indices in src_to_idx.values():
            for k in range(1, len(indices)):
                uf.union(indices[0], indices[k])

        # Merge connected indices sharing identical media file hash
        for indices in hash_to_idx.values():
            for k in range(1, len(indices)):
                uf.union(indices[0], indices[k])

        # 3. Form atomic grouping clusters from connected components
        raw_clusters: Dict[int, List[VideoRecord]] = defaultdict(list)
        for i, r in enumerate(candidates):
            root = uf.find(i)
            raw_clusters[root].append(r)

        clusters: List[List[VideoRecord]] = []
        for cl in raw_clusters.values():
            cl.sort(key=lambda x: x.record_id)
            clusters.append(cl)

        # Canonical deterministic ordering by primary record_id in each cluster
        clusters.sort(key=lambda cl: cl[0].record_id)

        # Deterministic shuffle with seed
        rng = random.Random(self.seed)
        rng.shuffle(clusters)

        # 4. Determine verified split level guarantee
        all_have_subjects = all(
            r.subject_id is not None and len(str(r.subject_id).strip()) > 0
            for r in candidates
        )
        all_have_sources = all(
            (r.source_video_id is not None and len(str(r.source_video_id).strip()) > 0)
            or (r.subject_id is not None and len(str(r.subject_id).strip()) > 0)
            for r in candidates
        )

        if all_have_subjects:
            split_level = SplitLevel.SUBJECT_LEVEL
            notes = (
                f"Verified zero-leakage split at SUBJECT_LEVEL. "
                f"Total samples: {len(candidates)}, Clusters: {len(clusters)}."
            )
        elif all_have_sources:
            split_level = SplitLevel.SOURCE_VIDEO_LEVEL
            notes = (
                f"Split level downgraded to SOURCE_VIDEO_LEVEL because one or more records lack "
                f"verified human subject identifiers. Source-video disjointness verified. "
                f"Total samples: {len(candidates)}, Clusters: {len(clusters)}."
            )
        else:
            split_level = SplitLevel.VIDEO_LEVEL
            notes = (
                f"Split level downgraded to VIDEO_LEVEL because grouping identifiers (subject/source) "
                f"are not universally available across samples. Exact duplicate hash disjointness verified. "
                f"Total samples: {len(candidates)}, Clusters: {len(clusters)}."
            )

        logger.info(
            "Dataset %s partitioned into %d clusters at verified level: %s",
            manifest.dataset_id,
            len(clusters),
            split_level.value,
        )

        # 5. Deficit-based greedy cluster allocation to partitions
        total_samples = len(candidates)
        target_train = total_samples * self.train_ratio
        target_val = total_samples * self.val_ratio
        target_test = total_samples * self.test_ratio

        train_records: List[VideoRecord] = []
        val_records: List[VideoRecord] = []
        test_records: List[VideoRecord] = []

        start_idx = 0
        if len(clusters) >= 3:
            # Seed each partition with at least one cluster to avoid starvation
            train_records.extend(clusters[0])
            val_records.extend(clusters[1])
            test_records.extend(clusters[2])
            start_idx = 3

        for cluster in clusters[start_idx:]:
            d_train = target_train - len(train_records)
            d_val = target_val - len(val_records)
            d_test = target_test - len(test_records)

            best_bucket = max(
                [("train", d_train), ("val", d_val), ("test", d_test)],
                key=lambda x: x[1],
            )[0]

            if best_bucket == "train":
                train_records.extend(cluster)
            elif best_bucket == "val":
                val_records.extend(cluster)
            else:
                test_records.extend(cluster)

        # Extract record IDs and sort deterministically
        train_ids = sorted([r.record_id for r in train_records])
        val_ids = sorted([r.record_id for r in val_records])
        test_ids = sorted([r.record_id for r in test_records])

        # 6. Strict Disjointness and Zero-Leakage Verification
        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        assert train_set.isdisjoint(val_set), "Leakage detected: train and val share video records!"
        assert train_set.isdisjoint(test_set), "Leakage detected: train and test share video records!"
        assert val_set.isdisjoint(test_set), "Leakage detected: val and test share video records!"

        # Verify subject-level disjointness if applicable
        train_subjs = {r.subject_id for r in train_records if r.subject_id}
        val_subjs = {r.subject_id for r in val_records if r.subject_id}
        test_subjs = {r.subject_id for r in test_records if r.subject_id}
        assert train_subjs.isdisjoint(val_subjs), "Subject leakage detected between train and val!"
        assert train_subjs.isdisjoint(test_subjs), "Subject leakage detected between train and test!"
        assert val_subjs.isdisjoint(test_subjs), "Subject leakage detected between val and test!"

        # Verify source-video disjointness if applicable
        train_sources = {r.source_video_id for r in train_records if r.source_video_id}
        val_sources = {r.source_video_id for r in val_records if r.source_video_id}
        test_sources = {r.source_video_id for r in test_records if r.source_video_id}
        assert train_sources.isdisjoint(val_sources), "Source video leakage detected between train and val!"
        assert train_sources.isdisjoint(test_sources), "Source video leakage detected between train and test!"
        assert val_sources.isdisjoint(test_sources), "Source video leakage detected between val and test!"

        # Verify SHA-256 duplicate content disjointness
        train_hashes = {r.sha256_hash for r in train_records if r.sha256_hash and r.status == SampleStatus.VALID}
        val_hashes = {r.sha256_hash for r in val_records if r.sha256_hash and r.status == SampleStatus.VALID}
        test_hashes = {r.sha256_hash for r in test_records if r.sha256_hash and r.status == SampleStatus.VALID}
        assert train_hashes.isdisjoint(val_hashes), "Identical SHA-256 media found in both train and val!"
        assert train_hashes.isdisjoint(test_hashes), "Identical SHA-256 media found in both train and test!"
        assert val_hashes.isdisjoint(test_hashes), "Identical SHA-256 media found in both val and test!"

        s_id = split_id or f"split_{manifest.dataset_id.lower().replace('-', '_').replace('+', 'p')}_s{self.seed}"
        s_hash = DatasetSplitManifest.compute_split_hash(
            dataset_id=manifest.dataset_id,
            seed=self.seed,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        return DatasetSplitManifest(
            split_id=s_id,
            dataset_id=manifest.dataset_id,
            split_level=split_level,
            seed=self.seed,
            created_at=datetime.utcnow(),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            train_record_ids=train_ids,
            val_record_ids=val_ids,
            test_record_ids=test_ids,
            split_hash=s_hash,
            notes=notes,
        )

    @staticmethod
    def save_split(split_manifest: DatasetSplitManifest, output_path: str) -> None:
        """Serializes the partition split to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(split_manifest.model_dump_json(indent=2))
        logger.info("Saved dataset split manifest to %s (hash: %s)", out, split_manifest.split_hash[:12])

    @staticmethod
    def load_split(split_path: str) -> DatasetSplitManifest:
        """Loads a split manifest from JSON."""
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DatasetSplitManifest(**data)
