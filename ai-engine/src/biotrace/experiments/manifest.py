"""Dataset manifest builder, serialization, and cryptographic verification.

Scientific Principles & Forensic Integrity:
    1. Deterministic Ordering: Manifest records are sorted canonically to guarantee
       that two manifest generations from identical files yield identical SHA-256 digests.
    2. Cryptographic Content Hashing: The manifest embeds a SHA-256 digest computed across
       all sample hashes and labels. Any modification or file corruption alters the digest.
    3. Duplicate Detection: Detects identical files via SHA-256 collisions within and across datasets.
    4. Safe Provenance: Paths are sanitized during serialization to protect local workstation privacy.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from biotrace.core.logging import get_logger
from biotrace.experiments.data_validation import VideoDataValidator
from biotrace.experiments.domain import DatasetManifest, SampleStatus, VideoRecord, sanitize_path_for_report

logger = get_logger(__name__)


class ManifestBuilder:
    """Builds and serializes reproducible dataset manifests from local video archives."""

    def __init__(
        self,
        validator: Optional[VideoDataValidator] = None,
    ):
        self.validator = validator or VideoDataValidator()

    def build_manifest(
        self,
        dataset_id: str,
        dataset_root: str,
        sample_descriptors: List[Dict[str, any]],
        manifest_id: Optional[str] = None,
    ) -> DatasetManifest:
        """Processes a list of sample descriptors, validates each video, and generates a manifest.

        Args:
            dataset_id: Canonical name of the dataset.
            dataset_root: Local root directory of the dataset.
            sample_descriptors: List of dicts specifying sample paths and known metadata:
                - "video_path": full or relative path
                - "record_id": optional unique identifier
                - "label": 0 or 1 if known
                - "source_video_id": optional original source ID
                - "subject_id": optional subject identifier
                - "manipulation_type": optional algorithm name
                - "compression": optional compression tag
                - "ground_truth_path": optional path to reference signal file
                - "metadata": optional additional key-value annotations
            manifest_id: Optional custom manifest identifier.
        """
        records: List[VideoRecord] = []
        seen_hashes: Dict[str, str] = {}  # sha256 -> record_id
        duplicate_count = 0

        root_path = Path(dataset_root).resolve() if dataset_root else Path(".")
        sanitized_root = sanitize_path_for_report(str(root_path))

        for desc in sample_descriptors:
            raw_path = desc.get("video_path", "")
            p = Path(raw_path)
            if not p.is_absolute() and dataset_root:
                full_path = str((root_path / p).resolve())
            else:
                full_path = str(p.resolve())

            record = self.validator.validate_file(
                file_path=full_path,
                dataset_id=dataset_id,
                record_id=desc.get("record_id"),
                dataset_root=str(root_path),
                label=desc.get("label"),
                source_video_id=desc.get("source_video_id"),
                subject_id=desc.get("subject_id"),
                manipulation_type=desc.get("manipulation_type"),
                compression=desc.get("compression"),
                ground_truth_path=desc.get("ground_truth_path"),
                extra_metadata=desc.get("metadata", {}),
            )

            # Check duplicate by SHA-256 hash (only if successfully computed)
            if record.sha256_hash and record.sha256_hash not in ("MISSING_FILE", "EMPTY_FILE", "READ_ERROR"):
                if record.sha256_hash in seen_hashes:
                    duplicate_count += 1
                    first_seen = seen_hashes[record.sha256_hash]
                    logger.warning(
                        "Duplicate file content detected: %s matches %s (SHA-256: %s)",
                        record.record_id,
                        first_seen,
                        record.sha256_hash[:12],
                    )
                    record.metadata["duplicate_of"] = first_seen
                else:
                    seen_hashes[record.sha256_hash] = record.record_id

            records.append(record)

        # Deterministic sorting by record_id
        records.sort(key=lambda r: r.record_id)

        valid_count = sum(1 for r in records if r.status == SampleStatus.VALID)
        corrupt_count = sum(1 for r in records if r.status in (SampleStatus.CORRUPT, SampleStatus.MISSING_FILE))
        unsupported_count = sum(1 for r in records if r.status in (SampleStatus.UNSUPPORTED, SampleStatus.DEGRADED))

        m_hash = DatasetManifest.compute_manifest_hash(records)
        m_id = manifest_id or f"manifest_{dataset_id.lower().replace('-', '_').replace('+', 'p')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        return DatasetManifest(
            manifest_id=m_id,
            dataset_id=dataset_id,
            created_at=datetime.utcnow(),
            dataset_root=sanitized_root,
            total_samples=len(records),
            valid_samples=valid_count,
            corrupt_samples=corrupt_count,
            unsupported_samples=unsupported_count,
            duplicate_samples_detected=duplicate_count,
            manifest_hash=m_hash,
            records=records,
        )

    @staticmethod
    def save_manifest(manifest: DatasetManifest, output_path: str) -> None:
        """Serializes the manifest to a formatted JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        logger.info("Saved dataset manifest to %s (hash: %s)", out, manifest.manifest_hash[:12])

    @staticmethod
    def load_manifest(manifest_path: str, verify_hash: bool = True) -> DatasetManifest:
        """Loads and optionally cryptographically validates a dataset manifest."""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = DatasetManifest(**data)
        if verify_hash:
            calculated_hash = DatasetManifest.compute_manifest_hash(manifest.records)
            if calculated_hash != manifest.manifest_hash:
                raise ValueError(
                    f"Manifest hash mismatch! Recorded: {manifest.manifest_hash}, "
                    f"Calculated from records: {calculated_hash}. Data may be corrupted."
                )
        return manifest
