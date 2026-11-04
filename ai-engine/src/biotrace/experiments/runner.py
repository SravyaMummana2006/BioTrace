"""Reproducible research experiment orchestrator, runner, and result serialiser.

Scientific Principles & Forensic Integrity:
    1. Exact Reproducibility: Every experiment run outputs a machine-readable JSON result
       and human-readable Markdown summary linking the experiment configuration,
       random seed, dataset manifest hash, split hash, and checkpoint digest.
    2. Zero Fabrication & Honest Availability: When a dataset is not authorized or present,
       the runner records an explicit AWAITING_DATASET_ACCESS record detailing the
       authorisation requirements, never substituting fabricated numbers.
    3. Failure Recovery: Individual video errors are logged and counted in failure metrics
       without aborting the entire benchmark suite.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

from biotrace.core.logging import get_logger
from biotrace.experiments.dataset_registry import DatasetRegistry
from biotrace.experiments.domain import (
    DataSourceType,
    DatasetIdentifier,
    DatasetManifest,
    DatasetSplitManifest,
    ExperimentStatus,
)
from biotrace.experiments.experiment_a import ExperimentARunner
from biotrace.experiments.experiment_b import ExperimentBRunner
from biotrace.experiments.experiment_c import ExperimentCRunner
from biotrace.experiments.experiment_config import (
    BaseExperimentConfig,
    ExperimentAConfig,
    ExperimentBConfig,
    ExperimentCConfig,
    ExperimentDConfig,
    ExperimentEConfig,
)
from biotrace.experiments.experiment_d import ExperimentDRunner
from biotrace.experiments.experiment_e import ExperimentERunner

logger = get_logger(__name__)


def get_system_provenance() -> Dict[str, str]:
    """Captures runtime environment versions for strict provenance."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "os_platform": platform.platform(),
    }


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across Python, NumPy, and PyTorch runtimes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ResearchExperimentRunner:
    """Central orchestrator executing and logging reproducible forensic experiments."""

    def __init__(
        self,
        results_dir: str = "experiments/results",
        datasets_base_dir: str = "datasets",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.registry = DatasetRegistry(datasets_base_dir=datasets_base_dir)

    def run_experiment_a(
        self,
        config: Optional[ExperimentAConfig] = None,
        manifest: Optional[DatasetManifest] = None,
    ) -> Dict[str, Any]:
        """Executes Experiment A (rPPG validation)."""
        cfg = config or ExperimentAConfig()
        set_seed(cfg.seed)

        # Check dataset availability if manifest not directly provided
        if manifest is None:
            status, info = self.registry.get_dataset_status(cfg.target_dataset)
            if status != ExperimentStatus.DATASET_AVAILABLE:
                return self._record_unavailable_experiment(cfg.experiment_id, cfg.target_dataset, info)

        runner = ExperimentARunner(cfg)
        result = runner.run_experiment(manifest=manifest)
        self._save_results(cfg.experiment_id, result)
        return result

    def run_experiment_b(
        self,
        config: Optional[ExperimentBConfig] = None,
        manifest: Optional[DatasetManifest] = None,
        split_manifest: Optional[DatasetSplitManifest] = None,
    ) -> Dict[str, Any]:
        """Executes Experiment B (Visual detection baseline)."""
        cfg = config or ExperimentBConfig()
        set_seed(cfg.seed)

        if manifest is None:
            status, info = self.registry.get_dataset_status(cfg.target_dataset)
            if status != ExperimentStatus.DATASET_AVAILABLE:
                return self._record_unavailable_experiment(cfg.experiment_id, cfg.target_dataset, info)

        runner = ExperimentBRunner(cfg)
        result = runner.run_experiment(manifest=manifest, split_manifest=split_manifest)
        self._save_results(cfg.experiment_id, result)
        return result

    def run_experiment_c(
        self,
        config: Optional[ExperimentCConfig] = None,
        source_manifest: Optional[DatasetManifest] = None,
        target_manifest: Optional[DatasetManifest] = None,
    ) -> Dict[str, Any]:
        """Executes Experiment C (Cross-dataset generalization)."""
        cfg = config or ExperimentCConfig()
        set_seed(cfg.seed)

        if source_manifest is None or target_manifest is None:
            status, info = self.registry.get_dataset_status(cfg.target_dataset)
            if status != ExperimentStatus.DATASET_AVAILABLE:
                return self._record_unavailable_experiment(cfg.experiment_id, cfg.target_dataset, info)

        runner = ExperimentCRunner(cfg)
        result = runner.run_experiment(source_manifest=source_manifest, target_manifest=target_manifest)
        self._save_results(cfg.experiment_id, result)
        return result

    def run_experiment_d(
        self,
        config: Optional[ExperimentDConfig] = None,
        manifest: Optional[DatasetManifest] = None,
    ) -> Dict[str, Any]:
        """Executes Experiment D (Physiological evidence under manipulation)."""
        cfg = config or ExperimentDConfig()
        set_seed(cfg.seed)

        if manifest is None:
            status, info = self.registry.get_dataset_status(cfg.target_dataset)
            if status != ExperimentStatus.DATASET_AVAILABLE:
                return self._record_unavailable_experiment(cfg.experiment_id, cfg.target_dataset, info)

        runner = ExperimentDRunner(cfg)
        result = runner.run_experiment(manifest=manifest)
        self._save_results(cfg.experiment_id, result)
        return result

    def run_experiment_e(
        self,
        config: Optional[ExperimentEConfig] = None,
        manifest: Optional[DatasetManifest] = None,
        split_manifest: Optional[DatasetSplitManifest] = None,
    ) -> Dict[str, Any]:
        """Executes Experiment E (Multimodal evidence evaluation)."""
        cfg = config or ExperimentEConfig()
        set_seed(cfg.seed)

        if manifest is None:
            status, info = self.registry.get_dataset_status(cfg.target_dataset)
            if status != ExperimentStatus.DATASET_AVAILABLE:
                return self._record_unavailable_experiment(cfg.experiment_id, cfg.target_dataset, info)

        runner = ExperimentERunner(cfg)
        result = runner.run_experiment(manifest=manifest, split_manifest=split_manifest)
        self._save_results(cfg.experiment_id, result)
        return result

    def _record_unavailable_experiment(
        self,
        experiment_id: str,
        target_dataset: str,
        details: Optional[str],
    ) -> Dict[str, Any]:
        """Constructs an explicit AWAITING_DATASET_ACCESS result record."""
        record = {
            "experiment_id": experiment_id,
            "status": ExperimentStatus.AWAITING_DATASET_ACCESS.value,
            "target_dataset": target_dataset,
            "details": details or "Awaiting user approval and dataset download.",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": None,
        }
        self._save_results(experiment_id, record)
        return record

    def _save_results(self, experiment_id: str, results: Dict[str, Any]) -> Tuple[Path, Path]:
        """Saves machine-readable JSON results and human-readable Markdown summaries."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"{experiment_id.lower()}_{ts}"

        # Embed system provenance
        results["system_provenance"] = get_system_provenance()
        if "data_source_type" not in results:
            results["data_source_type"] = DataSourceType.EMPIRICAL_BENCHMARK.value

        # 1. JSON
        json_path = self.results_dir / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Markdown summary
        md_path = self.results_dir / f"{base_name}.md"
        md_content = self._generate_markdown_summary(results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info("Saved experiment results to %s and %s", json_path.name, md_path.name)
        return json_path, md_path

    def _generate_markdown_summary(self, res: Dict[str, Any]) -> str:
        """Renders structured Markdown from experiment outputs."""
        exp_id = res.get("experiment_id", "UNKNOWN_EXPERIMENT")
        status = res.get("status", "COMPLETED")
        ts = res.get("timestamp", datetime.utcnow().isoformat())
        source_type = res.get("data_source_type", DataSourceType.EMPIRICAL_BENCHMARK.value)

        md = [
            f"# BIOTrace Research Experiment Report — {exp_id}",
            f"**Execution Timestamp:** `{ts}`  ",
            f"**Status:** `{status}`  ",
            f"**Data Source Classification:** `{source_type}`  \n",
        ]

        if status == ExperimentStatus.AWAITING_DATASET_ACCESS.value:
            md.extend([
                "## Status Notice: Awaiting Dataset Access",
                f"> **Target Dataset:** {res.get('target_dataset')}",
                f"> **Details:** {res.get('details')}",
                "\n*In adherence to strict research governance, experiments do not execute on fabricated or unlicensed data.*",
            ])
            return "\n".join(md)

        # Metrics Table
        metrics = res.get("metrics")
        if metrics:
            md.extend([
                "## Summary Performance Metrics",
                "```json",
                json.dumps(metrics, indent=2),
                "```\n",
            ])

        # Configuration & Provenance
        cfg = res.get("config")
        sys_prov = res.get("system_provenance", {})
        md.extend([
            "## Configuration & Cryptographic Provenance",
            f"- **Random Seed:** `{cfg.get('seed', 42) if cfg else 'N/A'}`",
            f"- **Dataset Manifest Hash:** `{res.get('manifest_hash', 'N/A')}`",
            f"- **Split Manifest Hash:** `{res.get('split_hash', 'N/A')}`",
            f"- **Checkpoint Digest:** `{res.get('checkpoint_digest', 'N/A')}`",
            f"- **Python Version:** `{sys_prov.get('python_version', 'N/A')}`",
            f"- **PyTorch Version:** `{sys_prov.get('torch_version', 'N/A')}`",
            f"- **OpenCV Version:** `{sys_prov.get('opencv_version', 'N/A')}`",
            f"- **Operating System:** `{sys_prov.get('os_platform', 'N/A')}`\n",
        ])

        return "\n".join(md)
