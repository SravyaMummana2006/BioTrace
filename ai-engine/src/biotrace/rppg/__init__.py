"""BIOTrace rPPG Research Package - Optical Physiological Signal Extraction."""

from biotrace.rppg.algorithms import extract_chrom_rppg, extract_pos_rppg
from biotrace.rppg.domain import (
    AlgorithmProvenance,
    FilterConfig,
    PulseEstimate,
    QualityConfig,
    QualityFlag,
    QualityGrade,
    RPPGQualityReport,
    WindowConfig,
)
from biotrace.rppg.extractor import RPPGSignalExtractor
from biotrace.rppg.filtering import (
    apply_bandpass_filter,
    compute_welch_psd,
    detrend_signal,
    find_dominant_cardiac_peak,
)
from biotrace.rppg.quality import assess_signal_quality, calculate_de_haan_snr

__all__ = [
    "RPPGSignalExtractor",
    "extract_chrom_rppg",
    "extract_pos_rppg",
    "assess_signal_quality",
    "calculate_de_haan_snr",
    "detrend_signal",
    "apply_bandpass_filter",
    "compute_welch_psd",
    "find_dominant_cardiac_peak",
    "WindowConfig",
    "FilterConfig",
    "QualityConfig",
    "QualityGrade",
    "QualityFlag",
    "PulseEstimate",
    "AlgorithmProvenance",
    "RPPGQualityReport",
]
