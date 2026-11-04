"""Spatial Biological Consistency research module for multi-region rPPG pulse analysis."""

from biotrace.spatial_consistency.domain import (
    FacialRegionId,
    GlobalConsistencyState,
    PairConsistencyState,
    PairwiseComparison,
    RegionalSample,
    RegionalSignalSummary,
    RegionTrackingStatus,
    SpatialConsistencyConfig,
    SpatialConsistencyResult,
)
from biotrace.spatial_consistency.extractor import SpatialConsistencyExtractor
from biotrace.spatial_consistency.quality import (
    assess_regional_signal,
    compute_cardiac_snr,
    extract_pos_pulse_from_rgb,
)
from biotrace.spatial_consistency.region_extractor import (
    FacialRegionTracker,
    FOREHEAD_LANDMARKS,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
)
from biotrace.spatial_consistency.spatial_analyzer import (
    SpatialBiologicalAnalyzer,
    compare_region_pair,
    compute_lagged_cross_correlation,
    compute_spectral_overlap,
)

__all__ = [
    "FacialRegionId",
    "RegionTrackingStatus",
    "PairConsistencyState",
    "GlobalConsistencyState",
    "SpatialConsistencyConfig",
    "RegionalSample",
    "RegionalSignalSummary",
    "PairwiseComparison",
    "SpatialConsistencyResult",
    "FacialRegionTracker",
    "SpatialBiologicalAnalyzer",
    "SpatialConsistencyExtractor",
    "extract_pos_pulse_from_rgb",
    "compute_cardiac_snr",
    "assess_regional_signal",
    "compare_region_pair",
    "compute_lagged_cross_correlation",
    "compute_spectral_overlap",
    "FOREHEAD_LANDMARKS",
    "LEFT_CHEEK_LANDMARKS",
    "RIGHT_CHEEK_LANDMARKS",
]
