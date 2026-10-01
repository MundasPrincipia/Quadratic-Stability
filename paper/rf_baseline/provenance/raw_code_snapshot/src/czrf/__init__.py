"""Reproducible finite-dimensional RF fingerprint simulation utilities."""

from .core import (
    add_awgn,
    apply_channel,
    build_dataset,
    centroid_separation_metrics,
    channel_bound_diagnostic,
    generate_device_bank,
    generate_probe_bank,
    grouped_split,
    memory_polynomial,
    nearest_centroid_metrics,
    normalized_probe_features,
)
from .experiments import (
    ExperimentDataset,
    build_experiment_dataset,
    evaluate_cross_domain,
    nominal_channel_residual,
    stratified_group_split,
)

__all__ = [
    "add_awgn",
    "apply_channel",
    "build_dataset",
    "centroid_separation_metrics",
    "channel_bound_diagnostic",
    "generate_device_bank",
    "generate_probe_bank",
    "grouped_split",
    "memory_polynomial",
    "nearest_centroid_metrics",
    "normalized_probe_features",
    "ExperimentDataset",
    "build_experiment_dataset",
    "evaluate_cross_domain",
    "nominal_channel_residual",
    "stratified_group_split",
]
