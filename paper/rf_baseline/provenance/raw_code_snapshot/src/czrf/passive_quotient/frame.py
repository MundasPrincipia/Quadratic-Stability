"""Restricted empirical frame audits on observable waveform secants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RestrictedFrameAudit:
    secant_count: int
    ambient_dimension: int
    analysis_dimension: int
    lower_bound: float
    upper_bound: float
    condition_number: float
    collision_count: int
    qualified: bool


def restricted_frame_audit(
    analysis: Any,
    secants: Any,
    *,
    collision_tolerance: float = 1e-12,
) -> RestrictedFrameAudit:
    """Evaluate ``inf ||Av||`` and ``sup ||Av||`` on normalized secants.

    The result is empirical unless ``secants`` exhaust a mathematically
    specified set.  The name does not turn a finite sample into a uniform
    theorem.
    """

    operator = np.asarray(analysis, dtype=np.complex128)
    values = np.asarray(secants, dtype=np.complex128)
    if operator.ndim != 2 or min(operator.shape) == 0:
        raise ValueError("analysis must be a nonempty matrix")
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("secants must be a nonempty row matrix")
    if operator.shape[1] != values.shape[1]:
        raise ValueError("analysis and secants have incompatible dimensions")
    if not np.all(np.isfinite(operator)) or not np.all(np.isfinite(values)):
        raise ValueError("analysis and secants must be finite")
    if collision_tolerance < 0.0 or not np.isfinite(collision_tolerance):
        raise ValueError("collision_tolerance must be finite and nonnegative")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(np.float64).tiny):
        raise ValueError("secants must be nonzero")
    unit = values / norms[:, None]
    transformed = (operator @ unit.T).T
    gains = np.linalg.norm(transformed, axis=1)
    lower = float(np.min(gains))
    upper = float(np.max(gains))
    collisions = int(np.sum(gains <= collision_tolerance))
    condition = float(np.inf if lower <= 0.0 else upper / lower)
    return RestrictedFrameAudit(
        secant_count=int(unit.shape[0]),
        ambient_dimension=int(operator.shape[1]),
        analysis_dimension=int(operator.shape[0]),
        lower_bound=lower,
        upper_bound=upper,
        condition_number=condition,
        collision_count=collisions,
        qualified=bool(lower > collision_tolerance),
    )
