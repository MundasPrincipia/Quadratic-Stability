"""Selective Maximal Justified Invariance decision rule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class MJIStatus(str, Enum):
    SELECT = "SELECT"
    NO_QUALIFIED = "NO-QUALIFIED-REPRESENTATION-UNDER-EVALUATOR"
    LEAKAGE_NOT_IDENTIFIABLE = "LEAKAGE-NOT-IDENTIFIABLE"
    DEFER = "DEFER-UNCERTAIN"


@dataclass(frozen=True)
class MJIThresholds:
    average_risk_margin: float
    tail_risk_margin: float
    leakage_limit: float

    def __post_init__(self) -> None:
        values = (
            self.average_risk_margin,
            self.tail_risk_margin,
            self.leakage_limit,
        )
        if not all(np.isfinite(values)) or any(value < 0.0 for value in values):
            raise ValueError("MJI thresholds must be finite and nonnegative")


@dataclass(frozen=True)
class MJIResult:
    status: MJIStatus
    selected: str | None
    certified: tuple[str, ...]
    reasons: tuple[str, ...]
    margins: Mapping[str, Mapping[str, float]]


def _finite_square(values: Any, size: int, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    return array


def select_mji(
    *,
    group_order: Sequence[str],
    expected_group_order: Sequence[str],
    average_risk_upper: Any,
    tail_risk_upper: Any,
    leakage_upper: Any | None,
    thresholds: MJIThresholds,
    contract_sha256: str,
    prediction_contract_sha256: str,
    leakage_identifiable: bool,
    absolute_utility_lower: Any | None = None,
    absolute_utility_floor: float | None = None,
) -> MJIResult:
    """Select the coarsest candidate certified against every comparator.

    Inputs are already simultaneous one-sided bounds.  The selector performs no
    plug-in uncertainty estimation and fails closed on stale provenance, group
    order changes, non-finite bounds, or unidentified mandatory leakage.
    """

    groups = tuple(str(value) for value in group_order)
    expected = tuple(str(value) for value in expected_group_order)
    if not contract_sha256 or prediction_contract_sha256 != contract_sha256:
        return MJIResult(
            MJIStatus.DEFER,
            None,
            (),
            ("stale-or-missing-provenance",),
            {},
        )
    if groups != expected or len(set(groups)) != len(groups):
        return MJIResult(MJIStatus.DEFER, None, (), ("group-order-mismatch",), {})
    size = len(groups)
    try:
        average = _finite_square(average_risk_upper, size, "average_risk_upper")
        tail = _finite_square(tail_risk_upper, size, "tail_risk_upper")
    except ValueError as error:
        return MJIResult(MJIStatus.DEFER, None, (), (str(error),), {})
    if not leakage_identifiable:
        return MJIResult(
            MJIStatus.LEAKAGE_NOT_IDENTIFIABLE,
            None,
            (),
            ("leakage-is-mandatory-but-not-identifiable",),
            {},
        )
    leakage = np.asarray(leakage_upper, dtype=np.float64)
    if leakage.shape != (size,) or not np.all(np.isfinite(leakage)):
        return MJIResult(MJIStatus.DEFER, None, (), ("invalid-leakage-upper",), {})
    margins: dict[str, dict[str, float]] = {}
    certified: list[str] = []
    for index, group in enumerate(groups):
        average_margin = thresholds.average_risk_margin - float(np.max(average[index]))
        tail_margin = thresholds.tail_risk_margin - float(np.max(tail[index]))
        leakage_margin = thresholds.leakage_limit - float(leakage[index])
        margins[group] = {
            "average": average_margin,
            "tail": tail_margin,
            "leakage": leakage_margin,
        }
        if min(average_margin, tail_margin, leakage_margin) >= 0.0:
            certified.append(group)
    if certified:
        return MJIResult(
            MJIStatus.SELECT,
            certified[0],
            tuple(certified),
            ("coarsest-simultaneously-certified",),
            margins,
        )
    if absolute_utility_lower is not None and absolute_utility_floor is not None:
        utility = np.asarray(absolute_utility_lower, dtype=np.float64)
        if (
            utility.shape == (size,)
            and np.all(np.isfinite(utility))
            and np.isfinite(absolute_utility_floor)
            and np.all(utility < float(absolute_utility_floor))
        ):
            return MJIResult(
                MJIStatus.NO_QUALIFIED,
                None,
                (),
                ("all-absolute-utility-lower-bounds-below-floor",),
                margins,
            )
    return MJIResult(
        MJIStatus.DEFER,
        None,
        (),
        ("no-candidate-simultaneously-certified",),
        margins,
    )

