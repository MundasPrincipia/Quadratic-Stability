"""Minimal fold-local preprocessing with an explicit group access ledger."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


def _canonical_groups(groups: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in groups}))


def _group_digest(groups: tuple[str, ...]) -> str:
    joined = "\n".join(groups).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


@dataclass(frozen=True)
class StandardizerState:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    fit_rows: int
    fit_groups: tuple[str, ...]
    fit_group_sha256: str

    def transform(self, values: Any) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.mean.size:
            raise ValueError("values have incompatible standardizer dimensions")
        if not np.all(np.isfinite(array)):
            raise ValueError("values must be finite")
        return (array - self.mean) / self.scale


def fit_group_standardizer(
    values: Any,
    group_ids: Iterable[Any],
    *,
    forbidden_groups: Iterable[Any] = (),
) -> StandardizerState:
    """Fit a z-score transform and reject any forbidden group overlap."""

    array = np.asarray(values, dtype=np.float64)
    groups_array = np.asarray(list(group_ids), dtype=object)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("values must be a nonempty row matrix")
    if groups_array.shape != (array.shape[0],):
        raise ValueError("one group identifier is required per row")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    fit_groups = _canonical_groups(groups_array.tolist())
    blocked = set(_canonical_groups(forbidden_groups))
    overlap = blocked.intersection(fit_groups)
    if overlap:
        raise ValueError(f"forbidden groups entered fit: {sorted(overlap)}")
    mean = np.mean(array, axis=0, dtype=np.float64)
    scale = np.std(array, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale > np.finfo(np.float64).tiny, scale, 1.0)
    mean.setflags(write=False)
    scale.setflags(write=False)
    return StandardizerState(
        mean=mean,
        scale=scale,
        fit_rows=int(array.shape[0]),
        fit_groups=fit_groups,
        fit_group_sha256=_group_digest(fit_groups),
    )
