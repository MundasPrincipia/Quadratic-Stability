"""Canonical right-symmetry lattice for projective coefficient matrices.

The implementation follows the frozen Track S convention: coefficient matrices
are ``K x T``, rows are flattened in C/row-major order, and the DFT uses
``numpy.fft.fft(..., norm="ortho")``.  The explicit feature maps have Euclidean
inner products equal to the Hilbert--Schmidt kernels of the corresponding
trace-preserving conditional expectations.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .mechanism import coefficient_matrices, hermitian_hs_coordinates


RealArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _finite_coefficient_matrix(values: Any) -> ComplexArray:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("C must be a nonempty K x T matrix")
    if matrix.shape[1] < 2:
        raise ValueError("the symmetry chain requires T >= 2")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("C must be finite")
    norm_squared = float(np.vdot(matrix, matrix).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("C must have nonzero finite Frobenius norm")
    return matrix


def projective_state(C: Any, *, row_major: bool = True) -> ComplexArray:
    """Return ``vec(C) vec(C)* / ||C||_F^2`` under an explicit convention."""

    matrix = _finite_coefficient_matrix(C)
    vector = matrix.reshape(-1, order="C" if row_major else "F")
    norm_squared = float(np.vdot(vector, vector).real)
    return np.asarray(np.outer(vector, vector.conj()) / norm_squared)


def hermitian_coordinates(matrix: Any) -> RealArray:
    """Isometric real coordinates for one Hermitian matrix or a matrix batch."""

    return np.asarray(hermitian_hs_coordinates(matrix), dtype=np.float64)


def dihedral_frequency_orbits(window_count: int) -> tuple[tuple[int, ...], ...]:
    """Return the orbits of ``omega -> -omega`` in canonical order."""

    if window_count < 2:
        raise ValueError("window_count must be at least two")
    seen: set[int] = set()
    output: list[tuple[int, ...]] = []
    for omega in range(window_count):
        if omega in seen:
            continue
        partner = (-omega) % window_count
        orbit = tuple(sorted({omega, partner}))
        seen.update(orbit)
        output.append(orbit)
    return tuple(output)


def _fourier_outer_blocks(C: Any) -> tuple[ComplexArray, ComplexArray]:
    matrix = _finite_coefficient_matrix(C)
    norm_squared = float(np.vdot(matrix, matrix).real)
    fourier = np.fft.fft(matrix, axis=1, norm="ortho")
    blocks = np.einsum(
        "kw,lw->wkl", fourier, fourier.conj(), optimize=True
    ) / norm_squared
    gram = matrix @ matrix.conj().T / norm_squared
    return np.asarray(blocks), np.asarray(gram)


def _fourier_projectors(window_count: int) -> ComplexArray:
    # q_w[t] = exp(+2 pi i t w/T)/sqrt(T), hence q_w^* c is numpy's FFT.
    basis = np.fft.ifft(np.eye(window_count), axis=0, norm="ortho")
    return np.asarray(
        np.einsum("tw,sw->wts", basis, basis.conj(), optimize=True),
        dtype=np.complex128,
    )


def unitary_twirl_closed_form(C: Any) -> ComplexArray:
    """Closed Haar twirl over the complete right unitary group ``U(T)``."""

    matrix = _finite_coefficient_matrix(C)
    _, gram = _fourier_outer_blocks(matrix)
    return np.asarray(np.kron(gram, np.eye(matrix.shape[1]) / matrix.shape[1]))


def permutation_twirl_closed_form(C: Any) -> ComplexArray:
    """Closed twirl over the full right permutation group ``S_T``."""

    matrix = _finite_coefficient_matrix(C)
    blocks, gram = _fourier_outer_blocks(matrix)
    window_count = matrix.shape[1]
    q0 = np.ones((window_count, window_count), dtype=np.complex128) / window_count
    q1 = np.eye(window_count, dtype=np.complex128) - q0
    a0 = blocks[0]
    a1 = gram - a0
    return np.asarray(np.kron(a0, q0) + np.kron(a1 / (window_count - 1), q1))


def cyclic_twirl_fourier(C: Any) -> ComplexArray:
    """Closed cyclic twirl using the frozen orthonormal FFT convention."""

    matrix = _finite_coefficient_matrix(C)
    blocks, _ = _fourier_outer_blocks(matrix)
    projectors = _fourier_projectors(matrix.shape[1])
    output = np.zeros(
        (matrix.size, matrix.size), dtype=np.complex128
    )
    for block, projector in zip(blocks, projectors, strict=True):
        output += np.kron(block, projector)
    return np.asarray(output)


def cyclic_twirl_lag(C: Any) -> ComplexArray:
    """Independent time-lag construction of the cyclic conditional expectation."""

    matrix = _finite_coefficient_matrix(C)
    rows, window_count = matrix.shape
    norm_squared = float(np.vdot(matrix, matrix).real)
    output = np.empty((rows * window_count, rows * window_count), dtype=np.complex128)
    for first in range(rows):
        for second in range(rows):
            block = np.empty((window_count, window_count), dtype=np.complex128)
            for left in range(window_count):
                for right in range(window_count):
                    lag = (right - left) % window_count
                    block[left, right] = np.sum(
                        matrix[first] * np.roll(matrix[second].conj(), -lag)
                    ) / (window_count * norm_squared)
            output[
                first * window_count : (first + 1) * window_count,
                second * window_count : (second + 1) * window_count,
            ] = block
    return output


def dihedral_twirl_fourier(C: Any) -> ComplexArray:
    """Closed dihedral twirl, pairing Fourier frequencies ``omega`` and ``-omega``."""

    matrix = _finite_coefficient_matrix(C)
    blocks, _ = _fourier_outer_blocks(matrix)
    projectors = _fourier_projectors(matrix.shape[1])
    output = np.zeros((matrix.size, matrix.size), dtype=np.complex128)
    for orbit in dihedral_frequency_orbits(matrix.shape[1]):
        spatial = np.sum(blocks[np.asarray(orbit)], axis=0) / len(orbit)
        temporal = np.sum(projectors[np.asarray(orbit)], axis=0)
        output += np.kron(spatial, temporal)
    return np.asarray(output)


def right_permutation_twirl(P: Any, permutation: Sequence[int]) -> ComplexArray:
    """Conjugate an operator by one declared right column permutation."""

    operator = np.asarray(P, dtype=np.complex128)
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or order.size < 2 or not np.array_equal(
        np.sort(order), np.arange(order.size)
    ):
        raise ValueError("permutation must contain 0,...,T-1 exactly once")
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("P must be square")
    if operator.shape[0] % order.size:
        raise ValueError("P dimension must be divisible by T")
    rows = operator.shape[0] // order.size
    right = np.eye(order.size, dtype=np.complex128)[:, order]
    action = np.kron(np.eye(rows, dtype=np.complex128), right.T)
    return np.asarray(action @ operator @ action.conj().T)


def finite_twirl_enumerated(C: Any, group: str) -> ComplexArray:
    """Enumerate a finite exact twirl used only by small algebra audits.

    ``group='U'`` uses the finite Weyl unitary one-design, which reproduces the
    first Haar moment exactly.  ``S`` enumerates all permutations; ``C`` and
    ``D`` enumerate cyclic and dihedral actions.
    """

    matrix = _finite_coefficient_matrix(C)
    rows, window_count = matrix.shape
    state = projective_state(matrix)
    actions: list[ComplexArray] = []
    identity = np.eye(window_count, dtype=np.complex128)
    if group == "U":
        omega = np.exp(2j * np.pi / window_count)
        phase = np.diag(omega ** np.arange(window_count))
        shift = np.roll(identity, 1, axis=0)
        for first in range(window_count):
            for second in range(window_count):
                actions.append(np.linalg.matrix_power(shift, first) @ np.linalg.matrix_power(phase, second))
    elif group == "S":
        actions = [identity[:, np.asarray(order)] for order in permutations(range(window_count))]
    elif group in {"C", "D"}:
        reflection = identity[:, (-np.arange(window_count)) % window_count]
        for shift_count in range(window_count):
            cyclic = np.roll(identity, shift_count, axis=0)
            actions.append(cyclic)
            if group == "D":
                actions.append(cyclic @ reflection)
    else:
        raise ValueError("group must be one of U, S, C, D")
    # Form transformed rank-one states directly.  This is algebraically
    # identical to ``(I_K kron R.T) P (I_K kron R.T)*`` under row-major
    # vectorization, while keeping the T=8 full-permutation audit inexpensive.
    output = np.zeros_like(state)
    for right in actions:
        vector = (matrix @ right).reshape(-1, order="C")
        output += np.outer(vector, vector.conj()) / float(np.vdot(vector, vector).real)
    return np.asarray(output / len(actions))


def _feature_bank(values: Any, *, matrix_rows: int) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    matrices = coefficient_matrices(values, matrix_rows=matrix_rows)
    fourier = np.fft.fft(matrices, axis=2, norm="ortho")
    blocks = np.einsum(
        "nkw,nlw->nwkl", fourier, fourier.conj(), optimize=True
    )
    gram = np.sum(blocks, axis=1)
    return matrices, np.asarray(blocks), np.asarray(gram)


def symmetry_chain_feature_bank(values: Any, *, matrix_rows: int) -> dict[str, RealArray]:
    """Return explicit U/S/D/C feature maps for normalized row-major inputs."""

    matrices, blocks, gram = _feature_bank(values, matrix_rows=matrix_rows)
    sample_count, _, window_count = matrices.shape
    matrix_dimension = matrix_rows * matrix_rows
    u = hermitian_coordinates(gram) / np.sqrt(window_count)
    s = np.concatenate(
        (
            hermitian_coordinates(blocks[:, 0]),
            hermitian_coordinates(gram - blocks[:, 0]) / np.sqrt(window_count - 1),
        ),
        axis=1,
    )
    dihedral_parts = []
    for orbit in dihedral_frequency_orbits(window_count):
        spatial = np.sum(blocks[:, np.asarray(orbit)], axis=1)
        dihedral_parts.append(hermitian_coordinates(spatial) / np.sqrt(len(orbit)))
    d = np.concatenate(dihedral_parts, axis=1)
    c = hermitian_coordinates(blocks.reshape(-1, matrix_rows, matrix_rows)).reshape(
        sample_count, window_count * matrix_dimension
    )
    return {
        "U": np.asarray(u, dtype=np.float64),
        "S": np.asarray(s, dtype=np.float64),
        "D": np.asarray(d, dtype=np.float64),
        "C": np.asarray(c, dtype=np.float64),
    }


def symmetry_chain_features(C: Any) -> dict[str, RealArray]:
    """Single-matrix convenience wrapper around :func:`symmetry_chain_feature_bank`."""

    matrix = _finite_coefficient_matrix(C)
    output = symmetry_chain_feature_bank(matrix.reshape(1, -1), matrix_rows=matrix.shape[0])
    return {name: values[0] for name, values in output.items()}


def singleton_partition_feature_bank(
    values: Any, *, matrix_rows: int, singleton_frequency: int
) -> RealArray:
    """Matched two-block feature using one declared Fourier singleton."""

    # Avoid materializing the full N x T x K x K Fourier outer-product bank.
    # Gate 1 needs only one singleton and its orthogonal complement; this form
    # keeps the S1 peak memory bounded by a few N x K x K arrays.
    matrices = coefficient_matrices(values, matrix_rows=matrix_rows)
    window_count = matrices.shape[2]
    singleton = int(singleton_frequency) % window_count
    fourier = np.fft.fft(matrices, axis=2, norm="ortho")
    selected_vector = fourier[:, :, singleton]
    selected = np.einsum(
        "nk,nl->nkl", selected_vector, selected_vector.conj(), optimize=True
    )
    gram = np.zeros_like(selected)
    for frequency in range(window_count):
        vector = fourier[:, :, frequency]
        gram += np.einsum(
            "nk,nl->nkl", vector, vector.conj(), optimize=True
        )
    return np.concatenate(
        (
            hermitian_coordinates(selected),
            hermitian_coordinates(gram - selected) / np.sqrt(window_count - 1),
        ),
        axis=1,
    ).astype(np.float64, copy=False)


def symmetry_chain_kernels(
    rows: Any, landmarks: Any, *, matrix_rows: int
) -> dict[str, RealArray]:
    """Return row-to-landmark U/S/D/C/I kernel blocks."""

    left_features = symmetry_chain_feature_bank(rows, matrix_rows=matrix_rows)
    right_features = symmetry_chain_feature_bank(landmarks, matrix_rows=matrix_rows)
    output = {
        name: np.asarray(left_features[name] @ right_features[name].T, dtype=np.float64)
        for name in ("U", "S", "D", "C")
    }
    left = coefficient_matrices(rows, matrix_rows=matrix_rows).reshape(len(np.asarray(rows)), -1)
    right = coefficient_matrices(landmarks, matrix_rows=matrix_rows).reshape(
        len(np.asarray(landmarks)), -1
    )
    output["I"] = np.asarray(np.abs(left @ right.conj().T) ** 2, dtype=np.float64)
    return output


def symmetry_band_kernels(
    rows: Any, landmarks: Any, *, matrix_rows: int
) -> dict[str, RealArray]:
    """Return the four adjacent PSD band kernel blocks and their reconstruction."""

    chain = symmetry_chain_kernels(rows, landmarks, matrix_rows=matrix_rows)
    return {
        "S-U": np.asarray(chain["S"] - chain["U"]),
        "D-S": np.asarray(chain["D"] - chain["S"]),
        "C-D": np.asarray(chain["C"] - chain["D"]),
        "I-C": np.asarray(chain["I"] - chain["C"]),
    }


def chain_twirl_closed_forms(C: Any) -> Mapping[str, ComplexArray]:
    """Return all five conditional expectations for a single matrix."""

    matrix = _finite_coefficient_matrix(C)
    return {
        "U": unitary_twirl_closed_form(matrix),
        "S": permutation_twirl_closed_form(matrix),
        "D": dihedral_twirl_fourier(matrix),
        "C": cyclic_twirl_fourier(matrix),
        "I": projective_state(matrix),
    }


def operator_twirl_closed_form(
    operator: Any, *, matrix_rows: int, group: str
) -> ComplexArray:
    """Apply one chain conditional expectation to an arbitrary operator."""

    values = np.asarray(operator, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("operator must be square")
    if matrix_rows <= 0 or values.shape[0] % matrix_rows:
        raise ValueError("matrix_rows must divide the operator dimension")
    window_count = values.shape[0] // matrix_rows
    if window_count < 2:
        raise ValueError("the symmetry chain requires T >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("operator must be finite")
    if group == "I":
        return values.copy()
    projectors = _fourier_projectors(window_count)
    q0 = projectors[0]
    q1 = np.eye(window_count, dtype=np.complex128) - q0
    output = np.zeros_like(values)
    for first in range(matrix_rows):
        for second in range(matrix_rows):
            block = values[
                first * window_count : (first + 1) * window_count,
                second * window_count : (second + 1) * window_count,
            ]
            if group == "U":
                averaged = np.trace(block) * np.eye(window_count) / window_count
            elif group == "S":
                averaged = (
                    np.trace(q0 @ block) * q0
                    + np.trace(q1 @ block) * q1 / (window_count - 1)
                )
            elif group == "C":
                averaged = np.zeros_like(block)
                for projector in projectors:
                    averaged += np.trace(projector @ block) * projector
            elif group == "D":
                averaged = np.zeros_like(block)
                for orbit in dihedral_frequency_orbits(window_count):
                    temporal = np.sum(projectors[np.asarray(orbit)], axis=0)
                    coefficient = sum(
                        np.trace(projectors[index] @ block) for index in orbit
                    ) / len(orbit)
                    averaged += coefficient * temporal
            else:
                raise ValueError("group must be one of U, S, D, C, I")
            output[
                first * window_count : (first + 1) * window_count,
                second * window_count : (second + 1) * window_count,
            ] = averaged
    return np.asarray(output)


def strictness_witnesses(window_count: int = 7) -> Mapping[str, tuple[ComplexArray, ComplexArray]]:
    """Construct deterministic witnesses for every strict adjacent inclusion."""

    if window_count < 5:
        raise ValueError("strictness witnesses require T >= 5")
    basis = np.fft.ifft(np.eye(window_count), axis=0, norm="ortho")
    # U != S: equal norm/left Gram, different common-mode energy.
    us = (basis[:, 0][None, :], basis[:, 1][None, :])
    # S != D: equal zero/nonzero energy, different dihedral orbit.
    sd = (basis[:, 1][None, :], basis[:, 2][None, :])
    # D != C: exchange +omega and -omega inside one orbit.
    dc = (basis[:, 1][None, :], basis[:, -1][None, :])
    first = np.zeros(window_count, dtype=np.complex128)
    second = np.zeros(window_count, dtype=np.complex128)
    first[1] = first[2] = 1.0 / np.sqrt(2.0)
    second[1] = 1.0 / np.sqrt(2.0)
    second[2] = 1j / np.sqrt(2.0)
    ci = (
        np.fft.ifft(first, norm="ortho")[None, :],
        np.fft.ifft(second, norm="ortho")[None, :],
    )
    return {"U-S": us, "S-D": sd, "D-C": dc, "C-I": ci}
