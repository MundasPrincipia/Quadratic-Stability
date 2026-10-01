"""Batched ordered-twirl RF adapter and exact pairwise quotient geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ordered_twirl_stability import ordered_twirl_holder_constant
from .passive_quotient.ordered_filtration import discrete_polynomial_basis


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def _stable_close_pair_squares(
    first_x: ComplexArray,
    first_y: ComplexArray,
    second_x: ComplexArray,
    second_y: ComplexArray,
) -> tuple[float, float]:
    """Evaluate a nearly coincident pair without subtractive cancellation.

    The fast batched formulas below subtract almost equal trace terms.  That is
    accurate away from zero, but can turn a true zero into a distance of order
    ``sqrt(machine epsilon)``.  Direct phase/Procrustes alignment and direct
    Gram differences retain genuine very small distances instead of imposing a
    numerical dead zone.
    """

    if first_x.size:
        inner = np.vdot(first_x, second_x)
        phase = np.conj(inner) / abs(inner) if abs(inner) else 1.0 + 0.0j
        resolved_quotient_square = float(
            np.linalg.norm(first_x - phase * second_x) ** 2
        )
        first_resolved_gram = np.outer(first_x, first_x.conj())
        second_resolved_gram = np.outer(second_x, second_x.conj())
        resolved_output_square = float(
            np.linalg.norm(first_resolved_gram - second_resolved_gram, ord="fro")
            ** 2
        )
    else:
        resolved_quotient_square = 0.0
        resolved_output_square = 0.0
    residual_dimension = int(first_y.shape[1])
    if residual_dimension:
        left, _, right_h = np.linalg.svd(
            first_y.conj().T @ second_y,
            full_matrices=False,
        )
        unitary = right_h.conj().T @ left.conj().T
        residual_quotient_square = float(
            np.linalg.norm(first_y - second_y @ unitary, ord="fro") ** 2
        )
        first_residual_gram = first_y @ first_y.conj().T
        second_residual_gram = second_y @ second_y.conj().T
        residual_output_square = float(
            np.linalg.norm(first_residual_gram - second_residual_gram, ord="fro")
            ** 2
            / residual_dimension
        )
    else:
        residual_quotient_square = 0.0
        residual_output_square = 0.0
    return (
        resolved_quotient_square + residual_quotient_square,
        resolved_output_square + residual_output_square,
    )


def ordered_twirl_factors(
    values: Any,
    *,
    matrix_rows: int,
    time_length: int,
    level: int,
    basis: Any | None = None,
) -> tuple[ComplexArray, ComplexArray]:
    """Normalize RF rows and split them into resolved vectors/residual matrices."""

    rows = int(matrix_rows)
    length = int(time_length)
    selected = int(level)
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 1:
        array = array[None, :]
    if (
        rows < 1
        or length < 1
        or selected < -1
        or selected >= length
        or array.ndim != 2
        or array.shape[1] != rows * length
    ):
        raise ValueError("invalid RF array, tensor dimensions, or level")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("RF rows must be finite")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("RF rows must be nonzero")
    unit = array / norms[:, None]
    ordered = (
        discrete_polynomial_basis(length).astype(np.complex128)
        if basis is None
        else np.asarray(basis, dtype=np.complex128)
    )
    if ordered.shape != (length, length):
        raise ValueError("basis has the wrong shape")
    gram_error = float(
        np.linalg.norm(ordered.conj().T @ ordered - np.eye(length), ord=2)
    )
    if not np.isfinite(gram_error) or gram_error > 1e-10:
        raise ValueError("basis must be unitary")
    adapted = unit.reshape(array.shape[0], rows, length) @ ordered.conj()
    resolved_dimension = selected + 1
    resolved = adapted[:, :, :resolved_dimension].reshape(
        array.shape[0], rows * resolved_dimension
    )
    residual = adapted[:, :, resolved_dimension:]
    return (
        np.asarray(resolved, dtype=np.complex128),
        np.asarray(residual, dtype=np.complex128),
    )


def ordered_twirl_pairwise_geometry(
    values: Any,
    prototypes: Any,
    *,
    matrix_rows: int,
    time_length: int,
    level: int,
    basis: Any | None = None,
    batch_size: int = 128,
) -> dict[str, RealArray]:
    """Return exact quotient, output, and global Holder-bound matrices.

    Rows index ``values`` and columns index ``prototypes``.  The implementation
    uses only resolved inner products and batched SVDs of at most ``T x T``
    residual cross-Gram matrices; it never materializes the ambient
    ``(K*T) x (K*T)`` projectors.
    """

    first_x, first_y = ordered_twirl_factors(
        values,
        matrix_rows=matrix_rows,
        time_length=time_length,
        level=level,
        basis=basis,
    )
    second_x, second_y = ordered_twirl_factors(
        prototypes,
        matrix_rows=matrix_rows,
        time_length=time_length,
        level=level,
        basis=basis,
    )
    if second_x.shape[0] < 1:
        raise ValueError("at least one prototype is required")
    selected_batch = int(batch_size)
    if selected_batch < 1:
        raise ValueError("batch_size must be positive")
    count = first_x.shape[0]
    prototype_count = second_x.shape[0]
    quotient = np.empty((count, prototype_count), dtype=np.float64)
    output = np.empty_like(quotient)
    residual_dimension = int(time_length) - (int(level) + 1)
    first_x_norm = np.einsum("ni,ni->n", first_x.conj(), first_x).real
    second_x_norm = np.einsum("pi,pi->p", second_x.conj(), second_x).real
    first_y_norm = np.einsum("nkt,nkt->n", first_y.conj(), first_y).real
    second_y_norm = np.einsum("pkt,pkt->p", second_y.conj(), second_y).real
    if residual_dimension:
        first_self = np.einsum(
            "nkt,nku->ntu", first_y.conj(), first_y, optimize=True
        )
        second_self = np.einsum(
            "pkt,pku->ptu", second_y.conj(), second_y, optimize=True
        )
        first_gram_frobenius_square = np.sum(
            np.abs(first_self) ** 2, axis=(1, 2)
        )
        second_gram_frobenius_square = np.sum(
            np.abs(second_self) ** 2, axis=(1, 2)
        )
    else:
        first_gram_frobenius_square = np.zeros(count, dtype=np.float64)
        second_gram_frobenius_square = np.zeros(
            prototype_count, dtype=np.float64
        )

    for start in range(0, count, selected_batch):
        stop = min(count, start + selected_batch)
        local_x = first_x[start:stop]
        resolved_inner = local_x.conj() @ second_x.T
        resolved_quotient_square = (
            first_x_norm[start:stop, None]
            + second_x_norm[None, :]
            - 2.0 * np.abs(resolved_inner)
        )
        resolved_output_square = (
            first_x_norm[start:stop, None] ** 2
            + second_x_norm[None, :] ** 2
            - 2.0 * np.abs(resolved_inner) ** 2
        )
        if residual_dimension:
            cross = np.einsum(
                "nkt,pku->nptu",
                first_y[start:stop].conj(),
                second_y,
                optimize=True,
            )
            singular = np.linalg.svd(cross, compute_uv=False)
            residual_quotient_square = (
                first_y_norm[start:stop, None]
                + second_y_norm[None, :]
                - 2.0 * np.sum(singular, axis=-1)
            )
            cross_frobenius_square = np.sum(np.abs(cross) ** 2, axis=(2, 3))
            residual_output_square = (
                first_gram_frobenius_square[start:stop, None]
                + second_gram_frobenius_square[None, :]
                - 2.0 * cross_frobenius_square
            ) / residual_dimension
        else:
            residual_quotient_square = 0.0
            residual_output_square = 0.0
        quotient_square = resolved_quotient_square + residual_quotient_square
        output_square = resolved_output_square + residual_output_square
        # Re-evaluate only cancellation-prone pairs.  This is deliberately not
        # a threshold-to-zero rule: a genuine 1e-10 distance remains 1e-10.
        close_pairs = np.argwhere(
            (quotient_square <= 1e-12) | (output_square <= 1e-12)
        )
        for local_row, column in close_pairs:
            stable_quotient_square, stable_output_square = (
                _stable_close_pair_squares(
                    first_x[start + int(local_row)],
                    first_y[start + int(local_row)],
                    second_x[int(column)],
                    second_y[int(column)],
                )
            )
            quotient_square[local_row, column] = stable_quotient_square
            output_square[local_row, column] = stable_output_square
        quotient[start:stop] = np.sqrt(np.maximum(0.0, quotient_square))
        output[start:stop] = np.sqrt(np.maximum(0.0, output_square))
    holder = (
        ordered_twirl_holder_constant(matrix_rows, time_length, level)
        * np.sqrt(output)
    )
    return {
        "quotient_distance": quotient,
        "output_frobenius_distance": output,
        "holder_upper": np.asarray(holder, dtype=np.float64),
    }
