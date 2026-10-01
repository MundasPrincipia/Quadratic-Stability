"""Frozen, label-blind source adapter for the downloaded Oregon LoRa subset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


SAMPLE_RATE_HZ = 1_000_000
BANDWIDTH_HZ = 125_000
SYMBOL_SAMPLES = 1024
FRAME_SEGMENTS = 8
CORRELATION_THRESHOLD = 0.99
MINIMUM_CORRELATION_RUN = 6
MINIMUM_UP_DECHIRP_SCORE = 0.04
MINIMUM_UP_DOWN_RATIO = 5.0
MINIMUM_FRAME_RMS = 1.0e-6
MINIMUM_SEPARATION_SYMBOLS = 128


@dataclass(frozen=True)
class OregonPreambleFrame:
    start_sample: int
    start_block: int
    correlation_run_length: int
    correlation_min: float
    correlation_median: float
    frame_rms: float
    up_dechirp_min: float
    up_dechirp_median: float
    down_dechirp_median: float
    up_over_down_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_oregon_complex64(path: str | Path) -> np.memmap:
    """Memory-map the publisher's interleaved little-endian Float32 I/Q."""

    source = Path(path)
    if not source.is_file() or source.stat().st_size % np.dtype("<c8").itemsize:
        raise ValueError(f"invalid Oregon LoRa raw-IQ file: {source}")
    return np.memmap(source, dtype="<c8", mode="r")


def adjacent_symbol_correlations(
    values: NDArray[np.complexfloating], *, chunk_blocks: int = 2048
) -> NDArray[np.float64]:
    """Compute normalized lag-one correlations on the frozen 1024-sample grid."""

    iq = np.asarray(values)
    if iq.ndim != 1 or iq.size < 2 * SYMBOL_SAMPLES:
        raise ValueError("raw IQ must contain at least two symbols")
    block_count = iq.size // SYMBOL_SAMPLES
    blocks = iq[: block_count * SYMBOL_SAMPLES].reshape(block_count, SYMBOL_SAMPLES)
    output = np.empty(block_count - 1, dtype=np.float64)
    for begin in range(0, block_count - 1, chunk_blocks):
        stop = min(begin + chunk_blocks, block_count - 1)
        left = np.asarray(blocks[begin:stop], dtype=np.complex64)
        right = np.asarray(blocks[begin + 1 : stop + 1], dtype=np.complex64)
        numerator = np.abs(
            np.sum(np.conj(left) * right, axis=1, dtype=np.complex128)
        )
        left_energy = np.sum(np.abs(left) ** 2, axis=1, dtype=np.float64)
        right_energy = np.sum(np.abs(right) ** 2, axis=1, dtype=np.float64)
        denominator = np.sqrt(left_energy * right_energy)
        output[begin:stop] = np.divide(
            numerator,
            denominator,
            out=np.zeros(stop - begin, dtype=np.float64),
            where=denominator > 0,
        )
    return output


def _true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(stop))
        for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def _ideal_upchirp() -> NDArray[np.complex128]:
    sample = np.arange(SYMBOL_SAMPLES, dtype=np.float64)
    time = sample / SAMPLE_RATE_HZ
    duration = SYMBOL_SAMPLES / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * (
        -0.5 * BANDWIDTH_HZ * time
        + 0.5 * BANDWIDTH_HZ / duration * time**2
    )
    return np.exp(1j * phase)


def _dechirp_score(
    segment: NDArray[np.complexfloating], reference: NDArray[np.complex128]
) -> float:
    values = np.asarray(segment, dtype=np.complex128)
    rms = float(np.sqrt(np.mean(np.abs(values) ** 2)))
    if not np.isfinite(rms) or rms <= 0:
        return 0.0
    spectrum = np.abs(np.fft.fft(values / rms * np.conj(reference), n=4096)) ** 2
    return float(np.max(spectrum) / np.sum(spectrum))


def detect_oregon_preamble_frames(
    values: NDArray[np.complexfloating],
) -> list[OregonPreambleFrame]:
    """Detect fixed 8-segment upchirp/sync frame units without using labels."""

    iq = np.asarray(values)
    if iq.ndim != 1 or not np.all(np.isfinite(iq)):
        raise ValueError("raw IQ must be one-dimensional and finite")
    correlations = adjacent_symbol_correlations(iq)
    block_count = iq.size // SYMBOL_SAMPLES
    blocks = iq[: block_count * SYMBOL_SAMPLES].reshape(block_count, SYMBOL_SAMPLES)
    up = _ideal_upchirp()
    down = np.conj(up)
    accepted: list[OregonPreambleFrame] = []
    last_start = -MINIMUM_SEPARATION_SYMBOLS
    for run_start, run_stop in _true_runs(correlations >= CORRELATION_THRESHOLD):
        run_length = run_stop - run_start
        if run_length < MINIMUM_CORRELATION_RUN:
            continue
        if run_start - last_start < MINIMUM_SEPARATION_SYMBOLS:
            continue
        segments = np.asarray(
            blocks[run_start : run_start + FRAME_SEGMENTS], dtype=np.complex64
        )
        if segments.shape != (FRAME_SEGMENTS, SYMBOL_SAMPLES):
            continue
        frame_rms = float(
            np.sqrt(np.mean(np.abs(segments.astype(np.complex128, copy=False)) ** 2))
        )
        up_scores = np.asarray([_dechirp_score(row, up) for row in segments])
        down_scores = np.asarray([_dechirp_score(row, down) for row in segments])
        up_median = float(np.median(up_scores))
        down_median = float(np.median(down_scores))
        ratio = up_median / max(down_median, np.finfo(float).tiny)
        if (
            frame_rms < MINIMUM_FRAME_RMS
            or float(np.min(up_scores)) < MINIMUM_UP_DECHIRP_SCORE
            or ratio < MINIMUM_UP_DOWN_RATIO
        ):
            continue
        accepted.append(
            OregonPreambleFrame(
                start_sample=run_start * SYMBOL_SAMPLES,
                start_block=run_start,
                correlation_run_length=run_length,
                correlation_min=float(np.min(correlations[run_start:run_stop])),
                correlation_median=float(np.median(correlations[run_start:run_stop])),
                frame_rms=frame_rms,
                up_dechirp_min=float(np.min(up_scores)),
                up_dechirp_median=up_median,
                down_dechirp_median=down_median,
                up_over_down_ratio=ratio,
            )
        )
        last_start = run_start
    return accepted


def materialize_oregon_frames(
    values: NDArray[np.complexfloating], frames: list[OregonPreambleFrame]
) -> NDArray[np.complex64]:
    """Copy detected frame segments into the frozen frontend tensor layout."""

    iq = np.asarray(values)
    output = np.empty(
        (len(frames), FRAME_SEGMENTS, SYMBOL_SAMPLES), dtype=np.complex64
    )
    for index, frame in enumerate(frames):
        stop = frame.start_sample + FRAME_SEGMENTS * SYMBOL_SAMPLES
        segment = iq[frame.start_sample:stop]
        if segment.size != FRAME_SEGMENTS * SYMBOL_SAMPLES:
            raise ValueError("detected frame exceeds raw-IQ bounds")
        output[index] = np.asarray(segment, dtype=np.complex64).reshape(
            FRAME_SEGMENTS, SYMBOL_SAMPLES
        )
    if output.size and not np.all(np.isfinite(output)):
        raise ValueError("materialized frame contains non-finite IQ")
    return output

