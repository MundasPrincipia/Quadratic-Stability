"""Fail-closed provenance helpers for Track S."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def sha256_file(path: str | Path, *, block_size: int = 16 * 1024 * 1024) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def contract_digest(payload: Mapping[str, Any]) -> str:
    stripped = {key: value for key, value in payload.items() if key != "contract_sha256"}
    return hashlib.sha256(canonical_json_bytes(stripped)).hexdigest()


def attach_contract_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    output["contract_sha256"] = contract_digest(output)
    return output


def validate_contract_digest(payload: Mapping[str, Any]) -> bool:
    declared = payload.get("contract_sha256")
    return isinstance(declared, str) and declared == contract_digest(payload)


def string_array_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def incidence_identifiability(device: Any, environment: Any) -> dict[str, Any]:
    devices = np.asarray(device)
    environments = np.asarray(environment)
    if devices.ndim != 1 or environments.ndim != 1 or devices.size != environments.size:
        raise ValueError("device and environment must be aligned one-dimensional arrays")
    unique_devices = np.unique(devices)
    unique_environments = np.unique(environments)
    device_environment_counts = {
        str(item): int(np.unique(environments[devices == item]).size)
        for item in unique_devices
    }
    environment_device_counts = {
        str(item): int(np.unique(devices[environments == item]).size)
        for item in unique_environments
    }
    same_device_cross_environment = any(value >= 2 for value in device_environment_counts.values())
    same_environment_cross_device = any(value >= 2 for value in environment_device_counts.values())
    return {
        "identifiable": bool(same_device_cross_environment and same_environment_cross_device),
        "same_device_cross_environment": bool(same_device_cross_environment),
        "same_environment_cross_device": bool(same_environment_cross_device),
        "device_count": int(unique_devices.size),
        "environment_count": int(unique_environments.size),
        "minimum_environments_per_device": int(min(device_environment_counts.values())),
        "maximum_environments_per_device": int(max(device_environment_counts.values())),
        "minimum_devices_per_environment": int(min(environment_device_counts.values())),
        "maximum_devices_per_environment": int(max(environment_device_counts.values())),
    }


def validate_parent_hashes(root: str | Path, parents: Mapping[str, Mapping[str, Any]]) -> list[str]:
    base = Path(root)
    failures: list[str] = []
    for name, specification in parents.items():
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            failures.append(f"{name}:missing")
        elif sha256_file(path) != specification["sha256"]:
            failures.append(f"{name}:sha256")
    return failures

