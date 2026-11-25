"""Parsing and provenance fields of the ``.sysinfo`` sidecar files.

MTL5's benchmarks write one ``key=value`` sidecar next to every data file
(``native-t1.json`` has ``native-t1.json.sysinfo``). The sidecar records the
machine, the build and the machine state the run was taken in. This module
parses it and reduces it to the fields the comparability key is built from.
"""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "REQUIRED_KEYS",
    "parse_sysinfo",
    "machine_fingerprint",
    "machine_identity",
    "state_class",
    "build_config",
    "backend",
    "short_hash",
]

#: Sidecar keys a run needs for its provenance to count as complete. A run
#: missing any of them is ingested in the ``incomplete`` tier, which trend
#: queries exclude.
REQUIRED_KEYS = (
    # what ran
    "benchmark",
    "label",
    "machine",
    "class",
    # machine identity
    "cpu_vendor",
    "cpu_brand",
    "cpu_arch",
    "cpu_logical_cores",
    "preflight_host",
    # build
    "git_commit",
    "git_dirty",
    "compiler",
    "compiler_version",
    "cpp_standard",
    "build_type",
    "cxx_flags",
    "build_isa",
    "simd_backend",
    "os_name",
    # threads
    "pool_size",
    # machine state
    "governor",
    "turbo",
    "power_mode",
    "preflight_gates",
)


def parse_sysinfo(text: str) -> tuple[dict[str, str], list[str]]:
    """Parse a sidecar into a dict, plus a list of problems found.

    Blank lines and lines starting with ``#`` are skipped. A line without
    ``=`` is a problem. So is a key repeated with a different value: which
    value describes the run is then unknown. A key repeated with the same
    value is harmless and is not reported.
    """
    fields: dict[str, str] = {}
    problems: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            problems.append(f"sidecar line {lineno} is not key=value")
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key in fields and fields[key] != value:
            problems.append(f"sidecar key {key!r} has conflicting values")
        fields[key] = value
    return fields, problems


def short_hash(text: str) -> str:
    """The first 16 hex digits of the SHA-256 of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _host(fields: dict[str, str]) -> str:
    # sw-13 and sw-13.stillwater-sc.com are the same box.
    return fields.get("preflight_host", "").strip().lower().split(".")[0]


def machine_identity(fields: dict[str, str]) -> str:
    """The physical machine, as a readable string.

    It is built from the hardware and the host, not from the ``machine``
    label: the label is chosen by whoever ran the benchmark, and the
    committed data spells one machine more than one way (``ryzen-9-8945hs``
    and ``AMD Ryzen 9 8945HS w/ Radeon 780M Graphics``). The host is part of
    it because two boxes with the same CPU are two machines: the data has two
    Xeon E5-2420 v2 hosts.
    """
    parts = [
        fields.get("cpu_vendor", ""),
        fields.get("cpu_brand", ""),
        fields.get("cpu_arch", ""),
        f"{fields.get('cpu_logical_cores', '')} cpus",
        f"host {_host(fields)}",
    ]
    return " | ".join(parts)


def machine_fingerprint(fields: dict[str, str]) -> str:
    """A short hash of :func:`machine_identity`."""
    return short_hash(machine_identity(fields))


def state_class(fields: dict[str, str]) -> str:
    """The machine state a run was taken in, as a readable string.

    Governor, turbo, power mode, the CPUs the benchmark was allowed and
    pinned to, and whether the preflight gates were enforced. Thermals and
    load average are not part of it: they are recorded per run as
    measurements, not as the class a run belongs to.
    """
    parts = [
        f"governor {fields.get('governor', '?')}",
        f"turbo {fields.get('turbo', '?')}",
        f"power {fields.get('power_mode', '?')}",
        f"affinity {fields.get('cpu_affinity', '?')}",
        f"bench_cpus {fields.get('bench_cpus', '?')}",
        f"gates {fields.get('preflight_gates', '?')}",
    ]
    return " | ".join(parts)


def build_config(fields: dict[str, str]) -> str:
    """The build, excluding the commit, as a readable string.

    The commit is the trend axis, so it is left out. Everything else that
    changes the generated code is in: compiler and version, language
    standard, build type, flags, target ISA, SIMD backend, and the OS.
    """
    flags = re.sub(r"\s+", " ", fields.get("cxx_flags", "")).strip()
    parts = [
        f"{fields.get('compiler', '?')} {fields.get('compiler_version', '?')}",
        f"c++{fields.get('cpp_standard', '?')}",
        fields.get("build_type", "?"),
        f"flags {flags or '-'}",
        f"isa {fields.get('build_isa', '?')}",
        f"simd {fields.get('simd_backend', '?')}",
        fields.get("os_name", "?"),
    ]
    return " | ".join(parts)


def backend(label: str, fields: dict[str, str]) -> str:
    """The backend: the run label (native, native-fast, blas) and BLAS vendor."""
    vendor = fields.get("blas_vendor", "")
    return f"{label} ({vendor})" if vendor else label
