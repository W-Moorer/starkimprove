#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    seen: set[str] = set()
    raw_candidates = [
        os.environ.get("CONDA_PREFIX", ""),
        sys.prefix,
        str(Path(sys.executable).resolve().parent.parent),
    ]
    for raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw)
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        prefixes.append(path)
    return prefixes


def configure_windows_dll_search() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    seen: set[str] = set()
    for prefix in _candidate_prefixes():
        dll_dir = prefix / "Library" / "bin"
        key = str(dll_dir).lower()
        if key in seen or not dll_dir.exists():
            continue
        os.add_dll_directory(str(dll_dir))
        seen.add(key)


def resolve_chrono_data_dir() -> Path:
    override = os.environ.get("STARK_CHRONO_DATA_DIR", "")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    for prefix in _candidate_prefixes():
        candidates.append(prefix / "Library" / "data")
    candidates.append(Path(r"E:\Anaconda\envs\chrono-baseline\Library\data"))

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else Path.cwd()
