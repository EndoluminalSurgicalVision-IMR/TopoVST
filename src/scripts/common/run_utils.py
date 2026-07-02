"""Shared helpers for `run_train.py` / `run_test.py`.

Keeps run-ID indexing, CSV parsing, and value coercion in one place so the
per-mode scripts stay focused on wiring config objects.
"""
import ast
import csv
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional


DATA_ROOT = ""
LOGS_ROOT = os.path.join(DATA_ROOT, "logs")
TESTS_ROOT = os.path.join(DATA_ROOT, "tests")
INDEX_DIR = os.path.join(DATA_ROOT, "index")
TRAIN_INDEX_PATH = os.path.join(INDEX_DIR, "train_index.json")
TEST_INDEX_PATH = os.path.join(INDEX_DIR, "test_index.json")


def load_row(csv_path: str, run_id: str) -> Dict[str, str]:
    """Return the CSV row whose ``run_id`` column matches ``run_id``."""

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_id", "").strip() == run_id:
                return {k: (v if v is not None else "") for k, v in row.items()}
    raise KeyError(f"run_id {run_id!r} not found in {csv_path}")


def to_float(v: str) -> float:
    return float(v)


def to_int(v: str) -> int:
    return int(v)


def to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def to_list(v: str):
    s = (v or "").strip()
    return ast.literal_eval(s) if s else []


def load_index(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@contextmanager
def _index_lock(index_path: str):
    """Hold an exclusive flock on a sidecar ``<index>.lock`` file.

    Why: ``update_index`` is invoked concurrently from parallel GPU workers
    launched by ``run_parse_ablations.sh`` / ``run_sequential.sh``. An unlocked
    read-modify-write race produced a corrupted JSON file (`Extra data` decode
    error). The lock serializes the RMW; ``os.replace`` makes the write itself
    atomic so lock-free readers (``load_index``) never observe a partial file.
    """

    lock_path = index_path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def update_index(
    index_path: str,
    run_id: str,
    save_dir: str,
    csv_row: Dict[str, str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or overwrite ``run_id`` in the JSON index at ``index_path``."""

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with _index_lock(index_path):
        idx = load_index(index_path)
        entry: Dict[str, Any] = {
            "save_dir": save_dir,
            "config_row": csv_row,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            entry.update(extra)
        idx[run_id] = entry
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(idx, f, indent=2)
        os.replace(tmp_path, index_path)


def device_index(device: str) -> int:
    """Extract the integer GPU index from a string like ``"cuda:0"``."""

    return int(device.rsplit(":", 1)[-1])
