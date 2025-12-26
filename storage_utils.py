import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import fcntl


@contextmanager
def locked_open(path: Path, mode: str = "r", shared: bool = True, **kwargs):
    """
    Open a file with an advisory lock.

    shared=True  -> read lock (LOCK_SH)
    shared=False -> write lock (LOCK_EX)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_obj = path.open(mode, encoding=kwargs.pop("encoding", "utf-8"), **kwargs)
    lock = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    fcntl.flock(file_obj.fileno(), lock)
    try:
        yield file_obj
        if not shared:
            file_obj.flush()
            os.fsync(file_obj.fileno())
    finally:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        file_obj.close()


def load_json_locked(path: Path, default: Optional[Any] = None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        with locked_open(path, "r", shared=True) as f:
            return json.load(f)
    except Exception:
        return default


def write_json_locked(path: Path, data: Any):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with locked_open(tmp, "w", shared=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
