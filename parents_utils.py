from pathlib import Path
from typing import Dict, Iterable, List

from storage_utils import load_json_locked, write_json_locked

PARENTS_F = Path("parents.json")


def _normalize_chats(chats: Iterable) -> List[int]:
    seen = set()
    normalized: List[int] = []
    for cid in chats or []:
        try:
            val = int(cid)
        except (TypeError, ValueError):
            continue
        if val not in seen:
            seen.add(val)
            normalized.append(val)
    return normalized


def load_parents(path: Path = PARENTS_F) -> Dict[str, List[int]]:
    data = load_json_locked(path, default={})
    if not isinstance(data, dict):
        return {}
    result: Dict[str, List[int]] = {}
    for sid, chats in data.items():
        sid_str = str(sid).strip()
        if not sid_str:
            continue
        normalized = _normalize_chats(chats)
        if normalized:
            result[sid_str] = normalized
    return result


def save_parents(parents: Dict[str, List[int]], path: Path = PARENTS_F):
    cleaned = {}
    for sid, chats in parents.items():
        sid_str = str(sid).strip()
        if not sid_str:
            continue
        normalized = _normalize_chats(chats)
        if normalized:
            cleaned[sid_str] = normalized
    write_json_locked(path, cleaned)


def remove_chat_binding(student_id: str, chat_id: int, path: Path = PARENTS_F) -> bool:
    parents = load_parents(path)
    if student_id not in parents:
        return False
    chats = [c for c in parents[student_id] if c != chat_id]
    if chats:
        parents[student_id] = chats
    else:
        parents.pop(student_id)
    save_parents(parents, path)
    return True
