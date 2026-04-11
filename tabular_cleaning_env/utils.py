"""Shared helpers for table normalization and deterministic comparison."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def clone_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [deepcopy(row) for row in rows]


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:.1f}".rstrip("0").rstrip(".")
    return str(value)


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def ordered_row(row: Dict[str, Any], columns: Sequence[str]) -> Dict[str, Any]:
    return {column: row.get(column) for column in columns}


def canonical_key(row: Dict[str, Any], key_fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(stringify(row.get(field)) for field in key_fields)


def canonical_sort(
    rows: Sequence[Dict[str, Any]],
    key_fields: Sequence[str],
    columns: Sequence[str],
) -> List[Dict[str, Any]]:
    return sorted(
        [ordered_row(row, columns) for row in rows],
        key=lambda row: (canonical_key(row, key_fields), stable_json(row)),
    )


def parse_datetime_like(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y/%m/%dT%H:%M",
    ]
    for candidate in candidates:
        try:
            return datetime.strptime(text, candidate)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_datetime_for_task(value: Any, include_time: bool) -> Optional[str]:
    parsed = parse_datetime_like(value)
    if parsed is None:
        return None
    if include_time:
        return parsed.strftime("%Y-%m-%dT%H:%M")
    return parsed.strftime("%Y-%m-%d")


def is_canonical_datetime_for_task(value: Any, include_time: bool) -> bool:
    canonical = format_datetime_for_task(value, include_time)
    if canonical is None:
        return False
    return str(value).strip() == canonical


def completeness_score(rows: Sequence[Dict[str, Any]], required_columns: Sequence[str]) -> float:
    if not rows or not required_columns:
        return 1
    total = len(rows) * len(required_columns)
    filled = 0
    for row in rows:
        for column in required_columns:
            if not is_missing(row.get(column)):
                filled += 1
    return filled / total


def count_non_missing(row: Dict[str, Any], columns: Iterable[str]) -> int:
    return sum(0 if is_missing(row.get(column)) else 1 for column in columns)


def coerce_dtype(value: Any, dtype: str) -> Any:
    if value is None:
        return None
    if dtype == "string":
        return str(value).strip()
    if dtype == "float":
        text = str(value).strip().replace("$", "").replace(",", "")
        if text == "":
            return None
        return round(float(text), 2)
    if dtype == "int":
        text = str(value).strip().replace(",", "")
        if text == "":
            return None
        return int(float(text))
    raise ValueError(f"Unsupported dtype: {dtype}")


def looks_like_email(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(EMAIL_RE.match(value.strip()))


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    required_columns: Sequence[str],
    duplicate_key_fields: Sequence[str] | None = None,
) -> Dict[str, Any]:
    duplicate_fields = list(duplicate_key_fields or [])
    duplicate_count = 0
    if duplicate_fields:
        counts = {}
        for row in rows:
            key = canonical_key(row, duplicate_fields)
            counts[key] = counts.get(key, 0) + 1
        duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "row_count": len(rows),
        "column_count": len(rows[0]) if rows else 0,
        "missing_required_cells": sum(
            1
            for row in rows
            for column in required_columns
            if column in row and is_missing(row.get(column))
        ),
        "duplicate_count": duplicate_count,
    }
