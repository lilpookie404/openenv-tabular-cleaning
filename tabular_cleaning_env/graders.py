"""Deterministic scalar grader for tabular cleaning tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .tasks import TaskDefinition, load_task_expected
from .utils import canonical_sort, ordered_row, stringify

SCORE_MIN = 0.1
SCORE_MAX = 0.9
NUMERIC_TOLERANCE = 1e-6


def _grade_columns(task: TaskDefinition) -> List[str]:
    return list(task.grade_columns or task.expected_columns)


def _blank_row(columns: Sequence[str]) -> Dict[str, Any]:
    return {column: None for column in columns}


def _normalize_rows_for_grading(
    rows: Sequence[Dict[str, Any]],
    task: TaskDefinition,
    columns: Sequence[str],
) -> List[Dict[str, Any]]:
    normalized = [ordered_row(dict(row), columns) for row in rows]
    if task.sort_rows:
        sort_fields = list(task.recommended_sort or task.primary_key or columns)
        return canonical_sort(normalized, sort_fields, columns)
    return normalized


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _values_match(column: str, current_value: Any, expected_value: Any, task: TaskDefinition) -> bool:
    if column in task.cast_columns:
        current_numeric = _coerce_numeric(current_value)
        expected_numeric = _coerce_numeric(expected_value)
        if current_numeric is not None and expected_numeric is not None:
            return abs(current_numeric - expected_numeric) <= NUMERIC_TOLERANCE
    return stringify(current_value) == stringify(expected_value)


def _raw_match_ratio(task: TaskDefinition, rows: Sequence[Dict[str, Any]]) -> float:
    columns = _grade_columns(task)
    if not columns:
        return 1.0

    current_rows = _normalize_rows_for_grading(rows, task, columns)
    expected_rows = _normalize_rows_for_grading(load_task_expected(task.task_id), task, columns)
    total_rows = max(len(current_rows), len(expected_rows), 1)
    total_cells = total_rows * len(columns)
    if total_cells == 0:
        return 1.0

    matches = 0
    blank_row = _blank_row(columns)
    for index in range(total_rows):
        current_row = current_rows[index] if index < len(current_rows) else blank_row
        expected_row = expected_rows[index] if index < len(expected_rows) else blank_row
        for column in columns:
            if _values_match(column, current_row.get(column), expected_row.get(column), task):
                matches += 1
    return matches / total_cells


def clamp_task_score(value: float) -> float:
    return round(min(max(float(value), SCORE_MIN), SCORE_MAX), 6)


def grade_task(task: TaskDefinition, rows: Sequence[Dict[str, Any]]) -> float:
    return clamp_task_score(_raw_match_ratio(task, rows))
