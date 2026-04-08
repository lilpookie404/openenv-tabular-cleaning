"""Deterministic graders for tabular cleaning tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

from .tasks import TaskDefinition, load_task_expected
from .utils import canonical_key, canonical_sort, completeness_score, format_datetime_for_task, ordered_row, stringify


def _normalize_rows_for_task(rows: Sequence[Dict[str, Any]], task: TaskDefinition) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row = ordered_row(row, task.expected_columns)
        for column, include_time in task.date_columns.items():
            if column in normalized_row:
                converted = format_datetime_for_task(normalized_row[column], include_time)
                normalized_row[column] = converted if converted is not None else normalized_row[column]
        normalized.append(normalized_row)
    return normalized


def _schema_score(rows: Sequence[Dict[str, Any]], task: TaskDefinition) -> float:
    if not rows:
        return 0.0
    columns = list(rows[0].keys())
    expected = list(task.expected_columns)
    if columns == expected:
        return 1.0
    matches = sum(1 for idx, column in enumerate(expected) if idx < len(columns) and columns[idx] == column)
    return matches / max(len(expected), 1)


def _row_alignment_score(
    current_rows: Sequence[Dict[str, Any]],
    expected_rows: Sequence[Dict[str, Any]],
    task: TaskDefinition,
) -> float:
    current_counts = Counter(canonical_key(row, task.primary_key) for row in current_rows)
    expected_counts = Counter(canonical_key(row, task.primary_key) for row in expected_rows)
    all_keys = set(current_counts) | set(expected_counts)
    if not all_keys:
        return 1.0
    total_diff = sum(abs(current_counts[key] - expected_counts[key]) for key in all_keys)
    max_total = max(sum(expected_counts.values()), 1)
    return max(0.0, 1.0 - (total_diff / max_total))


def _cell_match_score(
    current_rows: Sequence[Dict[str, Any]],
    expected_rows: Sequence[Dict[str, Any]],
    task: TaskDefinition,
) -> float:
    if not expected_rows:
        return 1.0
    current_by_key = {canonical_key(row, task.primary_key): row for row in current_rows}
    matches = 0
    total = len(expected_rows) * len(task.expected_columns)
    for expected_row in expected_rows:
        key = canonical_key(expected_row, task.primary_key)
        current_row = current_by_key.get(key, {})
        for column in task.expected_columns:
            if stringify(current_row.get(column)) == stringify(expected_row.get(column)):
                matches += 1
    return matches / total if total else 1.0


def _temporal_score(
    current_rows: Sequence[Dict[str, Any]],
    expected_rows: Sequence[Dict[str, Any]],
    task: TaskDefinition,
) -> float:
    if not task.date_columns:
        return 1.0
    current_by_key = {canonical_key(row, task.primary_key): row for row in current_rows}
    matches = 0
    total = len(expected_rows) * len(task.date_columns)
    for expected_row in expected_rows:
        key = canonical_key(expected_row, task.primary_key)
        current_row = current_by_key.get(key, {})
        for column, include_time in task.date_columns.items():
            current_value = format_datetime_for_task(current_row.get(column), include_time)
            expected_value = format_datetime_for_task(expected_row.get(column), include_time)
            if stringify(current_value) == stringify(expected_value):
                matches += 1
    return matches / total if total else 1.0


def grade_table(task: TaskDefinition, rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    raw_current_rows = [dict(row) for row in rows]
    expected_rows = _normalize_rows_for_task(load_task_expected(task.task_id), task)
    current_rows = _normalize_rows_for_task(rows, task)
    current_rows = canonical_sort(current_rows, task.primary_key, task.expected_columns)
    expected_rows = canonical_sort(expected_rows, task.primary_key, task.expected_columns)

    schema = _schema_score(raw_current_rows, task)
    rows_score = _row_alignment_score(current_rows, expected_rows, task)
    cell_score = _cell_match_score(current_rows, expected_rows, task)
    completeness = completeness_score(current_rows, task.required_columns)
    temporal = _temporal_score(current_rows, expected_rows, task)

    final_score = (
        (0.15 * schema)
        + (0.20 * rows_score)
        + (0.40 * cell_score)
        + (0.15 * completeness)
        + (0.10 * temporal)
    )
    return {
        "schema_score": round(schema, 6),
        "row_score": round(rows_score, 6),
        "cell_score": round(cell_score, 6),
        "completeness_score": round(completeness, 6),
        "temporal_score": round(temporal, 6),
        "score": round(min(max(final_score, 0.0), 1.0), 6),
    }
