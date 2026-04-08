"""Environment implementation for deterministic tabular data cleaning."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Sequence
from uuid import uuid4

from tabular_cleaning_env.graders import grade_table
from tabular_cleaning_env.models import (
    ActionType,
    CaseMode,
    TabularCleaningAction,
    TabularCleaningObservation,
    TabularCleaningState,
)
from tabular_cleaning_env.openenv_compat import Environment
from tabular_cleaning_env.tasks import DuplicateRule, TaskDefinition, get_task, load_task_input
from tabular_cleaning_env.utils import (
    canonical_key,
    clone_rows,
    coerce_dtype,
    count_non_missing,
    format_datetime_for_task,
    is_missing,
    stable_json,
)


class TabularCleaningEnvironment(Environment):
    """OpenEnv-compatible environment for cleaning messy tabular data."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task_id: str = "easy_contacts_cleanup"):
        super().__init__()
        self._default_task_id = default_task_id
        self._task = get_task(default_task_id)
        self._table: List[Dict[str, Any]] = []
        self._state = TabularCleaningState()
        self._preview_limit = 5
        self._last_action: Dict[str, Any] | None = None
        self._last_action_error: str | None = None
        self.reset(task_id=default_task_id)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> TabularCleaningObservation:
        del seed
        selected_task = get_task(task_id or self._default_task_id)
        self._task = selected_task
        self._table = clone_rows(load_task_input(selected_task.task_id))
        initial_grades = grade_table(selected_task, self._table)
        self._preview_limit = 5
        self._last_action = None
        self._last_action_error = None
        self._state = TabularCleaningState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=selected_task.task_id,
            current_table=clone_rows(self._table),
            current_columns=self._current_columns(),
            current_score=initial_grades["score"],
            best_score_so_far=initial_grades["score"],
            submitted=False,
            max_steps=selected_task.max_steps,
            task_rules=selected_task.task_rules,
        )
        return self._build_observation(
            reward=None,
            done=False,
            metadata={"grader_breakdown": initial_grades, "reset": True},
        )

    def step(
        self,
        action: TabularCleaningAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> TabularCleaningObservation:
        del timeout_s
        if self._state.submitted or self._state.step_count >= self._task.max_steps:
            return self._build_observation(
                reward=0.0,
                done=True,
                error="Episode already finished. Call reset() to start a new task.",
                metadata={"final_score": self._state.current_score, "reason": "episode_complete"},
            )

        self._state.step_count += 1
        self._last_action = action.model_dump(exclude_none=True)
        self._last_action_error = None

        previous_score = self._state.current_score
        previous_best = self._state.best_score_so_far
        before_table = clone_rows(self._table)
        info: Dict[str, Any] = {"action_type": action.action_type.value}
        error: str | None = None

        try:
            if action.action_type == ActionType.INSPECT_TABLE:
                self._preview_limit = max(1, min(action.preview_rows, 10))
                info["inspection"] = {
                    "column_count": len(self._current_columns()),
                    "row_count": len(self._table),
                }
            elif action.action_type == ActionType.INSPECT_COLUMN:
                column = self._require_column(action)
                values = [row.get(column) for row in self._table]
                unique_values = []
                seen = set()
                for value in values:
                    marker = stable_json(value)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    unique_values.append(value)
                    if len(unique_values) == 5:
                        break
                info["column_profile"] = {
                    "column": column,
                    "missing_count": sum(1 for value in values if is_missing(value)),
                    "unique_sample": unique_values,
                }
            elif action.action_type == ActionType.RENAME_COLUMN:
                self._rename_column(action)
            elif action.action_type == ActionType.STRIP_WHITESPACE:
                self._apply_to_columns(self._resolve_target_columns(action, string_only=True), self._strip_value)
            elif action.action_type == ActionType.NORMALIZE_CASE:
                if action.case_mode is None:
                    raise ValueError("normalize_case requires case_mode")
                self._apply_to_columns(
                    self._resolve_target_columns(action, string_only=True),
                    lambda value: self._case_value(value, action.case_mode),
                )
            elif action.action_type == ActionType.REPLACE_VALUES:
                if not action.replacements:
                    raise ValueError("replace_values requires replacements")
                self._replace_values(action)
            elif action.action_type == ActionType.STANDARDIZE_DATE:
                columns = [action.column] if action.column else list(self._task.date_columns.keys())
                for column in columns:
                    self._require_existing_column(column)
                    self._standardize_date_column(column)
            elif action.action_type == ActionType.FILL_MISSING:
                if action.fill_value is None:
                    raise ValueError("fill_missing requires fill_value")
                columns = [action.column] if action.column else list(self._task.fill_defaults.keys())
                for column in columns:
                    self._fill_missing(column, action.fill_value)
            elif action.action_type == ActionType.CAST_DTYPE:
                if action.dtype is None:
                    raise ValueError("cast_dtype requires dtype")
                self._cast_column(self._require_column(action), action.dtype)
            elif action.action_type == ActionType.DROP_DUPLICATES:
                self._drop_duplicates()
            elif action.action_type == ActionType.SORT_ROWS:
                self._sort_rows(action.sort_by or list(self._task.primary_key), action.ascending)
            elif action.action_type == ActionType.SUBMIT:
                self._state.submitted = True
                info["submitted"] = True
            else:  # pragma: no cover - protected by enum validation
                raise ValueError(f"Unsupported action type: {action.action_type}")
        except ValueError as exc:
            error = str(exc)
            self._table = before_table

        grades = grade_table(self._task, self._table)
        self._state.current_table = clone_rows(self._table)
        self._state.current_columns = self._current_columns()
        self._state.current_score = grades["score"]
        reward = 0.0

        if error is None:
            if self._table == before_table and action.action_type not in {
                ActionType.INSPECT_TABLE,
                ActionType.INSPECT_COLUMN,
                ActionType.SUBMIT,
            }:
                info["penalty_type"] = "no_op"
            elif grades["score"] < previous_score:
                info["penalty_type"] = "destructive"
            reward = max(0.0, round(grades["score"] - previous_best, 6))
            self._state.best_score_so_far = max(previous_best, grades["score"])
        else:
            reward = 0.0
            grades = grade_table(self._task, self._table)
            self._state.current_score = grades["score"]
            self._state.current_table = clone_rows(self._table)
            info["penalty_type"] = "invalid"

        if self._state.step_count >= self._task.max_steps:
            self._state.submitted = True
            info["termination_reason"] = "max_steps"

        done = self._state.submitted
        info["grader_breakdown"] = grades
        info["score_before_action"] = previous_score
        info["score_after_action"] = self._state.current_score

        return self._build_observation(
            reward=reward,
            done=done,
            error=error,
            metadata=info,
        )

    @property
    def state(self) -> TabularCleaningState:
        return self._state

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "tabular_cleaning_env",
            "description": (
                "A deterministic OpenEnv benchmark where agents clean real-world style "
                "messy tables using a structured action space."
            ),
            "version": "0.1.0",
            "author": "OpenEnv Hackathon Submission",
        }

    def _build_observation(
        self,
        reward: float | None,
        done: bool,
        error: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TabularCleaningObservation:
        self._last_action_error = error
        return TabularCleaningObservation(
            task_id=self._task.task_id,
            task_description=self._task.description,
            table_columns=self._current_columns(),
            table_rows_preview=clone_rows(self._table[: self._preview_limit]),
            row_count=len(self._table),
            issues_summary=self._issues_summary(),
            last_action=deepcopy(self._last_action),
            last_action_error=error,
            steps_taken=self._state.step_count,
            max_steps=self._task.max_steps,
            current_score_estimate=round(self._state.current_score, 6),
            available_actions=[action.value for action in ActionType],
            reward=reward,
            done=done,
            metadata=metadata or {},
        )

    def _current_columns(self) -> List[str]:
        if not self._table:
            return list(self._task.expected_columns)
        return list(self._table[0].keys())

    def _require_column(self, action: TabularCleaningAction) -> str:
        if not action.column:
            raise ValueError(f"{action.action_type.value} requires column")
        self._require_existing_column(action.column)
        return action.column

    def _require_existing_column(self, column: str | None) -> None:
        if not column or column not in self._current_columns():
            raise ValueError(f"Unknown column: {column}")

    def _rename_column(self, action: TabularCleaningAction) -> None:
        source = self._require_column(action)
        if not action.new_name:
            raise ValueError("rename_column requires new_name")
        if action.new_name in self._current_columns():
            raise ValueError(f"Column already exists: {action.new_name}")
        renamed_rows: List[Dict[str, Any]] = []
        for row in self._table:
            renamed = {}
            for key, value in row.items():
                renamed[action.new_name if key == source else key] = value
            renamed_rows.append(renamed)
        self._table = renamed_rows

    def _resolve_target_columns(self, action: TabularCleaningAction, string_only: bool = False) -> List[str]:
        columns = [action.column] if action.column else self._current_columns()
        resolved = []
        for column in columns:
            self._require_existing_column(column)
            if string_only and not any(isinstance(row.get(column), str) for row in self._table):
                continue
            resolved.append(column)
        return resolved

    def _apply_to_columns(self, columns: Sequence[str], fn: Any) -> None:
        for row in self._table:
            for column in columns:
                row[column] = fn(row.get(column))

    @staticmethod
    def _strip_value(value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @staticmethod
    def _case_value(value: Any, case_mode: CaseMode) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if case_mode == CaseMode.LOWER:
            return stripped.lower()
        if case_mode == CaseMode.UPPER:
            return stripped.upper()
        if case_mode == CaseMode.TITLE:
            return stripped.title()
        return stripped

    def _replace_values(self, action: TabularCleaningAction) -> None:
        column = self._require_column(action)
        normalized_map = {key.strip().lower(): value for key, value in action.replacements.items()}
        for row in self._table:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            key = value.strip().lower()
            if key in normalized_map:
                row[column] = normalized_map[key]

    def _standardize_date_column(self, column: str) -> None:
        include_time = self._task.date_columns.get(column, False)
        for row in self._table:
            converted = format_datetime_for_task(row.get(column), include_time)
            if converted is not None:
                row[column] = converted

    def _fill_missing(self, column: str | None, fill_value: str) -> None:
        self._require_existing_column(column)
        assert column is not None
        for row in self._table:
            if is_missing(row.get(column)):
                row[column] = fill_value

    def _cast_column(self, column: str, dtype: str) -> None:
        for row in self._table:
            value = row.get(column)
            if is_missing(value):
                row[column] = None
                continue
            row[column] = coerce_dtype(value, dtype)

    def _drop_duplicates(self) -> None:
        rule = self._task.duplicate_rule
        if rule is None:
            raise ValueError("This task does not support drop_duplicates")
        grouped: Dict[tuple[str, ...], List[Dict[str, Any]]] = {}
        for row in self._table:
            grouped.setdefault(canonical_key(row, rule.key_fields), []).append(row)
        deduplicated = [self._choose_best_row(rule, rows) for rows in grouped.values()]
        self._table = sorted(
            deduplicated,
            key=lambda row: (canonical_key(row, self._task.primary_key), stable_json(row)),
        )

    def _choose_best_row(
        self, rule: DuplicateRule, rows: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if len(rows) == 1:
            return deepcopy(rows[0])
        ranked = sorted(
            rows,
            key=lambda row: (
                count_non_missing(row, rule.completeness_fields),
                format_datetime_for_task(row.get(rule.latest_timestamp_field), True) if rule.latest_timestamp_field else "",
                stable_json(row),
            ),
            reverse=True,
        )
        return deepcopy(ranked[0])

    def _sort_rows(self, sort_by: Sequence[str], ascending: bool) -> None:
        for column in sort_by:
            self._require_existing_column(column)
        self._table = sorted(
            self._table,
            key=lambda row: tuple("" if row.get(column) is None else str(row.get(column)) for column in sort_by),
            reverse=not ascending,
        )

    def _issues_summary(self) -> List[str]:
        issues: List[str] = []
        current_columns = self._current_columns()
        if current_columns != list(self._task.expected_columns):
            issues.append("Schema does not match the expected cleaned table columns.")

        whitespace_columns = [
            column
            for column in current_columns
            if any(isinstance(row.get(column), str) and row.get(column) != row.get(column).strip() for row in self._table)
        ]
        if whitespace_columns:
            issues.append(f"Whitespace cleanup is still needed in: {', '.join(whitespace_columns[:3])}.")

        missing_count = sum(
            1
            for row in self._table
            for column in self._task.required_columns
            if column in row and is_missing(row.get(column))
        )
        if missing_count:
            issues.append(f"{missing_count} required cells are still missing values.")

        if self._task.duplicate_rule is not None:
            duplicate_counts = Counter(
                canonical_key(row, self._task.duplicate_rule.key_fields) for row in self._table
            )
            duplicates = sum(count - 1 for count in duplicate_counts.values() if count > 1)
            if duplicates:
                issues.append("Duplicate business keys still need to be resolved.")

        date_issues = []
        for column, include_time in self._task.date_columns.items():
            if column not in current_columns:
                continue
            for row in self._table:
                value = row.get(column)
                if is_missing(value):
                    continue
                canonical = format_datetime_for_task(value, include_time)
                if canonical is None or str(value) != canonical:
                    date_issues.append(column)
                    break
        if date_issues:
            issues.append(f"Date normalization is still needed in: {', '.join(date_issues)}.")

        for column, mapping in self._task.normalization_hints.items():
            if column not in current_columns:
                continue
            if any(
                isinstance(row.get(column), str)
                and row.get(column).strip().lower() in mapping
                and row.get(column) != mapping[row.get(column).strip().lower()]
                for row in self._table
            ):
                issues.append(f"Value normalization is still needed in column '{column}'.")
                break

        if not issues:
            issues.append("Table looks clean. Submit when you are confident.")
        return issues[:6]
