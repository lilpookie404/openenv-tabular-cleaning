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
    looks_like_email,
    stable_json,
    summarize_rows,
)


class TabularCleaningEnvironment(Environment):
    """OpenEnv-compatible environment for a governed tabular cleanup workflow."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    OPEN_INTERVAL_MIN = 1e-5
    OPEN_INTERVAL_MAX = 0.9999
    REWARD_MIN = 0.01
    TABLE_SCORE_WEIGHT = 0.9
    VALIDATION_BONUS = 0.02
    EXPORT_BONUS = 0.02
    PUBLISH_BONUS = 0.0599

    CLEANING_ACTION_TYPES = {
        ActionType.RENAME_COLUMN,
        ActionType.STRIP_WHITESPACE,
        ActionType.NORMALIZE_CASE,
        ActionType.REPLACE_VALUES,
        ActionType.STANDARDIZE_DATE,
        ActionType.FILL_MISSING,
        ActionType.CAST_DTYPE,
        ActionType.DROP_DUPLICATES,
        ActionType.SORT_ROWS,
    }
    RISKY_ACTION_TYPES = {
        ActionType.RENAME_COLUMN,
        ActionType.FILL_MISSING,
        ActionType.CAST_DTYPE,
        ActionType.DROP_DUPLICATES,
    }

    def __init__(self, default_task_id: str = "easy_contacts_cleanup"):
        super().__init__()
        self._default_task_id = default_task_id
        self._task = get_task(default_task_id)
        self._table: List[Dict[str, Any]] = []
        self._state = TabularCleaningState()
        self._preview_limit = 5
        self._last_action: Dict[str, Any] | None = None
        self._last_action_error: str | None = None
        self._change_counter = 0
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
        initial_score = self._compose_episode_score(
            initial_grades["score"],
            validation_status="not_run",
            has_export_artifact=False,
            published=False,
        )
        self._preview_limit = 5
        self._last_action = None
        self._last_action_error = None
        self._change_counter = 0
        self._state = TabularCleaningState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=selected_task.task_id,
            source_system=selected_task.source_system,
            current_table=clone_rows(self._table),
            current_columns=self._current_columns(),
            current_score=initial_score,
            best_score_so_far=initial_score,
            submitted=False,
            published=False,
            profiled=False,
            export_ready=False,
            validation_status="not_run",
            max_steps=selected_task.max_steps,
            task_rules=selected_task.task_rules,
            transformation_log=[],
            proposed_changes=[],
            approved_changes=[],
            rejected_changes=[],
            validation_results=[],
            export_artifacts={},
        )
        return self._build_observation(
            reward=self.REWARD_MIN,
            done=False,
            metadata={
                "reset": True,
                "workflow": self._workflow_metadata(),
            },
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
                reward=self.REWARD_MIN,
                done=True,
                error="Episode already finished. Call reset() to start a new task.",
                metadata={"final_quality_index": self._state.current_score, "reason": "episode_complete"},
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
            if self._state.proposed_changes and action.action_type in self.CLEANING_ACTION_TYPES:
                raise ValueError("Approve or reject the pending risky change before applying more data changes.")

            if action.action_type == ActionType.INSPECT_TABLE:
                self._preview_limit = max(1, min(action.preview_rows, 10))
                info["inspection"] = self._inspection_profile()
            elif action.action_type == ActionType.INSPECT_COLUMN:
                info["column_profile"] = self._inspect_column(action)
            elif action.action_type == ActionType.PROFILE_TABLE:
                self._state.profiled = True
                profile = self._profile_table()
                info["profile"] = profile
                self._append_workflow_log(
                    action_type=ActionType.PROFILE_TABLE.value,
                    status="completed",
                    reason="profiled_source_export",
                    details=profile,
                )
            elif action.action_type == ActionType.VIEW_CHANGE_SET:
                info["change_set"] = {
                    "summary": self._change_set_summary(),
                    "pending_changes": [self._public_change(change) for change in self._state.proposed_changes],
                    "approved_changes": self._state.approved_changes[-5:],
                }
                self._append_workflow_log(
                    action_type=ActionType.VIEW_CHANGE_SET.value,
                    status="completed",
                    reason="viewed_change_set",
                    details={"pending_review_count": len(self._state.proposed_changes)},
                )
            elif action.action_type == ActionType.RUN_VALIDATIONS:
                if not self._state.profiled:
                    raise ValueError("Profile the table before running validations.")
                if self._state.proposed_changes:
                    raise ValueError("Approve or reject pending risky changes before validation.")
                self._run_validations()
                info["validation_checks"] = clone_rows(self._state.validation_results)
                self._append_workflow_log(
                    action_type=ActionType.RUN_VALIDATIONS.value,
                    status=self._state.validation_status,
                    reason="validation_run",
                    details={"export_ready": self._state.export_ready},
                )
            elif action.action_type == ActionType.APPROVE_CHANGES:
                approved = self._approve_changes(action.change_id)
                info["approved_change_ids"] = [entry["change_id"] for entry in approved]
            elif action.action_type == ActionType.REJECT_CHANGE:
                rejected = self._reject_change(action.change_id)
                info["rejected_change_id"] = rejected["change_id"]
            elif action.action_type == ActionType.EXPORT_CLEANED_TABLE:
                artifact = self._export_cleaned_table(action.destination)
                info["export_destination"] = artifact["destination"]
            elif action.action_type in {ActionType.PUBLISH_TABLE, ActionType.SUBMIT}:
                self._publish_table()
                info["published"] = True
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
            else:  # pragma: no cover - protected by enum validation
                raise ValueError(f"Unsupported action type: {action.action_type}")
        except ValueError as exc:
            error = str(exc)
            self._table = before_table

        if error is None and action.action_type in self.CLEANING_ACTION_TYPES:
            if self._table != before_table:
                self._invalidate_delivery_state()
                change_record = self._record_transformation(action, before_table)
                if change_record is not None:
                    info["change_record"] = change_record
            else:
                info["penalty_type"] = "no_op"

        grades = grade_table(self._task, self._table)
        self._state.current_table = clone_rows(self._table)
        self._state.current_columns = self._current_columns()
        episode_score = self._compose_episode_score(grades["score"])
        self._state.current_score = episode_score
        reward = self.REWARD_MIN

        if error is None:
            if episode_score < previous_score:
                info["penalty_type"] = "destructive"
            reward_delta = round(episode_score - previous_best, 6)
            reward = self._emit_reward(reward_delta)
            self._state.best_score_so_far = max(previous_best, episode_score)
        else:
            reward = self.REWARD_MIN
            grades = grade_table(self._task, self._table)
            self._state.current_score = self._compose_episode_score(grades["score"])
            self._state.current_table = clone_rows(self._table)
            info["penalty_type"] = "invalid"

        if self._state.step_count >= self._task.max_steps:
            self._state.submitted = True
            info["termination_reason"] = "max_steps"

        done = self._state.submitted
        info["workflow"] = self._workflow_metadata()

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
                "A deterministic OpenEnv workbench for operational tabular cleanup with "
                "audit logs, human review, validation gates, and export/publish workflow."
            ),
            "version": "0.2.0",
            "author": "OpenEnv Hackathon Submission",
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        error: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TabularCleaningObservation:
        self._last_action_error = error
        return TabularCleaningObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            source_system=self._task.source_system,
            task_description=self._task.description,
            task_rules=self._task.task_rules,
            table_columns=self._current_columns(),
            table_rows_preview=clone_rows(self._table[: self._preview_limit]),
            row_count=len(self._table),
            issues_summary=self._issues_summary(),
            change_set_summary=self._change_set_summary(),
            proposed_changes_summary=[self._public_change(change) for change in self._state.proposed_changes],
            risky_changes=[self._public_change(change) for change in self._state.proposed_changes],
            validation_status=self._state.validation_status,
            validation_checks=clone_rows(self._state.validation_results),
            audit_log_preview=clone_rows(self._state.transformation_log[-5:]),
            export_ready=self._state.export_ready,
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

    def _workflow_score_bonus(self) -> float:
        bonus = 0.0
        if self._state.validation_status == "passed":
            bonus += self.VALIDATION_BONUS
        if bool(self._state.export_artifacts):
            bonus += self.EXPORT_BONUS
        if self._state.published:
            bonus += self.PUBLISH_BONUS
        return bonus

    def _compose_episode_score(
        self,
        table_score: float,
        *,
        validation_status: str | None = None,
        has_export_artifact: bool | None = None,
        published: bool | None = None,
    ) -> float:
        effective_validation = self._state.validation_status if validation_status is None else validation_status
        effective_export = bool(self._state.export_artifacts) if has_export_artifact is None else has_export_artifact
        effective_published = self._state.published if published is None else published
        bonus = 0.0
        if effective_validation == "passed":
            bonus += self.VALIDATION_BONUS
        if effective_export:
            bonus += self.EXPORT_BONUS
        if effective_published:
            bonus += self.PUBLISH_BONUS
        composed = (self.TABLE_SCORE_WEIGHT * table_score) + bonus
        return round(self._emit_open_interval(composed), 6)

    def _emit_open_interval(self, value: float) -> float:
        return min(max(float(value), self.OPEN_INTERVAL_MIN), self.OPEN_INTERVAL_MAX)

    def _emit_reward(self, value: float) -> float:
        return min(max(float(value), self.REWARD_MIN), self.OPEN_INTERVAL_MAX)

    def _inspection_profile(self) -> Dict[str, Any]:
        return {
            "column_count": len(self._current_columns()),
            "row_count": len(self._table),
            "missing_by_column": {
                column: sum(1 for row in self._table if is_missing(row.get(column)))
                for column in self._current_columns()
            },
            "duplicate_key_count": self._duplicate_key_count(),
            "date_columns": dict(self._task.date_columns),
            "normalization_columns": list(self._task.normalization_hints.keys()),
            "rule_pack_name": self._task.rule_pack_name,
            "source_system": self._task.source_system,
        }

    def _inspect_column(self, action: TabularCleaningAction) -> Dict[str, Any]:
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
        return {
            "column": column,
            "missing_count": sum(1 for value in values if is_missing(value)),
            "distinct_count": len({stable_json(value) for value in values}),
            "unique_sample": unique_values,
            "normalization_hints": self._task.normalization_hints.get(column, {}),
            "recommended_fill": self._task.fill_defaults.get(column),
            "recommended_case": self._task.case_columns.get(column).value
            if column in self._task.case_columns
            else None,
            "expected_dtype": self._task.cast_columns.get(column),
            "canonical_datetime": column in self._task.date_columns,
        }

    def _profile_table(self) -> Dict[str, Any]:
        suggested = self._suggested_changes()
        return {
            "source_system": self._task.source_system,
            "rule_pack_name": self._task.rule_pack_name,
            "row_count": len(self._table),
            "column_count": len(self._current_columns()),
            "issues_summary": self._issues_summary()[:4],
            "suggested_safe_actions": suggested["safe"],
            "suggested_risky_actions": suggested["risky"],
            "validation_rules": clone_rows(
                [
                    {"check_id": check_id, "description": description}
                    for check_id, description in self._task.validation_rules.items()
                ]
            ),
        }

    def _change_set_summary(self) -> Dict[str, Any]:
        suggested = self._suggested_changes()
        return {
            "profiled": self._state.profiled,
            "pending_review_count": len(self._state.proposed_changes),
            "approved_count": len(self._state.approved_changes),
            "rejected_count": len(self._state.rejected_changes),
            "has_export_artifact": bool(self._state.export_artifacts),
            "published": self._state.published,
            "next_stage": self._next_stage(),
            "suggested_safe_actions": suggested["safe"],
            "suggested_risky_actions": suggested["risky"],
        }

    def _workflow_metadata(self) -> Dict[str, Any]:
        return {
            "next_stage": self._next_stage(),
            "validation_status": self._state.validation_status,
            "pending_review_count": len(self._state.proposed_changes),
            "export_ready": self._state.export_ready,
            "published": self._state.published,
        }

    def _next_stage(self) -> str:
        if self._state.published:
            return "completed"
        if not self._state.profiled:
            return "profile"
        if self._state.proposed_changes:
            return "review"
        if self._has_cleaning_issues():
            return "clean"
        if self._state.validation_status != "passed":
            return "validate"
        if not self._state.export_artifacts:
            return "export"
        return "publish"

    def _invalidate_delivery_state(self) -> None:
        self._state.validation_status = "not_run"
        self._state.validation_results = []
        self._state.export_artifacts = {}
        self._state.export_ready = False
        self._state.published = False

    def _record_transformation(
        self,
        action: TabularCleaningAction,
        before_table: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if self._table == before_table:
            return None
        profile = self._risk_profile(action)
        summary_before = self._table_summary(before_table)
        summary_after = self._table_summary(self._table)
        affected = self._change_footprint(before_table, self._table)
        base_entry = {
            "change_id": self._next_change_id(),
            "step": self._state.step_count,
            "action_type": action.action_type.value,
            "risk_category": profile["risk_category"],
            "confidence": profile["confidence"],
            "reason": profile["reason"],
            "affected_columns": affected["columns"],
            "affected_row_count": affected["row_count"],
            "before_summary": summary_before,
            "after_summary": summary_after,
            "review_required": profile["review_required"],
            "source": "agent",
        }
        if profile["review_required"]:
            pending_entry = {
                **base_entry,
                "status": "pending_review",
                "snapshot_before": clone_rows(before_table),
                "snapshot_after": clone_rows(self._table),
            }
            self._state.proposed_changes.append(pending_entry)
            self._state.transformation_log.append(self._public_change(pending_entry))
            return self._public_change(pending_entry)

        auto_entry = {**base_entry, "status": "auto_applied"}
        self._state.transformation_log.append(auto_entry)
        return auto_entry

    def _approve_changes(self, change_id: str | None) -> List[Dict[str, Any]]:
        if not self._state.proposed_changes:
            raise ValueError("No risky changes are waiting for approval.")

        if change_id is None:
            pending = list(self._state.proposed_changes)
        else:
            pending = [change for change in self._state.proposed_changes if change["change_id"] == change_id]
            if not pending:
                raise ValueError(f"Unknown pending change: {change_id}")

        approved_entries: List[Dict[str, Any]] = []
        for change in pending:
            self._state.proposed_changes.remove(change)
            approved = {
                **self._public_change(change),
                "status": "approved",
                "reviewed_at_step": self._state.step_count,
            }
            self._state.approved_changes.append(approved)
            approved_entries.append(approved)
        self._state.transformation_log.append(
            {
                "change_id": f"audit-{self._state.step_count:03d}",
                "step": self._state.step_count,
                "action_type": ActionType.APPROVE_CHANGES.value,
                "status": "completed",
                "reviewed_change_ids": [entry["change_id"] for entry in approved_entries],
                "reason": "human_approved_risky_changes",
            }
        )
        return approved_entries

    def _reject_change(self, change_id: str | None) -> Dict[str, Any]:
        if not self._state.proposed_changes:
            raise ValueError("No risky changes are waiting for rejection.")

        target = self._state.proposed_changes[-1]
        if change_id is not None and target["change_id"] != change_id:
            raise ValueError("Only the latest pending risky change can be rejected deterministically.")

        self._table = clone_rows(target["snapshot_before"])
        self._state.current_table = clone_rows(self._table)
        self._state.current_columns = self._current_columns()
        self._state.proposed_changes.pop()
        self._invalidate_delivery_state()
        rejected = {
            **self._public_change(target),
            "status": "rejected",
            "reviewed_at_step": self._state.step_count,
        }
        self._state.rejected_changes.append(rejected)
        self._state.transformation_log.append(
            {
                "change_id": f"audit-{self._state.step_count:03d}",
                "step": self._state.step_count,
                "action_type": ActionType.REJECT_CHANGE.value,
                "status": "completed",
                "reviewed_change_ids": [rejected["change_id"]],
                "reason": "human_rejected_risky_change",
            }
        )
        return rejected

    def _run_validations(self) -> None:
        results: List[Dict[str, Any]] = []
        for check_id, description in self._task.validation_rules.items():
            passed, detail = self._evaluate_validation_rule(check_id)
            results.append(
                {
                    "check_id": check_id,
                    "description": description,
                    "status": "passed" if passed else "failed",
                    "severity": "error",
                    "detail": detail,
                }
            )
        self._state.validation_results = results
        self._state.validation_status = "passed" if all(item["status"] == "passed" for item in results) else "failed"
        self._state.export_ready = self._state.validation_status == "passed" and not self._state.proposed_changes

    def _evaluate_validation_rule(self, check_id: str) -> tuple[bool, str]:
        current_columns = self._current_columns()
        if check_id == "required_fields_present":
            missing = [
                (index, column)
                for index, row in enumerate(self._table)
                for column in self._task.required_columns
                if column in row and is_missing(row.get(column))
            ]
            if missing:
                return False, f"{len(missing)} required cells are still missing."
            return True, "All required fields are populated."
        if check_id == "schema_matches":
            passed = current_columns == list(self._task.expected_columns)
            return passed, "Schema matches expected export." if passed else "Schema does not match expected export."
        if check_id == "duplicates_resolved":
            duplicates = self._duplicate_key_count()
            return duplicates == 0, "Business keys are unique." if duplicates == 0 else f"{duplicates} duplicate rows remain."
        if check_id in {"dates_canonical", "timestamps_canonical"}:
            failing = []
            for column, include_time in self._task.date_columns.items():
                for row in self._table:
                    value = row.get(column)
                    if is_missing(value):
                        continue
                    canonical = format_datetime_for_task(value, include_time)
                    if canonical is None or str(value) != canonical:
                        failing.append(column)
                        break
            if failing:
                return False, f"Canonical date formatting still missing in: {', '.join(sorted(set(failing)))}."
            return True, "Date and timestamp fields are canonical."
        if check_id == "emails_valid":
            invalid = [row.get("email") for row in self._table if not looks_like_email(row.get("email"))]
            return not invalid, "Emails are valid." if not invalid else f"{len(invalid)} invalid emails remain."
        if check_id == "amounts_numeric_non_negative":
            invalid = [
                row.get("amount")
                for row in self._table
                if not isinstance(row.get("amount"), (int, float)) or float(row.get("amount")) < 0
            ]
            return not invalid, "Amounts are numeric and non-negative." if not invalid else "Some amounts are invalid."
        if check_id == "technician_assignments_valid":
            invalid = [row.get("technician") for row in self._table if is_missing(row.get("technician"))]
            return (
                not invalid,
                "Technician assignments are populated or intentionally reviewed."
                if not invalid
                else "Some technician fields are still blank.",
            )
        return True, "Validation rule not implemented explicitly; treated as passed."

    def _export_cleaned_table(self, destination: str | None) -> Dict[str, Any]:
        if self._state.proposed_changes:
            raise ValueError("Approve or reject pending risky changes before export.")
        if self._state.validation_status != "passed":
            raise ValueError("Run validations successfully before export.")
        export_destination = destination or self._task.default_export_destination
        artifact = {
            "artifact_id": f"{self._task.task_id}-artifact-{self._state.step_count:03d}",
            "destination": export_destination,
            "cleaned_table": clone_rows(self._table),
            "data_quality_report": {
                "task_id": self._task.task_id,
                "source_system": self._task.source_system,
                "quality_index": round(self._state.current_score, 6),
                "validation_status": self._state.validation_status,
                "validation_checks": clone_rows(self._state.validation_results),
            },
            "transformation_audit_log": clone_rows(self._state.transformation_log),
        }
        self._state.export_artifacts = artifact
        self._state.export_ready = True
        self._state.transformation_log.append(
            {
                "change_id": f"audit-{self._state.step_count:03d}",
                "step": self._state.step_count,
                "action_type": ActionType.EXPORT_CLEANED_TABLE.value,
                "status": "completed",
                "destination": export_destination,
                "reason": "exported_clean_table",
            }
        )
        return artifact

    def _publish_table(self) -> None:
        if self._state.proposed_changes:
            raise ValueError("Approve or reject pending risky changes before publishing.")
        if not self._state.export_artifacts:
            raise ValueError("Export the cleaned table before publishing.")
        self._state.published = True
        self._state.submitted = True
        self._state.transformation_log.append(
            {
                "change_id": f"audit-{self._state.step_count:03d}",
                "step": self._state.step_count,
                "action_type": ActionType.PUBLISH_TABLE.value,
                "status": "completed",
                "destination": self._state.export_artifacts.get("destination"),
                "reason": "published_clean_table",
            }
        )

    def _public_change(self, change: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: deepcopy(value)
            for key, value in change.items()
            if key not in {"snapshot_before", "snapshot_after"}
        }

    def _append_workflow_log(
        self,
        action_type: str,
        status: str,
        reason: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "change_id": f"audit-{self._state.step_count:03d}-{len(self._state.transformation_log):02d}",
            "step": self._state.step_count,
            "action_type": action_type,
            "status": status,
            "reason": reason,
        }
        if details:
            entry["details"] = deepcopy(details)
        self._state.transformation_log.append(entry)

    def _risk_profile(self, action: TabularCleaningAction) -> Dict[str, Any]:
        if action.action_type == ActionType.RENAME_COLUMN:
            return {
                "review_required": True,
                "risk_category": "high",
                "confidence": 0.74,
                "reason": "Renaming a column changes downstream schema mappings.",
            }
        if action.action_type == ActionType.FILL_MISSING:
            return {
                "review_required": True,
                "risk_category": "medium",
                "confidence": 0.72,
                "reason": "Imputing missing values changes source completeness.",
            }
        if action.action_type == ActionType.CAST_DTYPE:
            return {
                "review_required": True,
                "risk_category": "medium",
                "confidence": 0.78,
                "reason": "Type casting can coerce ambiguous raw values.",
            }
        if action.action_type == ActionType.DROP_DUPLICATES:
            return {
                "review_required": True,
                "risk_category": "high",
                "confidence": 0.66,
                "reason": "Dropping duplicates removes rows from the source export.",
            }
        return {
            "review_required": False,
            "risk_category": "low",
            "confidence": 0.97,
            "reason": "Normalization-only change with low downstream risk.",
        }

    def _table_summary(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        duplicate_fields = self._task.duplicate_rule.key_fields if self._task.duplicate_rule else ()
        return summarize_rows(rows, self._task.required_columns, duplicate_fields)

    def _change_footprint(
        self,
        before_rows: Sequence[Dict[str, Any]],
        after_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        max_len = max(len(before_rows), len(after_rows))
        changed_rows = 0
        changed_columns = set()
        for index in range(max_len):
            before = before_rows[index] if index < len(before_rows) else {}
            after = after_rows[index] if index < len(after_rows) else {}
            if stable_json(before) == stable_json(after):
                continue
            changed_rows += 1
            changed_columns.update(set(before.keys()) | set(after.keys()))
            for column in set(before.keys()) | set(after.keys()):
                if before.get(column) != after.get(column):
                    changed_columns.add(column)
        return {"row_count": changed_rows, "columns": sorted(changed_columns)}

    def _next_change_id(self) -> str:
        self._change_counter += 1
        return f"chg-{self._change_counter:03d}"

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
                format_datetime_for_task(row.get(rule.latest_timestamp_field), True)
                if rule.latest_timestamp_field
                else "",
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

    def _suggested_changes(self) -> Dict[str, List[str]]:
        current_columns = self._current_columns()
        safe: List[str] = []
        risky: List[str] = []

        if current_columns != list(self._task.expected_columns) and any(
            source in current_columns and target not in current_columns for source, target in self._task.rename_map.items()
        ):
            risky.append(ActionType.RENAME_COLUMN.value)
        if any(
            isinstance(row.get(column), str) and row.get(column) != row.get(column).strip()
            for row in self._table
            for column in current_columns
        ):
            safe.append(ActionType.STRIP_WHITESPACE.value)
        if self._task.case_columns:
            for column in self._task.case_columns:
                if column in current_columns:
                    safe.append(f"{ActionType.NORMALIZE_CASE.value}:{column}")
        if self._task.normalization_hints:
            for column in self._task.normalization_hints:
                if column in current_columns:
                    safe.append(f"{ActionType.REPLACE_VALUES.value}:{column}")
        if self._task.date_columns:
            safe.append(ActionType.STANDARDIZE_DATE.value)
        if self._task.fill_defaults:
            risky.extend(f"{ActionType.FILL_MISSING.value}:{column}" for column in self._task.fill_defaults if column in current_columns)
        if self._task.cast_columns:
            risky.extend(f"{ActionType.CAST_DTYPE.value}:{column}" for column in self._task.cast_columns if column in current_columns)
        if self._task.duplicate_rule is not None and self._duplicate_key_count():
            risky.append(ActionType.DROP_DUPLICATES.value)
        return {
            "safe": list(dict.fromkeys(safe)),
            "risky": list(dict.fromkeys(risky)),
        }

    def _has_cleaning_issues(self) -> bool:
        return any(issue.startswith(prefix) for issue in self._issues_summary() for prefix in [
            "Schema does not match",
            "Whitespace cleanup",
            "Value normalization",
            "Date normalization",
            "Required fields",
            "Duplicate business keys",
        ])

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
            issues.append(f"Required fields are still missing in {missing_count} cells.")

        duplicates = self._duplicate_key_count()
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
            issues.append(f"Date normalization is still needed in: {', '.join(sorted(set(date_issues)))}.")

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

        if self._state.proposed_changes:
            issues.append("Pending risky changes must be approved or rejected before validation.")
        elif not issues:
            if not self._state.profiled:
                issues.append("Profile the table to inspect the source export and suggested change set.")
            elif self._state.validation_status == "not_run":
                issues.append("Run validations before export.")
            elif self._state.validation_status == "failed":
                issues.append("Validation failures block export and publish.")
            elif not self._state.export_artifacts:
                issues.append("Validation passed. Export the cleaned table.")
            elif not self._state.published:
                issues.append("Export complete. Publish the cleaned table.")
            else:
                issues.append("Table has been published with validation and audit artifacts.")

        return issues[:8]

    def _duplicate_key_count(self) -> int:
        if self._task.duplicate_rule is None:
            return 0
        duplicate_counts = Counter(
            canonical_key(row, self._task.duplicate_rule.key_fields) for row in self._table
        )
        return sum(count - 1 for count in duplicate_counts.values() if count > 1)
