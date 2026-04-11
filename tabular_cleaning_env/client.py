"""Typed client for the tabular cleaning environment."""

from __future__ import annotations

from typing import Any, Dict

from .models import TabularCleaningAction, TabularCleaningObservation, TabularCleaningState
from .openenv_compat import EnvClient, StepResult


class TabularCleaningEnv(
    EnvClient[TabularCleaningAction, TabularCleaningObservation, TabularCleaningState]
):
    """Client for a running tabular cleaning environment server."""

    def _step_payload(self, action: TabularCleaningAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[TabularCleaningObservation]:
        obs_data = payload.get("observation", {})
        observation = TabularCleaningObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            source_system=obs_data.get("source_system", ""),
            task_description=obs_data.get("task_description", ""),
            task_rules=obs_data.get("task_rules", {}),
            table_columns=obs_data.get("table_columns", []),
            table_rows_preview=obs_data.get("table_rows_preview", []),
            row_count=obs_data.get("row_count", 0),
            issues_summary=obs_data.get("issues_summary", []),
            change_set_summary=obs_data.get("change_set_summary", {}),
            proposed_changes_summary=obs_data.get("proposed_changes_summary", []),
            risky_changes=obs_data.get("risky_changes", []),
            validation_status=obs_data.get("validation_status", "not_run"),
            validation_checks=obs_data.get("validation_checks", []),
            audit_log_preview=obs_data.get("audit_log_preview", []),
            export_ready=obs_data.get("export_ready", False),
            last_action=obs_data.get("last_action"),
            last_action_error=obs_data.get("last_action_error"),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 0),
            current_score_estimate=obs_data.get("current_score_estimate", 0.0),
            available_actions=obs_data.get("available_actions", []),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TabularCleaningState:
        return TabularCleaningState.model_validate(payload)
