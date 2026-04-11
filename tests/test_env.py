from __future__ import annotations

from fastapi.testclient import TestClient

import inference
from server.app import app
from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.tasks import TASKS


def test_app_import_smoke() -> None:
    assert app is not None


def test_root_page_has_core_links() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "/docs" in response.text
    assert "/metadata" in response.text
    assert "/schema" in response.text
    assert "/health" in response.text


def test_action_model_rejects_invalid_enum_and_extra_fields() -> None:
    try:
        TabularCleaningAction.model_validate({"action_type": "not_real"})
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid enum should fail")

    try:
        TabularCleaningAction.model_validate({"action_type": "inspect_table", "extra": True})
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("extra fields should fail")


def test_reset_and_state_for_each_task() -> None:
    env = TabularCleaningEnvironment()
    for task_id, task in TASKS.items():
        observation = env.reset(task_id=task_id)
        assert observation.task_id == task_id
        assert observation.difficulty == task.difficulty
        assert observation.source_system == task.source_system
        assert observation.task_rules["expected_columns"] == list(task.expected_columns)
        assert observation.max_steps == task.max_steps
        assert observation.steps_taken == 0
        assert observation.validation_status == "not_run"
        assert observation.change_set_summary["profiled"] is False
        assert env.state.task_id == task_id
        assert env.state.max_steps == task.max_steps
        assert env.state.source_system == task.source_system
        assert env.state.transformation_log == []


def test_risky_change_requires_approval_before_more_mutations() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")

    profiled = env.step(TabularCleaningAction(action_type=ActionType.PROFILE_TABLE))
    assert profiled.change_set_summary["profiled"] is True

    renamed = env.step(
        TabularCleaningAction(action_type=ActionType.RENAME_COLUMN, column="full_name", new_name="name")
    )
    assert renamed.risky_changes
    change_id = renamed.risky_changes[-1]["change_id"]

    blocked = env.step(TabularCleaningAction(action_type=ActionType.STRIP_WHITESPACE))
    assert blocked.reward == 0
    assert blocked.last_action_error is not None
    assert "Approve or reject" in blocked.last_action_error

    approved = env.step(TabularCleaningAction(action_type=ActionType.APPROVE_CHANGES, change_id=change_id))
    assert approved.risky_changes == []


def test_rule_based_run_produces_validated_export_and_publish_state() -> None:
    env = TabularCleaningEnvironment()
    observation = env.reset(task_id="easy_contacts_cleanup")
    payload = observation.model_dump(exclude_none=True)
    executed = set()
    result = observation
    while True:
        action = inference.fallback_action_from_observation(payload, executed)
        executed.add(inference._action_signature(action))
        result = env.step(action)
        payload = result.model_dump(exclude_none=True)
        if result.done:
            break
    assert result.done is True
    assert env.state.validation_status == "passed"
    assert env.state.export_artifacts
    assert env.state.published is True


def test_rule_based_fallback_reaches_near_perfect_open_interval_score() -> None:
    for task_id in TASKS:
        result = inference.run_task(task_id, client=None, model_name="deterministic-fallback")
        assert result["success"] is True
        assert 0.999 < result["score"] < 1


def test_rule_based_fallback_does_not_emit_sort_rows() -> None:
    for task_id in TASKS:
        env = TabularCleaningEnvironment()
        observation = env.reset(task_id=task_id)
        payload = observation.model_dump(exclude_none=True)
        executed = set()
        action_types = []
        while True:
            action = inference.fallback_action_from_observation(payload, executed)
            executed.add(inference._action_signature(action))
            action_types.append(action.action_type)
            result = env.step(action)
            payload = result.model_dump(exclude_none=True)
            if result.done:
                break
        env.close()
        assert ActionType.SORT_ROWS not in action_types


def test_invalid_action_has_zero_reward() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = env.step(
        TabularCleaningAction(action_type=ActionType.RENAME_COLUMN, column="missing", new_name="name")
    )
    assert result.reward == 0
    assert result.last_action_error is not None


def test_max_steps_terminates_episode() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = None
    for _ in range(TASKS["easy_contacts_cleanup"].max_steps):
        result = env.step(TabularCleaningAction(action_type=ActionType.INSPECT_TABLE))
    assert result is not None
    assert result.done is True
