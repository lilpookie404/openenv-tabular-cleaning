from __future__ import annotations

from fastapi.testclient import TestClient

import inference
from server.app import app
from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.graders import SCORE_MAX, SCORE_MIN, grade_task
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.tasks import TASKS

OPEN_INTERVAL_MIN = SCORE_MIN
OPEN_INTERVAL_MAX = SCORE_MAX
REWARD_MIN = 0.01


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
    assert blocked.reward == REWARD_MIN
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
        assert result["score"] == OPEN_INTERVAL_MAX


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


def test_invalid_action_has_minimum_visible_reward() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = env.step(
        TabularCleaningAction(action_type=ActionType.RENAME_COLUMN, column="missing", new_name="name")
    )
    assert result.reward == REWARD_MIN
    assert result.last_action_error is not None


def test_max_steps_terminates_episode() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = None
    for _ in range(TASKS["easy_contacts_cleanup"].max_steps):
        result = env.step(TabularCleaningAction(action_type=ActionType.INSPECT_TABLE))
    assert result is not None
    assert result.done is True


def test_schema_reward_is_non_null_number_with_default() -> None:
    client = TestClient(app)
    schema = client.get("/schema").json()
    reward_schema = schema["observation"]["properties"]["reward"]
    assert reward_schema["type"] == "number"
    assert reward_schema["default"] == REWARD_MIN
    assert "anyOf" not in reward_schema


def test_public_score_and_reward_surfaces_stay_inside_open_interval() -> None:
    client = TestClient(app)
    seen_score_keys: set[str] = set()
    allowed_score_keys = {"current_score_estimate", "current_score", "best_score_so_far", "score"}

    def audit(node: object, *, path: str = "root") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}.{key}"
                if "score" in key.lower() or "reward" in key.lower():
                    if "score" in key.lower():
                        seen_score_keys.add(key)
                    if isinstance(value, list):
                        for index, item in enumerate(value):
                            item_path = f"{next_path}[{index}]"
                            assert isinstance(item, (int, float)), (
                                f"{item_path} should be numeric, got {type(item).__name__}"
                            )
                            if "score" in key.lower():
                                assert OPEN_INTERVAL_MIN <= float(item) < 1, (
                                    f"{item_path} escaped open interval: {item!r}"
                                )
                                assert format(float(item), ".2f") not in {"0.00", "1.00"}, (
                                    f"{item_path} rounds to boundary: {item!r}"
                                )
                                assert format(float(item), ".1f") not in {"0.0", "1.0"}, (
                                    f"{item_path} rounds to 1dp boundary: {item!r}"
                                )
                            else:
                                assert REWARD_MIN <= float(item) < 1, (
                                    f"{item_path} escaped reward bounds: {item!r}"
                                )
                    else:
                        assert isinstance(value, (int, float)), (
                            f"{next_path} should be numeric, got {type(value).__name__}"
                        )
                        if "score" in key.lower():
                            assert OPEN_INTERVAL_MIN <= float(value) < 1, (
                                f"{next_path} escaped open interval: {value!r}"
                            )
                            assert format(float(value), ".2f") not in {"0.00", "1.00"}, (
                                f"{next_path} rounds to boundary: {value!r}"
                            )
                            assert format(float(value), ".1f") not in {"0.0", "1.0"}, (
                                f"{next_path} rounds to 1dp boundary: {value!r}"
                            )
                        else:
                            assert REWARD_MIN <= float(value) < 1, (
                                f"{next_path} escaped reward bounds: {value!r}"
                            )
                audit(value, path=next_path)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                audit(value, path=f"{path}[{index}]")

    reset = client.post("/reset", json={"task_id": "easy_contacts_cleanup"}).json()
    step = client.post("/step", json={"action": {"action_type": "inspect_table"}}).json()

    env = TabularCleaningEnvironment()
    observation = env.reset(task_id="easy_contacts_cleanup")
    payload = observation.model_dump(exclude_none=True)
    executed = set()
    solved_observation = observation
    while True:
        action = inference.fallback_action_from_observation(payload, executed)
        executed.add(inference._action_signature(action))
        solved_observation = env.step(action)
        payload = solved_observation.model_dump(exclude_none=True)
        if solved_observation.done:
            break
    state_payload = env.state.model_dump()
    inference_result = inference.run_task("easy_contacts_cleanup", client=None, model_name="deterministic-fallback")
    env.close()

    audit(reset)
    audit(step)
    audit(solved_observation.model_dump(exclude_none=True))
    audit(state_payload)
    audit(inference_result)
    assert seen_score_keys <= allowed_score_keys


def test_workflow_actions_do_not_inflate_task_score() -> None:
    env = TabularCleaningEnvironment()
    observation = env.reset(task_id="easy_contacts_cleanup")
    payload = observation.model_dump(exclude_none=True)
    executed = set()
    previous_score = observation.current_score_estimate

    while True:
        action = inference.fallback_action_from_observation(payload, executed)
        executed.add(inference._action_signature(action))
        result = env.step(action)
        if action.action_type in {
            ActionType.RUN_VALIDATIONS,
            ActionType.EXPORT_CLEANED_TABLE,
            ActionType.PUBLISH_TABLE,
        }:
            assert result.current_score_estimate == previous_score
        previous_score = result.current_score_estimate
        payload = result.model_dump(exclude_none=True)
        if result.done:
            break
    assert grade_task(TASKS["easy_contacts_cleanup"], env.state.current_table) == env.state.current_score
    env.close()
