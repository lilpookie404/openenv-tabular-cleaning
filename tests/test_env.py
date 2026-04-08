from __future__ import annotations

from server.app import app
from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.tasks import FALLBACK_POLICIES, TASKS


def test_app_import_smoke() -> None:
    assert app is not None


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
        assert observation.max_steps == task.max_steps
        assert observation.steps_taken == 0
        assert env.state.task_id == task_id
        assert env.state.max_steps == task.max_steps


def test_fallback_policy_reaches_perfect_score() -> None:
    env = TabularCleaningEnvironment()
    for task_id, policy in FALLBACK_POLICIES.items():
        env.reset(task_id=task_id)
        result = None
        for action in policy:
            result = env.step(action)
        assert result is not None
        assert result.done is True
        assert result.current_score_estimate == 1.0


def test_invalid_action_has_zero_reward() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = env.step(
        TabularCleaningAction(action_type=ActionType.RENAME_COLUMN, column="missing", new_name="name")
    )
    assert result.reward == 0.0
    assert result.last_action_error is not None


def test_max_steps_terminates_episode() -> None:
    env = TabularCleaningEnvironment()
    env.reset(task_id="easy_contacts_cleanup")
    result = None
    for _ in range(TASKS["easy_contacts_cleanup"].max_steps):
        result = env.step(TabularCleaningAction(action_type=ActionType.INSPECT_TABLE))
    assert result is not None
    assert result.done is True
