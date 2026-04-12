from __future__ import annotations

from typing import Dict, List

import inference
from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.graders import SCORE_MAX, SCORE_MIN, grade_task
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.tasks import TASKS, load_task_expected, load_task_input
from tabular_cleaning_env.utils import stable_json

OPEN_INTERVAL_MIN = SCORE_MIN
OPEN_INTERVAL_MAX = SCORE_MAX
PUBLIC_TASK_SCORE = 0.5
REWARD_MIN = 0.01


def _collect_rule_based_actions(task_id: str) -> List[TabularCleaningAction]:
    env = TabularCleaningEnvironment()
    observation = env.reset(task_id=task_id)
    obs_payload = observation.model_dump(exclude_none=True)
    executed = set()
    actions: List[TabularCleaningAction] = []
    while True:
        action = inference.fallback_action_from_observation(obs_payload, executed)
        actions.append(action)
        executed.add(stable_json(action.model_dump(exclude_none=True)))
        result = env.step(action)
        obs_payload = result.model_dump(exclude_none=True)
        if result.done:
            break
    env.close()
    return actions


def _run_actions(task_id: str, actions: List[TabularCleaningAction]) -> Dict[str, float]:
    env = TabularCleaningEnvironment()
    env.reset(task_id=task_id)
    result = None
    for action in actions:
        result = env.step(action)
        if result.done:
            break
    env.close()
    assert result is not None
    return {
        "score": result.current_score_estimate,
        "quality": grade_task(TASKS[task_id], env.state.current_table),
        "reward": float(result.reward) if result.reward is not None else REWARD_MIN,
        "done": bool(result.done),
        "published": bool(env.state.published),
    }


def test_raw_partial_and_gold_scores_are_ordered() -> None:
    env = TabularCleaningEnvironment()
    for task_id, task in TASKS.items():
        raw_table_score = grade_task(task, load_task_input(task_id))
        gold_score = grade_task(task, load_task_expected(task_id))
        assert OPEN_INTERVAL_MIN <= raw_table_score <= OPEN_INTERVAL_MAX
        assert OPEN_INTERVAL_MIN <= gold_score <= OPEN_INTERVAL_MAX
        observation = env.reset(task_id=task_id)
        assert observation.current_score_estimate == PUBLIC_TASK_SCORE
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        partial_result = observation
        partial_quality = raw_table_score
        for _ in range(5):
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            partial_result = env.step(action)
            partial_quality = grade_task(task, env.state.current_table)
            obs_payload = partial_result.model_dump(exclude_none=True)
            if partial_quality > raw_table_score:
                break
        assert raw_table_score < partial_quality <= OPEN_INTERVAL_MAX
        assert partial_result.current_score_estimate == PUBLIC_TASK_SCORE
        assert gold_score == OPEN_INTERVAL_MAX


def test_task_scores_are_rounding_safe_at_two_decimals() -> None:
    for task_id, task in TASKS.items():
        raw_score = grade_task(task, load_task_input(task_id))
        gold_score = grade_task(task, load_task_expected(task_id))
        missing_score = grade_task(task, [{"unknown_column": "value"}])
        for score in (raw_score, gold_score, missing_score):
            formatted = format(score, ".2f")
            assert formatted not in {"0.00", "1.00"}, f"{task_id} rounded out of range: {score}"


def test_task_scores_are_rounding_safe_at_one_decimal() -> None:
    for task_id, task in TASKS.items():
        raw_score = grade_task(task, load_task_input(task_id))
        gold_score = grade_task(task, load_task_expected(task_id))
        missing_score = grade_task(task, [{"unknown_column": "value"}])
        for score in (raw_score, gold_score, missing_score):
            formatted = format(score, ".1f")
            assert formatted not in {"0.0", "1.0"}, f"{task_id} rounded out of range at 1dp: {score}"


def test_missing_grading_columns_produce_low_but_in_range_score() -> None:
    for task in TASKS.values():
        score = grade_task(task, [{"unknown_column": "value"}])
        assert OPEN_INTERVAL_MIN <= score < 0.5
        assert format(score, ".2f") != "0.00"
        assert format(score, ".1f") != "0.0"


def test_terminal_publish_reward_is_positive_and_in_range() -> None:
    for task_id in TASKS:
        result = inference.run_task(task_id, client=None, model_name="deterministic-fallback")
        assert REWARD_MIN <= result["rewards"][-1] <= OPEN_INTERVAL_MAX
        assert result["score"] == PUBLIC_TASK_SCORE


def test_rewards_stay_bounded() -> None:
    env = TabularCleaningEnvironment()
    for task_id in TASKS:
        observation = env.reset(task_id=task_id)
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        total_reward = 0.0
        while True:
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            result = env.step(action)
            reward = float(result.reward) if result.reward is not None else REWARD_MIN
            assert REWARD_MIN <= reward <= OPEN_INTERVAL_MAX
            total_reward += reward
            obs_payload = result.model_dump(exclude_none=True)
            if result.done:
                break
        assert total_reward >= REWARD_MIN


def test_skipping_any_required_rule_based_action_lowers_final_score() -> None:
    for task_id in TASKS:
        actions = _collect_rule_based_actions(task_id)
        assert actions[-1].action_type == ActionType.PUBLISH_TABLE
        for index, action in enumerate(actions[:-1]):
            if action.action_type == ActionType.SORT_ROWS:
                continue
            rerun_actions = [candidate for offset, candidate in enumerate(actions) if offset != index]
            result = _run_actions(task_id, rerun_actions)
            assert (
                result["quality"] < OPEN_INTERVAL_MAX or (not result["done"]) or (not result["published"])
            ), f"{task_id} unexpectedly succeeded without {action.action_type.value}"


def test_date_and_fill_actions_improve_score() -> None:
    env = TabularCleaningEnvironment()
    for task_id in TASKS:
        observation = env.reset(task_id=task_id)
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        current_score = grade_task(TASKS[task_id], env.state.current_table)
        date_score_deltas: List[float] = []
        fill_score_deltas: List[float] = []
        while True:
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            result = env.step(action)
            score_delta = round(grade_task(TASKS[task_id], env.state.current_table) - current_score, 6)
            if action.action_type == ActionType.STANDARDIZE_DATE:
                date_score_deltas.append(score_delta)
            if action.action_type == ActionType.FILL_MISSING:
                fill_score_deltas.append(score_delta)
            current_score = grade_task(TASKS[task_id], env.state.current_table)
            obs_payload = result.model_dump(exclude_none=True)
            if result.done:
                break
        assert any(delta > 0 for delta in date_score_deltas)
        if TASKS[task_id].fill_defaults:
            assert fill_score_deltas
