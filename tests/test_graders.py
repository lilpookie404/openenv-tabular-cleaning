from __future__ import annotations

from typing import Dict, List

import inference
from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.graders import grade_table
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.tasks import TASKS, load_task_expected, load_task_input
from tabular_cleaning_env.utils import stable_json


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
        "reward": float(result.reward or 0),
        "done": float(bool(result.done)),
        "published": float(bool(env.state.published)),
    }


def test_raw_partial_and_gold_scores_are_ordered() -> None:
    env = TabularCleaningEnvironment()
    for task_id, task in TASKS.items():
        raw_table_score = grade_table(task, load_task_input(task_id))["score"]
        gold_score = grade_table(task, load_task_expected(task_id))["score"]
        assert 0 < raw_table_score < 1
        assert 0 < gold_score < 1
        observation = env.reset(task_id=task_id)
        raw_episode_score = observation.current_score_estimate
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        partial_result = None
        for _ in range(2):
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            partial_result = env.step(action)
            obs_payload = partial_result.model_dump(exclude_none=True)
        assert partial_result is not None
        partial_score = partial_result.current_score_estimate
        assert 0 < raw_episode_score < gold_score
        assert raw_episode_score < partial_score < 1
        assert 0.999 < gold_score < 1


def test_terminal_publish_reward_is_positive_and_in_range() -> None:
    for task_id in TASKS:
        result = inference.run_task(task_id, client=None, model_name="deterministic-fallback")
        assert 0 < result["rewards"][-1] < 1


def test_rewards_stay_bounded() -> None:
    env = TabularCleaningEnvironment()
    for task_id in TASKS:
        observation = env.reset(task_id=task_id)
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        total_reward = 0
        while True:
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            result = env.step(action)
            reward = float(result.reward or 0)
            assert 0 <= reward <= 1
            total_reward += reward
            obs_payload = result.model_dump(exclude_none=True)
            if result.done:
                break
        assert 0 <= total_reward <= 1


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
                result["score"] < 0.999 or result["done"] == 0 or result["published"] == 0
            ), f"{task_id} unexpectedly succeeded without {action.action_type.value}"


def test_date_and_fill_actions_improve_score() -> None:
    env = TabularCleaningEnvironment()
    for task_id in TASKS:
        observation = env.reset(task_id=task_id)
        obs_payload = observation.model_dump(exclude_none=True)
        executed = set()
        date_rewards: List[float] = []
        fill_rewards: List[float] = []
        while True:
            action = inference.fallback_action_from_observation(obs_payload, executed)
            executed.add(stable_json(action.model_dump(exclude_none=True)))
            result = env.step(action)
            reward = float(result.reward or 0)
            if action.action_type == ActionType.STANDARDIZE_DATE:
                date_rewards.append(reward)
            if action.action_type == ActionType.FILL_MISSING:
                fill_rewards.append(reward)
            obs_payload = result.model_dump(exclude_none=True)
            if result.done:
                break
        assert any(reward > 0 for reward in date_rewards)
        if TASKS[task_id].fill_defaults:
            assert fill_rewards and all(reward > 0 for reward in fill_rewards)
