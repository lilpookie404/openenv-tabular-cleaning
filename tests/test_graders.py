from __future__ import annotations

from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.graders import grade_table
from tabular_cleaning_env.tasks import FALLBACK_POLICIES, TASKS, load_task_expected, load_task_input


def test_raw_partial_and_gold_scores_are_ordered() -> None:
    env = TabularCleaningEnvironment()
    for task_id, task in TASKS.items():
        raw_score = grade_table(task, load_task_input(task_id))["score"]
        gold_score = grade_table(task, load_task_expected(task_id))["score"]
        env.reset(task_id=task_id)
        partial_result = None
        for action in FALLBACK_POLICIES[task_id][:2]:
            partial_result = env.step(action)
        assert partial_result is not None
        partial_score = partial_result.current_score_estimate
        assert raw_score < partial_score < gold_score
        assert gold_score == 1.0


def test_rewards_stay_bounded() -> None:
    env = TabularCleaningEnvironment()
    for task_id, policy in FALLBACK_POLICIES.items():
        env.reset(task_id=task_id)
        total_reward = 0.0
        for action in policy:
            result = env.step(action)
            reward = float(result.reward or 0.0)
            assert 0.0 <= reward <= 1.0
            total_reward += reward
        assert 0.0 <= total_reward <= 1.0
