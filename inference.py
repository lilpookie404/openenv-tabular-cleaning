"""Baseline inference runner for the tabular cleaning environment."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - local fallback path
    OpenAI = Any  # type: ignore[misc,assignment]

from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.models import TabularCleaningAction
from tabular_cleaning_env.tasks import FALLBACK_POLICIES, TASKS
from tabular_cleaning_env.utils import stable_json

ENV_NAME = "tabular_cleaning_env"
TASK_ORDER = [
    "easy_contacts_cleanup",
    "medium_orders_cleanup",
    "hard_appointments_cleanup",
]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _error_text(message: Optional[str]) -> str:
    return "null" if message is None else json.dumps(message, ensure_ascii=True)


def _action_text(action: TabularCleaningAction) -> str:
    return stable_json(action.model_dump(exclude_none=True))


def build_openai_client() -> Tuple[Optional[OpenAI], str]:
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct").strip()
    hf_token = os.getenv("HF_TOKEN")
    api_key = hf_token.strip() if hf_token else None
    if not (base_url and model_name and api_key) or OpenAI is Any:
        return None, model_name
    return OpenAI(base_url=base_url, api_key=api_key), model_name


def fallback_action(task_id: str, policy_index: int) -> TabularCleaningAction:
    policy = FALLBACK_POLICIES[task_id]
    bounded_index = min(policy_index, len(policy) - 1)
    return policy[bounded_index]


def llm_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
) -> TabularCleaningAction:
    prompt = (
        "You are cleaning tabular data in a deterministic benchmark.\n"
        "Return exactly one JSON object and no extra text.\n"
        "Use this schema:\n"
        '{"action_type":"...", "column":"...", "new_name":"...", "case_mode":"...", '
        '"replacements":{"old":"new"}, "fill_value":"...", "dtype":"...", '
        '"sort_by":["..."], "ascending":true, "preview_rows":5}\n'
        "Omit fields you do not use. Prefer structured, safe actions.\n"
        f"Task: {task_id}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=True)}"
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only valid JSON for the next action."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return TabularCleaningAction.model_validate(json.loads(content))


def run_task(
    task_id: str,
    client: Optional[OpenAI],
    model_name: str,
    env_factory: Callable[[], TabularCleaningEnvironment] = TabularCleaningEnvironment,
) -> Dict[str, Any]:
    env = env_factory()
    rewards: List[float] = []
    score = 0.0
    step_count = 0
    success = False
    last_error: Optional[str] = None
    print(f"[START] task={task_id} env={ENV_NAME} model={model_name}", flush=True)
    try:
        reset_obs = env.reset(task_id=task_id)
        observation = reset_obs.model_dump(exclude_none=True)
        policy_index = 0

        while True:
            try:
                action = (
                    llm_action(client, model_name, task_id, observation)
                    if client is not None
                    else fallback_action(task_id, policy_index)
                )
                if client is None:
                    policy_index += 1
            except Exception:
                action = fallback_action(task_id, policy_index)
                policy_index += 1

            result = env.step(action)
            step_count += 1
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            last_error = result.last_action_error
            score = result.current_score_estimate
            done = bool(result.done)
            print(
                f"[STEP] step={step_count} action={_action_text(action)} reward={reward:.2f} "
                f"done={_bool_text(done)} error={_error_text(last_error)}",
                flush=True,
            )
            observation = result.model_dump(exclude_none=True)
            if done:
                success = score >= 0.999
                break
    except Exception as exc:
        last_error = str(exc)
    finally:
        env.close()
        rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={_bool_text(success)} steps={step_count} score={score:.3f} rewards={rewards_text}",
            flush=True,
        )
    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "score": score,
        "rewards": rewards,
        "error": last_error,
    }


def main() -> List[Dict[str, Any]]:
    client, model_name = build_openai_client()
    return [run_task(task_id, client, model_name) for task_id in TASK_ORDER]


if __name__ == "__main__":
    main()
