"""Baseline inference runner for the tabular cleaning environment."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        OpenAI,
        PermissionDeniedError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - local fallback path
    OpenAI = Any  # type: ignore[misc,assignment]

    class _MissingOpenAIError(Exception):
        pass

    APIConnectionError = _MissingOpenAIError
    APIStatusError = _MissingOpenAIError
    APITimeoutError = _MissingOpenAIError
    AuthenticationError = _MissingOpenAIError
    BadRequestError = _MissingOpenAIError
    InternalServerError = _MissingOpenAIError
    NotFoundError = _MissingOpenAIError
    PermissionDeniedError = _MissingOpenAIError
    RateLimitError = _MissingOpenAIError

from pydantic import ValidationError

from server.environment import TabularCleaningEnvironment
from tabular_cleaning_env.models import ActionType, TabularCleaningAction
from tabular_cleaning_env.utils import stable_json

ENV_NAME = "tabular_cleaning_env"
TASK_ORDER = [
    "easy_contacts_cleanup",
    "medium_orders_cleanup",
    "hard_appointments_cleanup",
]


@dataclass
class LLMRuntimeState:
    enabled: bool = True
    disabled_reason: Optional[str] = None
    last_fallback_reason: Optional[str] = None
    fallback_count: int = 0


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _error_text(message: Optional[str]) -> str:
    if message is None:
        return "null"
    sanitized = str(message).replace("\r", " ").replace("\n", " ").strip()
    return " ".join(sanitized.split())


def _action_text(action: TabularCleaningAction) -> str:
    return stable_json(action.model_dump(exclude_none=True))


def _action_signature(action: TabularCleaningAction) -> str:
    return stable_json(action.model_dump(exclude_none=True))


def build_openai_client() -> Tuple[OpenAI, str]:
    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct").strip()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None or not hf_token.strip():
        raise ValueError("HF_TOKEN environment variable is required")
    if OpenAI is Any:
        raise RuntimeError("openai package is required to run inference.py")
    return OpenAI(
        base_url=base_url,
        api_key=hf_token.strip(),
        timeout=10.0,
        max_retries=0,
    ), model_name


def llm_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
) -> TabularCleaningAction:
    prompt = (
        "You are operating a human-in-the-loop tabular cleanup workbench.\n"
        "Return exactly one JSON object and no extra text.\n"
        "Use this schema:\n"
        '{"action_type":"...", "column":"...", "new_name":"...", "case_mode":"...", '
        '"replacements":{"old":"new"}, "fill_value":"...", "dtype":"...", '
        '"sort_by":["..."], "ascending":true, "preview_rows":5, '
        '"change_id":"...", "destination":"..."}\n'
        "Available workflow actions include profile_table, approve_changes, run_validations, "
        "export_cleaned_table, and publish_table.\n"
        "Omit fields you do not use. Prefer structured, safe actions.\n"
        f"Task: {task_id}\n"
        f"Observation: {json.dumps(observation, ensure_ascii=True)}"
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_completion_tokens=120,
        messages=[
            {"role": "system", "content": "Return only valid JSON for the next action."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return TabularCleaningAction.model_validate(json.loads(content))


def fallback_action_from_observation(
    observation: Dict[str, Any],
    executed_actions: Set[str],
) -> TabularCleaningAction:
    rules = observation.get("task_rules", {})
    columns = set(observation.get("table_columns", []))
    issues = observation.get("issues_summary", [])
    issues_text = " ".join(issues).lower()
    change_set = observation.get("change_set_summary", {})
    risky_changes = observation.get("risky_changes") or observation.get("proposed_changes_summary") or []
    validation_status = observation.get("validation_status", "not_run")
    profiled = bool(change_set.get("profiled"))
    has_export_artifact = bool(change_set.get("has_export_artifact"))
    published = bool(change_set.get("published"))

    cleaning_issue_prefixes = (
        "Schema does not match",
        "Whitespace cleanup",
        "Value normalization",
        "Date normalization",
        "Required fields",
        "Duplicate business keys",
    )

    def has_cleaning_issues() -> bool:
        return any(
            str(issue).startswith(prefix)
            for issue in issues
            for prefix in cleaning_issue_prefixes
        )

    def choose(action: TabularCleaningAction, condition: bool = True) -> Optional[TabularCleaningAction]:
        if not condition:
            return None
        signature = _action_signature(action)
        if signature in executed_actions:
            return None
        return action

    action = choose(
        TabularCleaningAction(action_type=ActionType.PROFILE_TABLE),
        condition=not profiled,
    )
    if action is not None:
        return action

    if risky_changes:
        latest_change = risky_changes[-1]
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.APPROVE_CHANGES,
                change_id=latest_change.get("change_id"),
            )
        )
        if action is not None:
            return action

    for source, target in rules.get("rename_map", {}).items():
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.RENAME_COLUMN,
                column=source,
                new_name=target,
            ),
            condition=(
                source in columns
                and target not in columns
                and "schema does not match the expected cleaned table columns" in issues_text
            ),
        )
        if action is not None:
            return action

    action = choose(
        TabularCleaningAction(action_type=ActionType.STRIP_WHITESPACE),
        condition="whitespace cleanup is still needed" in issues_text,
    )
    if action is not None:
        return action

    for column, case_mode in rules.get("case_columns", {}).items():
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.NORMALIZE_CASE,
                column=column,
                case_mode=case_mode,
            ),
            condition=column in columns,
        )
        if action is not None:
            return action

    for column, replacements in rules.get("normalization_hints", {}).items():
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.REPLACE_VALUES,
                column=column,
                replacements=replacements,
            ),
            condition=column in columns,
        )
        if action is not None:
            return action

    action = choose(
        TabularCleaningAction(action_type=ActionType.STANDARDIZE_DATE),
        condition=bool(rules.get("date_columns")) and any(column in columns for column in rules.get("date_columns", {})),
    )
    if action is not None:
        return action

    fill_defaults = rules.get("fill_defaults", {})
    shared_fill_values = {value for value in fill_defaults.values() if value is not None}
    if len(shared_fill_values) == 1 and len(fill_defaults) > 1:
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.FILL_MISSING,
                fill_value=next(iter(shared_fill_values)),
            ),
            condition=bool(fill_defaults) and "required fields are still missing" in issues_text,
        )
        if action is not None:
            return action

    for column, fill_value in rules.get("fill_defaults", {}).items():
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.FILL_MISSING,
                column=column,
                fill_value=fill_value,
            ),
            condition=column in columns and "required fields are still missing" in issues_text,
        )
        if action is not None:
            return action

    for column, dtype in rules.get("cast_columns", {}).items():
        action = choose(
            TabularCleaningAction(
                action_type=ActionType.CAST_DTYPE,
                column=column,
                dtype=dtype,
            ),
            condition=column in columns,
        )
        if action is not None:
            return action

    action = choose(
        TabularCleaningAction(action_type=ActionType.DROP_DUPLICATES),
        condition=rules.get("duplicate_rule") is not None and "duplicate business keys still need to be resolved" in issues_text,
    )
    if action is not None:
        return action

    if not has_cleaning_issues():
        action = choose(
            TabularCleaningAction(action_type=ActionType.RUN_VALIDATIONS),
            condition=validation_status != "passed",
        )
        if action is not None:
            return action

        action = choose(
            TabularCleaningAction(
                action_type=ActionType.EXPORT_CLEANED_TABLE,
                destination=rules.get("default_export_destination"),
            ),
            condition=validation_status == "passed" and not has_export_artifact,
        )
        if action is not None:
            return action

        action = choose(
            TabularCleaningAction(action_type=ActionType.PUBLISH_TABLE),
            condition=validation_status == "passed" and has_export_artifact and not published,
        )
        if action is not None:
            return action

    return TabularCleaningAction(action_type=ActionType.SUBMIT)


def classify_llm_exception(exc: Exception) -> Tuple[str, bool]:
    if isinstance(exc, (JSONDecodeError, ValidationError)):
        return "llm_output_invalid", False
    fatal_types = (
        TimeoutError,
        ConnectionError,
        OSError,
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        PermissionDeniedError,
        RateLimitError,
        InternalServerError,
        APIStatusError,
        BadRequestError,
        NotFoundError,
    )
    if isinstance(exc, fatal_types):
        return "llm_transport_error", True
    return "llm_runtime_error", True


def run_task(
    task_id: str,
    client: Optional[OpenAI],
    model_name: str,
    env_factory: Callable[[], TabularCleaningEnvironment] = TabularCleaningEnvironment,
    llm_state: Optional[LLMRuntimeState] = None,
) -> Dict[str, Any]:
    env = env_factory()
    rewards: List[float] = []
    score = 0.0
    step_count = 0
    success = False
    last_error: Optional[str] = None
    executed_actions: Set[str] = set()
    runtime_state = llm_state or LLMRuntimeState(enabled=client is not None)
    if client is None and runtime_state.enabled:
        runtime_state.enabled = False
        runtime_state.disabled_reason = runtime_state.disabled_reason or "llm_client_unavailable"
    print(f"[START] task={task_id} env={ENV_NAME} model={model_name}", flush=True)
    try:
        reset_obs = env.reset(task_id=task_id)
        observation = reset_obs.model_dump(exclude_none=True)

        while True:
            if client is not None and runtime_state.enabled:
                try:
                    action = llm_action(client, model_name, task_id, observation)
                except Exception as exc:
                    reason, disable_future = classify_llm_exception(exc)
                    runtime_state.last_fallback_reason = reason
                    runtime_state.fallback_count += 1
                    if disable_future:
                        runtime_state.enabled = False
                        runtime_state.disabled_reason = reason
                    action = fallback_action_from_observation(observation, executed_actions)
            else:
                action = fallback_action_from_observation(observation, executed_actions)

            result = env.step(action)
            executed_actions.add(_action_signature(action))
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
            f"[END] success={_bool_text(success)} steps={step_count} rewards={rewards_text}",
            flush=True,
        )
    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "score": score,
        "rewards": rewards,
        "error": last_error,
        "fallback_reason": runtime_state.last_fallback_reason,
        "llm_disabled": not runtime_state.enabled,
        "llm_disabled_reason": runtime_state.disabled_reason,
        "llm_fallback_count": runtime_state.fallback_count,
    }


def main() -> List[Dict[str, Any]]:
    client, model_name = build_openai_client()
    llm_state = LLMRuntimeState(enabled=True)
    return [run_task(task_id, client, model_name, llm_state=llm_state) for task_id in TASK_ORDER]


if __name__ == "__main__":
    main()
