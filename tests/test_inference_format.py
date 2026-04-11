from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import pytest

import inference


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(message=_FakeMessage(content=content))]


class _FakeCompletions:
    def __init__(self, responses: List[object]):
        self._responses = list(responses)
        self.calls = 0
        self.create_kwargs: List[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        self.create_kwargs.append(dict(kwargs))
        response = self._responses[min(self.calls - 1, len(self._responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(str(response))


class _FakeClient:
    def __init__(self, responses: List[object]):
        self.completions_api = _FakeCompletions(responses)
        self.chat = type("Chat", (), {"completions": self.completions_api})()


def test_inference_prints_required_sections(capsys) -> None:
    inference.run_task("easy_contacts_cleanup", client=None, model_name="deterministic-fallback")
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith(
        "[START] task=easy_contacts_cleanup env=tabular_cleaning_env model=deterministic-fallback"
    )
    assert any(line.startswith("[STEP]") for line in output)
    assert output[-1].startswith("[END] success=true steps=")
    assert "score=" not in output[-1]


def test_inference_always_emits_end_on_exception(capsys) -> None:
    class ExplodingEnv:
        def __init__(self) -> None:
            pass

        def reset(self, **_: Any) -> Any:
            raise RuntimeError("boom")

        def close(self) -> None:
            return None

    inference.run_task(
        "easy_contacts_cleanup",
        client=None,
        model_name="deterministic-fallback",
        env_factory=ExplodingEnv,
    )
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith("[START]")
    assert output[-1].startswith("[END] success=false steps=0 rewards=")
    assert "score=" not in output[-1]


def test_error_text_is_plain_single_line() -> None:
    assert inference._error_text(None) == "null"
    assert inference._error_text("bad value") == "bad value"
    assert inference._error_text("bad\nvalue\tagain") == "bad value again"


def test_build_openai_client_requires_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    with pytest.raises(ValueError):
        inference.build_openai_client()


def test_build_openai_client_sets_timeout_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    init_kwargs: dict[str, object] = {}

    class DummyOpenAI:
        def __init__(self, **kwargs: object):
            init_kwargs.update(kwargs)

    monkeypatch.setattr(inference, "OpenAI", DummyOpenAI)
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    client, model_name = inference.build_openai_client()
    assert isinstance(client, DummyOpenAI)
    assert model_name == "meta-llama/Llama-3.3-70B-Instruct"
    assert init_kwargs["timeout"] == 10.0
    assert init_kwargs["max_retries"] == 0


def test_transport_failure_disables_llm_for_remaining_run() -> None:
    client = _FakeClient([TimeoutError("endpoint down")])
    state = inference.LLMRuntimeState(enabled=True)
    result = inference.run_task(
        "easy_contacts_cleanup",
        client=client,
        model_name="test-model",
        llm_state=state,
    )
    assert result["success"] is True
    assert client.completions_api.calls == 1
    assert state.enabled is False
    assert result["llm_disabled"] is True
    assert result["llm_disabled_reason"] == "llm_transport_error"
    assert result["fallback_reason"] == "llm_transport_error"


def test_invalid_json_falls_back_without_disabling_llm() -> None:
    client = _FakeClient(["{not-json}"])
    state = inference.LLMRuntimeState(enabled=True)
    result = inference.run_task(
        "easy_contacts_cleanup",
        client=client,
        model_name="test-model",
        llm_state=state,
    )
    assert result["success"] is True
    assert client.completions_api.calls == result["steps"]
    assert state.enabled is True
    assert result["llm_disabled"] is False
    assert result["fallback_reason"] == "llm_output_invalid"
    assert client.completions_api.create_kwargs[0]["max_completion_tokens"] == 120


def test_fallback_planner_uses_task_rules() -> None:
    action = inference.fallback_action_from_observation(
        {
            "change_set_summary": {"profiled": False, "has_export_artifact": False, "published": False},
            "task_rules": {
                "rename_map": {"raw_name": "name"},
                "normalization_hints": {},
                "case_columns": {},
                "date_columns": {},
                "fill_defaults": {},
                "cast_columns": {},
                "primary_key": ["id"],
                "recommended_sort": ["id"],
                "duplicate_rule": None,
            },
            "table_columns": ["id", "raw_name"],
            "issues_summary": ["Schema does not match the expected cleaned table columns."],
            "steps_taken": 0,
            "max_steps": 8,
            "current_score_estimate": 0.00001,
        },
        executed_actions=set(),
    )
    assert action.action_type.value == "profile_table"

    action = inference.fallback_action_from_observation(
        {
            "change_set_summary": {"profiled": True, "has_export_artifact": False, "published": False},
            "task_rules": {
                "rename_map": {"raw_name": "name"},
                "normalization_hints": {},
                "case_columns": {},
                "date_columns": {},
                "fill_defaults": {},
                "cast_columns": {},
                "primary_key": ["id"],
                "recommended_sort": ["id"],
                "duplicate_rule": None,
            },
            "table_columns": ["id", "raw_name"],
            "issues_summary": ["Schema does not match the expected cleaned table columns."],
            "steps_taken": 1,
            "max_steps": 8,
            "current_score_estimate": 0.00001,
            "validation_status": "not_run",
        },
        executed_actions=set(),
    )
    assert action.action_type.value == "rename_column"
    assert action.column == "raw_name"
    assert action.new_name == "name"
