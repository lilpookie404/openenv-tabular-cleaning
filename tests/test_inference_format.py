from __future__ import annotations

from typing import Any, Dict, List

import inference


def test_inference_prints_required_sections(capsys) -> None:
    inference.run_task("easy_contacts_cleanup", client=None, model_name="deterministic-fallback")
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith("[START] task=easy_contacts_cleanup env=tabular_cleaning_env model=deterministic-fallback")
    assert any(line.startswith("[STEP]") for line in output)
    assert output[-1].startswith("[END] success=true steps=")


def test_inference_always_emits_end_on_exception(capsys) -> None:
    class ExplodingEnv:
        def __init__(self) -> None:
            pass

        def reset(self, **_: Any) -> Any:
            raise RuntimeError("boom")

        def close(self) -> None:
            return None

    inference.run_task("easy_contacts_cleanup", client=None, model_name="deterministic-fallback", env_factory=ExplodingEnv)
    output = capsys.readouterr().out.strip().splitlines()
    assert output[0].startswith("[START]")
    assert output[-1].startswith("[END] success=false steps=0 score=0.000 rewards=")
