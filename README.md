---
title: OpenEnv Tabular Cleaning
emoji: "🧹"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# openenv-tabular-cleaning

`tabular_cleaning_env` is a deterministic OpenEnv environment for real-world tabular data cleaning. The agent receives a messy table and must clean it with a structured, typed action space instead of editing arbitrary code. This keeps the benchmark easy to grade, easy to containerize, and aligned with the hackathon’s “real task, not a game” requirement.

## Why This Environment

People clean messy tables every day in operations, analytics, healthcare, finance, and support workflows. This environment simulates that workflow with tiny bundled datasets and deterministic graders so judges can see exactly what the agent is being rewarded for.

## Environment API

The environment follows the standard OpenEnv shape:

- `reset(task_id=...)` returns the initial observation for a chosen task.
- `step(action)` returns observation, reward, done, and `metadata` as the `info` equivalent.
- `state()` returns the current serializable internal state.

Core files:

- `server/environment.py`: `TabularCleaningEnvironment`
- `server/app.py`: FastAPI/OpenEnv app entrypoint
- `tabular_cleaning_env/models.py`: typed Action, Observation, State
- `tabular_cleaning_env/graders.py`: transparent deterministic grading
- `inference.py`: baseline inference runner with exact hackathon logging

## Observation Space

Each observation includes:

- `task_id`
- `task_description`
- `table_columns`
- `table_rows_preview`
- `row_count`
- `issues_summary`
- `last_action`
- `last_action_error`
- `steps_taken`
- `max_steps`
- `current_score_estimate`
- `available_actions`
- `done`
- `reward`
- `metadata`

## Action Space

The action space is typed and intentionally narrow:

- `inspect_table`
- `inspect_column`
- `rename_column`
- `strip_whitespace`
- `normalize_case`
- `replace_values`
- `standardize_date`
- `fill_missing`
- `cast_dtype`
- `drop_duplicates`
- `sort_rows`
- `submit`

The action model supports optional fields such as `column`, `new_name`, `case_mode`, `replacements`, `fill_value`, `dtype`, `sort_by`, `ascending`, and `preview_rows`.

## Tasks

Exactly three bundled tasks are included:

1. `easy_contacts_cleanup`
   Small employee/contact table with whitespace, case issues, a renamed column, department synonym normalization, one malformed date, and missing phone values.
2. `medium_orders_cleanup`
   Retail orders table with duplicate orders, inconsistent status labels, mixed numeric amount strings, inconsistent dates, and missing city/state values.
3. `hard_appointments_cleanup`
   Clinic appointments table with malformed timestamps, conflicting duplicates, inconsistent doctor and department labels, and deterministic conflict resolution based on completeness and latest `updated_at`.

Difficulty progression is controlled by both messiness and allowed step budget:

- Easy: `max_steps = 8`
- Medium: `max_steps = 10`
- Hard: `max_steps = 12`

## Graders

The grader is deterministic and transparent. For each task, the current table is compared against a bundled gold table using a weighted score in `[0.0, 1.0]`:

- `15%` schema correctness
- `20%` row-key and duplicate correctness
- `40%` exact cell correctness
- `15%` required-field completeness
- `10%` temporal normalization correctness

Gold tables score `1.0`. Raw input tables score below `1.0`. Partial cleanups score between the two.

## Reward Design

Reward shaping is nonnegative and bounded:

- Reward on each step is `max(0, current_score - best_score_so_far_before_action)`.
- Invalid, destructive, and no-op actions emit `0.0`.
- Harmful actions can lower the current score estimate, but they do not emit negative rewards.
- Episodes end on `submit` or when `max_steps` is reached.

`Observation.metadata` contains structured grader breakdowns and action diagnostics, which acts as the `info` equivalent.

## Baseline Results

The repo includes a deterministic fallback policy inside `inference.py`. Its reproducible baseline scores are:

- `easy_contacts_cleanup`: `1.00`
- `medium_orders_cleanup`: `1.00`
- `hard_appointments_cleanup`: `1.00`

When `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, `inference.py` first asks an OpenAI-compatible model for the next JSON action and falls back to the deterministic policy if parsing or validation fails.

## Local Setup

Python 3.11 is the safest local target because current official OpenEnv tooling requires Python `>=3.10`.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
python3 inference.py
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

If you want to use the official OpenEnv CLI path instead of the lightweight local fallback:

```bash
python3 -m uv python install 3.11
python3 -m uv lock --python 3.11
python3 -m uv run --python 3.11 openenv validate
python3 -m uv run --python 3.11 server
```

## Validation

Local validation commands:

```bash
python3 -m uv lock --python 3.11
python3 -m uv run --python 3.11 openenv validate
python3 -m uv run --python 3.11 pytest
```

The repo includes:

- `openenv.yaml`
- `pyproject.toml`
- `uv.lock`
- `server/app.py` with `main()`
- a root `Dockerfile`

## Docker

Build and run locally:

```bash
docker build -t tabular-cleaning-env .
docker run --rm -p 8000:8000 tabular-cleaning-env
```

Then validate the running server:

```bash
python3 -m uv run --python 3.11 openenv validate http://localhost:8000
```

## Hugging Face Spaces Deployment

This project is set up for a containerized Space:

1. Create a Docker Space on Hugging Face.
2. Push this repository as-is.
3. The Space will build from the root `Dockerfile`.
4. The app serves on port `8000`, matching `openenv.yaml` and the README front matter.

## Inference Runner

`inference.py` runs all three tasks in sequence and prints strict parser-safe logs:

```text
[START] task=<task_name> env=tabular_cleaning_env model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Run it with:

```bash
python3 inference.py
```

## Test Coverage

The test suite covers:

- model validation
- reset, step, and state behavior
- task graders on raw, partial, and gold tables
- reward bounds and termination logic
- inference log format
- app import smoke test
- README command coverage
