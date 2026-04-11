---
title: OpenEnv Tabular Cleaning
emoji: "🧹"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# openenv-tabular-cleaning

`tabular_cleaning_env` is a deterministic OpenEnv environment for a human-in-the-loop operational data cleanup workbench. Instead of treating table cleaning as a magic black-box transformation, the environment models the real workflow analysts and ops teams follow: profile a messy export, apply structured fixes, review risky changes, run validation gates, and export or publish an audited downstream-ready table.

## Why This Environment

People clean messy CSV and JSON exports every day in operations, analytics, healthcare, finance, and support workflows. The real pain is not just fixing whitespace or dates; it is making traceable changes without silently losing rows or pushing bad data downstream. This environment simulates that governed workflow with tiny bundled datasets and deterministic graders so judges can see exactly what the agent is being rewarded for.

## Why This Is A Real Benchmark

The benchmark models work that data analysts, operations teams, and healthcare admins genuinely do:

- resolve schema drift in exports
- normalize messy categorical labels and timestamps
- fill missing required values with documented defaults
- remove duplicates using a deterministic business rule
- review higher-risk changes before downstream publication
- validate cleaned data before export or publish
- preserve an audit trail for what changed and why

The grader is transparent and field-level. Agents only receive full credit when the current table actually matches the cleaned target table, including canonical date formatting and duplicate resolution outcomes.

## Real-World Workflow

The environment frames the task as an internal cleanup workbench:

1. import a raw operational export from a source system
2. profile the table and inspect suggested safe vs risky changes
3. apply cleanup actions through a typed action space
4. approve or reject risky changes such as schema renames, imputations, type casts, or duplicate removal
5. run validation gates
6. export a cleaned table plus validation report and audit log
7. publish the cleaned artifact for downstream use

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
- `difficulty`
- `source_system`
- `task_description`
- `task_rules`
- `table_columns`
- `table_rows_preview`
- `row_count`
- `issues_summary`
- `change_set_summary`
- `proposed_changes_summary`
- `risky_changes`
- `validation_status`
- `validation_checks`
- `audit_log_preview`
- `export_ready`
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

- `profile_table`
- `view_change_set`
- `run_validations`
- `approve_changes`
- `reject_change`
- `export_cleaned_table`
- `publish_table`
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

The action model supports optional fields such as `column`, `new_name`, `case_mode`, `replacements`, `fill_value`, `dtype`, `sort_by`, `ascending`, `preview_rows`, `change_id`, and `destination`.

`task_rules` gives agents the cleaning contract they need to act generically:

- source system and rule pack name
- expected columns and primary key
- date columns and required canonical formats
- normalization hints
- constant fill defaults
- dtype casts
- case normalization targets
- duplicate-resolution rule
- validation rules
- safe vs risky action types
- default export destination
- recommended sort order

## Tasks

Exactly three bundled tasks are included:

1. `easy_contacts_cleanup`
   A Workday-style employee/contact export with whitespace, case issues, a renamed column, department synonym normalization, malformed dates, and missing phone values. Rule pack: `contacts_cleanup_pack`.
2. `medium_orders_cleanup`
   A Shopify-style retail orders export with duplicate orders, inconsistent status labels, mixed numeric amount strings, inconsistent dates, and missing city/state values. Rule pack: `orders_cleanup_pack`.
3. `hard_appointments_cleanup`
   A clinic scheduling export with malformed timestamps, conflicting duplicates, inconsistent doctor and department labels, and deterministic conflict resolution based on completeness and latest `updated_at`. Rule pack: `appointments_cleanup_pack`.

Difficulty progression is controlled by both messiness and allowed step budget:

- Easy: `max_steps = 13`
- Medium: `max_steps = 15`
- Hard: `max_steps = 15`

## Graders

The grader is deterministic and transparent. For each task, the current table is compared against a bundled gold table using a weighted score in `[0.0, 1.0]`:

- `15%` schema correctness
- `20%` row-key and duplicate correctness
- `40%` exact cell correctness on the agent's current cleaned table
- `15%` required-field completeness
- `10%` temporal normalization correctness

Gold tables score `1.0`. Raw input tables score below `1.0`. Partial cleanups score between the two. Date and datetime columns only receive full temporal credit when the agent has already converted them into the canonical task format.

## Reward Design

Reward shaping is nonnegative and bounded:

- Reward on each step is `max(0, current_score - best_score_so_far_before_action)`.
- Invalid, destructive, and no-op actions emit `0.0`.
- Harmful actions can lower the current score estimate, but they do not emit negative rewards.
- Risky changes can improve score immediately, but they must still be approved before validation, export, and publish.
- Episodes end when the table has been published or when `max_steps` is reached.

`Observation.metadata` contains structured grader breakdowns and action diagnostics, which acts as the `info` equivalent.

## Baseline Results

The repo includes a deterministic rule-based fallback planner inside `inference.py`. It reads `task_rules`, `change_set_summary`, `risky_changes`, and validation/export state from the observation instead of branching on task id. The planner follows the governed workflow:

1. `profile_table`
2. apply the next cleanup action
3. `approve_changes` whenever a risky mutation is pending
4. `run_validations`
5. `export_cleaned_table`
6. `publish_table`

Its reproducible baseline scores are:

- `easy_contacts_cleanup`: `1.00`
- `medium_orders_cleanup`: `1.00`
- `hard_appointments_cleanup`: `1.00`

When `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, `inference.py` first asks an OpenAI-compatible model for the next JSON action and falls back to the deterministic planner if the API call fails, the JSON is invalid, or local action validation fails. Endpoint, auth, and timeout failures trip a one-way circuit breaker so the rest of the run does not keep waiting on a broken LLM path.

## Local Setup

Python 3.11 is the safest local target because current official OpenEnv tooling requires Python `>=3.10`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Quick Start

Run the inference baseline with the required environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="<your-hf-token>"
python3 inference.py
```

Connect to the running environment server:

```python
from tabular_cleaning_env.client import TabularCleaningEnv
from tabular_cleaning_env.models import TabularCleaningAction

env = TabularCleaningEnv(base_url="http://localhost:8000")
result = env.reset(task_id="easy_contacts_cleanup")
print(result.observation.task_rules)
result = env.step(TabularCleaningAction(action_type="profile_table"))
print(result.observation.metadata)
env.close()
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
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Environment variables:

- `API_BASE_URL` with a default value
- `MODEL_NAME` with a default value
- `HF_TOKEN` with no default and required at runtime

Run it with:

```bash
export HF_TOKEN="<your-hf-token>"
python3 inference.py
```

`HF_TOKEN` is required and has no default. `API_BASE_URL` and `MODEL_NAME` include defaults.

## Export Artifacts

After a successful run, the workbench produces three downstream-ready artifacts in environment state:

- `cleaned_table`
- `data_quality_report`
- `transformation_audit_log`

This mirrors the real operational need to not only clean data, but also explain what changed and why it was safe to publish.

## Test Coverage

The test suite covers:

- model validation
- reset, step, and state behavior
- task graders on raw, partial, and gold tables
- reward bounds and termination logic
- inference log format
- app import smoke test
- README command coverage
