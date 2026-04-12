---
title: Commerce Ops Cleanup Workbench
emoji: "📦"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

# openenv-tabular-cleaning

`tabular_cleaning_env` is a deterministic OpenEnv environment for a **commerce data operations cleanup workbench**.

It simulates a very real workflow inside ops, analytics, revops, and support teams: a messy export comes in from a business system, an agent profiles it, applies structured cleanup actions, gets risky changes reviewed, runs validations, and publishes an audited clean table for downstream use.

This is a governed data-cleaning workflow built around the standard OpenEnv API.

## What Problem It Solves

Teams routinely receive broken exports from CRMs, storefronts, and scheduling tools:

- columns are renamed or inconsistent
- dates use mixed formats
- labels drift over time
- required fields are missing
- duplicates appear after manual merges or sync bugs

In the real world, this work is often done manually in spreadsheets, with little auditability and a high risk of silently damaging data. This repository packages that workflow as a deterministic environment with transparent scoring.

## Real-World Workflow

The environment models a realistic internal workflow:

1. import a raw operational export
2. profile the table and inspect the change set
3. apply structured cleanup actions
4. approve or reject risky mutations
5. run validation gates
6. export a cleaned artifact bundle
7. publish the final table

That is the core product story: **a human-in-the-loop data cleanup workbench for commerce operations**.

## Design Goals

- The task models a concrete operational workflow.
- Grading is deterministic and transparent.
- Reward shaping is dense and bounded.
- The action space is structured, typed, and safe.
- The tasks cover realistic source systems without requiring external data or custom training.

## Environment API

The environment follows the standard OpenEnv shape:

- `reset(task_id=...)` returns the initial observation
- `step(action)` returns the next observation with `reward`, `done`, and `metadata`
- `state()` returns the full serializable internal state

Core implementation files:

- [inference.py](inference.py)
- [server/environment.py](server/environment.py)
- [server/app.py](server/app.py)
- [tabular_cleaning_env/models.py](tabular_cleaning_env/models.py)
- [tabular_cleaning_env/tasks.py](tabular_cleaning_env/tasks.py)
- [tabular_cleaning_env/graders.py](tabular_cleaning_env/graders.py)

## Bundled Tasks

The benchmark ships exactly 3 tasks with increasing difficulty.

| Task | Difficulty | Source System | Rows | What the agent must do |
|---|---|---|---|---|
| `easy_contacts_cleanup` | Easy | CRM customer contacts export | `18` raw / `18` gold | Fix schema drift, normalize names/emails/customer segments, standardize signup dates, fill missing phones, validate, export, publish |
| `medium_orders_cleanup` | Medium | E-commerce orders export | `20` raw / `16` gold | Normalize statuses and dates, cast amounts, fill missing location fields, remove true duplicates, validate, export, publish |
| `hard_appointments_cleanup` | Hard | Field-service scheduling export | `20` raw / `16` gold | Normalize technician and service-line labels, standardize timestamps, fill missing values, resolve duplicate conflicts deterministically, validate, export, publish |

Step budgets:

- Easy: `13`
- Medium: `15`
- Hard: `15`

## Sample Data

All bundled task data now uses a task-oriented folder structure under [tasks](tasks).
Each task ships with:

- `raw.csv`
- `ground_truth.csv`
- `metadata.json`

Task folders:

- [easy_contacts_cleanup](tasks/easy_contacts_cleanup)
- [medium_orders_cleanup](tasks/medium_orders_cleanup)
- [hard_appointments_cleanup](tasks/hard_appointments_cleanup)

Example dataset files:

- CRM contacts raw export: [tasks/easy_contacts_cleanup/raw.csv](tasks/easy_contacts_cleanup/raw.csv)
- CRM contacts cleaned reference: [tasks/easy_contacts_cleanup/ground_truth.csv](tasks/easy_contacts_cleanup/ground_truth.csv)
- Orders raw export: [tasks/medium_orders_cleanup/raw.csv](tasks/medium_orders_cleanup/raw.csv)
- Orders cleaned reference: [tasks/medium_orders_cleanup/ground_truth.csv](tasks/medium_orders_cleanup/ground_truth.csv)
- Service scheduling raw export: [tasks/hard_appointments_cleanup/raw.csv](tasks/hard_appointments_cleanup/raw.csv)
- Service scheduling cleaned reference: [tasks/hard_appointments_cleanup/ground_truth.csv](tasks/hard_appointments_cleanup/ground_truth.csv)

Current dataset sizes:

- contacts task: `18` raw rows and `18` gold rows
- orders task: `20` raw rows and `16` gold rows
- service scheduling task: `20` raw rows and `16` gold rows

Example dataset preview:

### CRM Contacts (`raw.csv`)

```csv
customer_id,full_name,email,customer_segment,signup_date,phone
C001," alice johnson ","ALICE.JOHNSON@EXAMPLE.COM "," vip ",2024/01/15," 555-0101 "
```

### Orders Export (`raw.csv`)

```csv
order_id,customer_name,status,amount,order_date,city,state
ORD-1001,Ava Patel," shipped ","$120.50",2024/03/01," Seattle ",WA
```

### Service Scheduling (`raw.csv`)

```csv
appointment_id,customer_name,service_line,technician,appointment_time,status,notes,updated_at
APT-001,"maya singh "," delivery ",alex cole,"2024/04/10 09:30",confirmed," gate code confirmed ","2024/04/01 08:00"
```

These are curated bundled datasets on purpose: they are large enough to feel like real cleanup work, still deterministic to grade, and still light enough to validate quickly in Docker or on Hugging Face Spaces.

## Action Space

The action space is typed and intentionally narrow.

Workflow actions:

- `profile_table`
- `view_change_set`
- `run_validations`
- `approve_changes`
- `reject_change`
- `export_cleaned_table`
- `publish_table`

Inspection and cleanup actions:

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

Supported action fields include:

- `column`
- `new_name`
- `case_mode`
- `replacements`
- `fill_value`
- `dtype`
- `sort_by`
- `ascending`
- `preview_rows`
- `change_id`
- `destination`

## Observation and State

Each observation contains the information an agent needs to act generically:

- task metadata such as `task_id`, `difficulty`, `source_system`, and `task_description`
- table context such as `table_columns`, `table_rows_preview`, `row_count`, and `issues_summary`
- workflow state such as `change_set_summary`, `risky_changes`, `validation_status`, and `export_ready`
- trajectory state such as `last_action`, `last_action_error`, `steps_taken`, and `current_score_estimate`
- `task_rules`, which define the cleaning contract for the current source-system export

The serialized state also tracks:

- current working table
- proposed, approved, and rejected changes
- validation results
- export artifacts
- append-only transformation log

## Rule Packs

Each task exposes a rule pack through `task_rules`, including:

- expected schema
- required columns
- primary key
- date columns
- normalization hints
- fill defaults
- dtype casts
- case normalization targets
- duplicate-resolution rules
- validation checks
- safe vs risky action types
- default export destination

This lets the baseline and any external agent behave generically instead of branching on task name.

## Deterministic Grading

Each task has a bundled reference table and one deterministic scalar grader that emits a task score strictly inside `(0, 1)`.

Grading behavior:

- the official task score comes only from the current cleaned table versus the bundled reference table
- grading aligns the configured output columns, optionally sorts rows by task metadata, and compares cells directly
- numeric grading columns use a small tolerance so formatted numeric outputs can still match exactly intended values
- workflow steps such as validation, export, and publish do not inflate the official task score

## Reward Design

Rewards are shaped but bounded:

- `reward = max(min_visible_reward, current_score - best_score_so_far_before_action)`
- invalid, destructive, or no-op actions emit the minimum visible reward floor instead of `0`
- risky actions may improve score immediately, but they still must be approved before validation/export/publish
- the episode ends when the table is published or when `max_steps` is reached

This produces dense learning signal without unstable negative rewards.

## Baseline Inference

The project includes a root [inference.py](inference.py) that:

- uses the `OpenAI` client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- requires `HF_TOKEN` with no default
- falls back to a deterministic rule-based planner if the LLM path fails
- emits the exact required parser-safe stdout format

The fallback planner follows the same governed workflow as the environment:

1. `profile_table`
2. apply the next cleanup step
3. `approve_changes` whenever a risky mutation is pending
4. `run_validations`
5. `export_cleaned_table`
6. `publish_table`

Reproducible baseline scores:

- `easy_contacts_cleanup`: `0.999`
- `medium_orders_cleanup`: `0.999`
- `hard_appointments_cleanup`: `0.999`

### Required Inference Log Format

```text
[START] task=<task_name> env=tabular_cleaning_env model=<model_name>
[STEP] step=<n> action=<action_str> reward=<reward_value> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

## Quick Start

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="<your-hf-token>"
python3 inference.py
```

### Example client usage

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

## Validation

Local validation commands:

```bash
python3 -m uv lock --python 3.11
python3 -m uv run --python 3.11 openenv validate
python3 -m uv run --python 3.11 pytest
```

The project includes the core runtime files:

- `openenv.yaml`
- `pyproject.toml`
- `uv.lock`
- root `Dockerfile`
- root `inference.py`
- `server/app.py` with `main()`

## Docker

Build and run locally:

```bash
docker build -t tabular-cleaning-env .
docker run --rm -p 8000:8000 tabular-cleaning-env
```

Then validate the live container:

```bash
python3 -m uv run --python 3.11 openenv validate http://localhost:8000
```

## Hugging Face Spaces

This project is designed for a containerized Hugging Face Space:

1. create a Docker Space
2. push this repository
3. let the Space build from the root `Dockerfile`
4. confirm the Space is `Running`
5. validate the public runtime

```bash
python3 -m uv run --python 3.11 openenv validate https://<your-space>.hf.space
```

## Export Artifacts

A successful run produces three downstream-ready artifacts in environment state:

- `cleaned_table`
- `data_quality_report`
- `transformation_audit_log`

That is the real-world value proposition: not just cleaning data, but cleaning it in a way that is reviewable, explainable, and safe to publish.

## Test Coverage

The test suite covers:

- model validation
- environment reset, step, and state behavior
- deterministic grading
- reward bounds
- workflow approval and publish semantics
- inference log formatting
- README command coverage
