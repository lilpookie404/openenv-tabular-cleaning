"""Bundled tabular cleaning tasks and deterministic policies."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .models import ActionType, CaseMode, TabularCleaningAction
from .utils import DATA_DIR


@dataclass(frozen=True)
class DuplicateRule:
    key_fields: Sequence[str]
    completeness_fields: Sequence[str]
    latest_timestamp_field: str | None = None


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: str
    description: str
    input_path: Path
    expected_path: Path
    expected_columns: Sequence[str]
    required_columns: Sequence[str]
    primary_key: Sequence[str]
    date_columns: Dict[str, bool]
    rename_map: Dict[str, str] = field(default_factory=dict)
    normalization_hints: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fill_defaults: Dict[str, str] = field(default_factory=dict)
    cast_columns: Dict[str, str] = field(default_factory=dict)
    case_columns: Dict[str, CaseMode] = field(default_factory=dict)
    max_steps: int = 8
    duplicate_rule: DuplicateRule | None = None

    @property
    def task_rules(self) -> Dict[str, Any]:
        return {
            "expected_columns": list(self.expected_columns),
            "required_columns": list(self.required_columns),
            "primary_key": list(self.primary_key),
            "date_columns": dict(self.date_columns),
            "rename_map": dict(self.rename_map),
            "fill_defaults": dict(self.fill_defaults),
            "cast_columns": dict(self.cast_columns),
            "case_columns": {key: value.value for key, value in self.case_columns.items()},
            "duplicate_rule": (
                {
                    "key_fields": list(self.duplicate_rule.key_fields),
                    "completeness_fields": list(self.duplicate_rule.completeness_fields),
                    "latest_timestamp_field": self.duplicate_rule.latest_timestamp_field,
                }
                if self.duplicate_rule
                else None
            ),
        }


def _data_path(name: str) -> Path:
    return DATA_DIR / name


TASKS: Dict[str, TaskDefinition] = {
    "easy_contacts_cleanup": TaskDefinition(
        task_id="easy_contacts_cleanup",
        difficulty="easy",
        description=(
            "Clean a small employee contact table by renaming columns, trimming whitespace, "
            "normalizing department labels and dates, and filling constant missing values."
        ),
        input_path=_data_path("easy_input.json"),
        expected_path=_data_path("easy_expected.json"),
        expected_columns=["employee_id", "name", "email", "department", "start_date", "phone"],
        required_columns=["employee_id", "name", "email", "department", "start_date", "phone"],
        primary_key=["employee_id"],
        date_columns={"start_date": False},
        rename_map={"full_name": "name"},
        normalization_hints={
            "department": {
                "hr": "Human Resources",
                "human resources": "Human Resources",
                "engineering": "Engineering",
                "eng": "Engineering",
                "finance": "Finance",
            }
        },
        fill_defaults={"phone": "UNKNOWN"},
        case_columns={"name": CaseMode.TITLE, "email": CaseMode.LOWER},
        max_steps=8,
    ),
    "medium_orders_cleanup": TaskDefinition(
        task_id="medium_orders_cleanup",
        difficulty="medium",
        description=(
            "Clean a retail orders table by normalizing statuses and dates, casting amounts, "
            "filling missing city/state values, and removing only true duplicates."
        ),
        input_path=_data_path("medium_input.json"),
        expected_path=_data_path("medium_expected.json"),
        expected_columns=["order_id", "customer_name", "status", "amount", "order_date", "city", "state"],
        required_columns=["order_id", "customer_name", "status", "amount", "order_date", "city", "state"],
        primary_key=["order_id"],
        date_columns={"order_date": False},
        normalization_hints={
            "status": {
                " shipped ": "shipped",
                "shipped": "shipped",
                "shipped": "shipped",
                "pending": "pending",
                " pending ": "pending",
                "cancelled": "cancelled",
                "canceled": "cancelled",
            }
        },
        fill_defaults={"city": "UNKNOWN", "state": "UNKNOWN"},
        cast_columns={"amount": "float"},
        max_steps=10,
        duplicate_rule=DuplicateRule(
            key_fields=["order_id"],
            completeness_fields=["customer_name", "status", "amount", "order_date", "city", "state"],
        ),
    ),
    "hard_appointments_cleanup": TaskDefinition(
        task_id="hard_appointments_cleanup",
        difficulty="hard",
        description=(
            "Clean a clinic appointments table by standardizing timestamps, normalizing doctor "
            "and department labels, filling required fields, and resolving conflicting duplicates "
            "with a deterministic business rule."
        ),
        input_path=_data_path("hard_input.json"),
        expected_path=_data_path("hard_expected.json"),
        expected_columns=[
            "appointment_id",
            "patient_name",
            "department",
            "doctor",
            "appointment_time",
            "status",
            "notes",
            "updated_at",
        ],
        required_columns=[
            "appointment_id",
            "patient_name",
            "department",
            "doctor",
            "appointment_time",
            "status",
            "notes",
            "updated_at",
        ],
        primary_key=["appointment_id"],
        date_columns={"appointment_time": True, "updated_at": True},
        normalization_hints={
            "department": {
                "cardio": "Cardiology",
                "cardiology": "Cardiology",
                "ortho": "Orthopedics",
                "orthopedics": "Orthopedics",
                "neurology": "Neurology",
                "neuro": "Neurology",
            },
            "doctor": {
                "dr. anne li": "Dr. Anne Li",
                "dr anne li": "Dr. Anne Li",
                "anne li": "Dr. Anne Li",
                "dr. omar reed": "Dr. Omar Reed",
                "omar reed": "Dr. Omar Reed",
                "dr omar reed": "Dr. Omar Reed",
                "dr. jo park": "Dr. Jo Park",
                "jo park": "Dr. Jo Park",
            },
        },
        fill_defaults={"doctor": "TBD", "notes": "UNKNOWN"},
        case_columns={"patient_name": CaseMode.TITLE},
        max_steps=12,
        duplicate_rule=DuplicateRule(
            key_fields=["appointment_id"],
            completeness_fields=[
                "patient_name",
                "department",
                "doctor",
                "appointment_time",
                "status",
                "notes",
                "updated_at",
            ],
            latest_timestamp_field="updated_at",
        ),
    ),
}


def load_table(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def load_task_input(task_id: str) -> List[Dict[str, Any]]:
    return load_table(get_task(task_id).input_path)


def load_task_expected(task_id: str) -> List[Dict[str, Any]]:
    return load_table(get_task(task_id).expected_path)


FALLBACK_POLICIES: Dict[str, List[TabularCleaningAction]] = {
    "easy_contacts_cleanup": [
        TabularCleaningAction(action_type=ActionType.RENAME_COLUMN, column="full_name", new_name="name"),
        TabularCleaningAction(action_type=ActionType.STRIP_WHITESPACE),
        TabularCleaningAction(action_type=ActionType.NORMALIZE_CASE, column="name", case_mode=CaseMode.TITLE),
        TabularCleaningAction(action_type=ActionType.NORMALIZE_CASE, column="email", case_mode=CaseMode.LOWER),
        TabularCleaningAction(
            action_type=ActionType.REPLACE_VALUES,
            column="department",
            replacements={"hr": "Human Resources", "human resources": "Human Resources", "eng": "Engineering"},
        ),
        TabularCleaningAction(action_type=ActionType.STANDARDIZE_DATE, column="start_date"),
        TabularCleaningAction(action_type=ActionType.FILL_MISSING, column="phone", fill_value="UNKNOWN"),
        TabularCleaningAction(action_type=ActionType.SUBMIT),
    ],
    "medium_orders_cleanup": [
        TabularCleaningAction(action_type=ActionType.STRIP_WHITESPACE),
        TabularCleaningAction(
            action_type=ActionType.REPLACE_VALUES,
            column="status",
            replacements={" shipped ": "shipped", "Shipped": "shipped", " pending ": "pending", "canceled": "cancelled"},
        ),
        TabularCleaningAction(action_type=ActionType.STANDARDIZE_DATE, column="order_date"),
        TabularCleaningAction(action_type=ActionType.CAST_DTYPE, column="amount", dtype="float"),
        TabularCleaningAction(action_type=ActionType.FILL_MISSING, column="city", fill_value="UNKNOWN"),
        TabularCleaningAction(action_type=ActionType.FILL_MISSING, column="state", fill_value="UNKNOWN"),
        TabularCleaningAction(action_type=ActionType.DROP_DUPLICATES),
        TabularCleaningAction(action_type=ActionType.SUBMIT),
    ],
    "hard_appointments_cleanup": [
        TabularCleaningAction(action_type=ActionType.STRIP_WHITESPACE),
        TabularCleaningAction(action_type=ActionType.NORMALIZE_CASE, column="patient_name", case_mode=CaseMode.TITLE),
        TabularCleaningAction(
            action_type=ActionType.REPLACE_VALUES,
            column="department",
            replacements={"cardio": "Cardiology", "ortho": "Orthopedics", "neuro": "Neurology"},
        ),
        TabularCleaningAction(
            action_type=ActionType.REPLACE_VALUES,
            column="doctor",
            replacements={
                "anne li": "Dr. Anne Li",
                "dr anne li": "Dr. Anne Li",
                "dr. anne li": "Dr. Anne Li",
                "omar reed": "Dr. Omar Reed",
                "dr omar reed": "Dr. Omar Reed",
                "dr. omar reed": "Dr. Omar Reed",
                "jo park": "Dr. Jo Park",
                "dr jo park": "Dr. Jo Park",
                "dr. jo park": "Dr. Jo Park",
            },
        ),
        TabularCleaningAction(action_type=ActionType.STANDARDIZE_DATE),
        TabularCleaningAction(action_type=ActionType.FILL_MISSING, column="doctor", fill_value="TBD"),
        TabularCleaningAction(action_type=ActionType.FILL_MISSING, column="notes", fill_value="UNKNOWN"),
        TabularCleaningAction(action_type=ActionType.DROP_DUPLICATES),
        TabularCleaningAction(action_type=ActionType.SUBMIT),
    ],
}
