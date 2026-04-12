"""Bundled tabular cleaning tasks and deterministic metadata."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .models import ActionType, CaseMode
from .utils import TASKS_DIR


@dataclass(frozen=True)
class DuplicateRule:
    key_fields: Sequence[str]
    completeness_fields: Sequence[str]
    latest_timestamp_field: str | None = None


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: str
    domain: str
    source_system: str
    rule_pack_name: str
    description: str
    input_path: Path
    expected_path: Path
    expected_columns: Sequence[str]
    required_columns: Sequence[str]
    primary_key: Sequence[str]
    date_columns: Dict[str, bool]
    grade_columns: Sequence[str] = field(default_factory=tuple)
    sort_rows: bool = True
    rename_map: Dict[str, str] = field(default_factory=dict)
    normalization_hints: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fill_defaults: Dict[str, str] = field(default_factory=dict)
    cast_columns: Dict[str, str] = field(default_factory=dict)
    case_columns: Dict[str, CaseMode] = field(default_factory=dict)
    recommended_sort: Sequence[str] = field(default_factory=tuple)
    validation_rules: Dict[str, str] = field(default_factory=dict)
    risky_action_types: Sequence[ActionType] = field(default_factory=tuple)
    default_export_destination: str = "warehouse_ready_json"
    max_steps: int = 12
    duplicate_rule: DuplicateRule | None = None

    @property
    def safe_action_types(self) -> List[str]:
        candidates = [
            ActionType.STRIP_WHITESPACE,
            ActionType.NORMALIZE_CASE,
            ActionType.REPLACE_VALUES,
            ActionType.STANDARDIZE_DATE,
            ActionType.SORT_ROWS,
        ]
        risky = {action.value for action in self.risky_action_types}
        return [action.value for action in candidates if action.value not in risky]

    @property
    def task_rules(self) -> Dict[str, Any]:
        return {
            "source_system": self.source_system,
            "rule_pack_name": self.rule_pack_name,
            "expected_columns": list(self.expected_columns),
            "required_columns": list(self.required_columns),
            "primary_key": list(self.primary_key),
            "date_columns": dict(self.date_columns),
            "grade_columns": list(self.grade_columns or self.expected_columns),
            "sort_rows": self.sort_rows,
            "rename_map": dict(self.rename_map),
            "normalization_hints": {
                column: dict(mapping) for column, mapping in self.normalization_hints.items()
            },
            "fill_defaults": dict(self.fill_defaults),
            "cast_columns": dict(self.cast_columns),
            "case_columns": {key: value.value for key, value in self.case_columns.items()},
            "recommended_sort": list(self.recommended_sort or self.primary_key),
            "validation_rules": dict(self.validation_rules),
            "safe_action_types": list(self.safe_action_types),
            "risky_action_types": [action.value for action in self.risky_action_types],
            "default_export_destination": self.default_export_destination,
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


def _task_dir(task_id: str) -> Path:
    return TASKS_DIR / task_id


def _task_file(task_id: str, filename: str) -> Path:
    return _task_dir(task_id) / filename


COMMON_RISKY_ACTIONS = (
    ActionType.RENAME_COLUMN,
    ActionType.FILL_MISSING,
    ActionType.CAST_DTYPE,
    ActionType.DROP_DUPLICATES,
)

INTEGER_RE = re.compile(r"^-?\d+$")
FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


TASKS: Dict[str, TaskDefinition] = {
    "easy_contacts_cleanup": TaskDefinition(
        task_id="easy_contacts_cleanup",
        difficulty="easy",
        domain="customer operations",
        source_system="crm_customer_contacts_export",
        rule_pack_name="customer_contacts_cleanup_pack",
        description=(
            "Clean a CRM customer contacts export by fixing schema drift, normalizing names, "
            "emails, customer segments, and signup dates, then validating and publishing a "
            "warehouse-ready contacts table."
        ),
        input_path=_task_file("easy_contacts_cleanup", "raw.csv"),
        expected_path=_task_file("easy_contacts_cleanup", "ground_truth.csv"),
        expected_columns=["customer_id", "contact_name", "email", "customer_segment", "signup_date", "phone"],
        required_columns=["customer_id", "contact_name", "email", "customer_segment", "signup_date", "phone"],
        primary_key=["customer_id"],
        date_columns={"signup_date": False},
        rename_map={"full_name": "contact_name"},
        normalization_hints={
            "customer_segment": {
                "vip": "VIP",
                "wholesale": "Wholesale",
                "retail": "Retail",
                "loyalty": "Loyalty",
            }
        },
        fill_defaults={"phone": "UNKNOWN"},
        case_columns={"contact_name": CaseMode.TITLE, "email": CaseMode.LOWER},
        recommended_sort=("customer_id",),
        validation_rules={
            "required_fields_present": "All required customer contact fields are populated.",
            "schema_matches": "The cleaned table matches the published customer contacts schema.",
            "dates_canonical": "Signup dates use the canonical YYYY-MM-DD format.",
            "emails_valid": "Emails are syntactically valid and contain no spaces.",
        },
        risky_action_types=(ActionType.RENAME_COLUMN, ActionType.FILL_MISSING),
        default_export_destination="customer_contacts_warehouse_ready_json",
        max_steps=13,
    ),
    "medium_orders_cleanup": TaskDefinition(
        task_id="medium_orders_cleanup",
        difficulty="medium",
        domain="retail operations",
        source_system="shopify_orders_export",
        rule_pack_name="orders_cleanup_pack",
        description=(
            "Clean a retail orders export by standardizing statuses and dates, casting amounts, "
            "reviewing imputed location fields, resolving true duplicates, and publishing an "
            "operations-ready orders dataset."
        ),
        input_path=_task_file("medium_orders_cleanup", "raw.csv"),
        expected_path=_task_file("medium_orders_cleanup", "ground_truth.csv"),
        expected_columns=["order_id", "customer_name", "status", "amount", "order_date", "city", "state"],
        required_columns=["order_id", "customer_name", "status", "amount", "order_date", "city", "state"],
        primary_key=["order_id"],
        date_columns={"order_date": False},
        normalization_hints={
            "status": {
                " shipped ": "shipped",
                "shipped": "shipped",
                "pending": "pending",
                " pending ": "pending",
                "cancelled": "cancelled",
                "canceled": "cancelled",
            }
        },
        fill_defaults={"city": "UNKNOWN", "state": "UNKNOWN"},
        cast_columns={"amount": "float"},
        recommended_sort=("order_id",),
        validation_rules={
            "required_fields_present": "All required order fields are populated.",
            "schema_matches": "The cleaned table matches the published orders schema.",
            "duplicates_resolved": "Order business keys are unique.",
            "amounts_numeric_non_negative": "Amounts are numeric and non-negative.",
            "dates_canonical": "Order dates use the canonical YYYY-MM-DD format.",
        },
        risky_action_types=COMMON_RISKY_ACTIONS,
        default_export_destination="orders_warehouse_ready_json",
        max_steps=15,
        duplicate_rule=DuplicateRule(
            key_fields=["order_id"],
            completeness_fields=["customer_name", "status", "amount", "order_date", "city", "state"],
        ),
    ),
    "hard_appointments_cleanup": TaskDefinition(
        task_id="hard_appointments_cleanup",
        difficulty="hard",
        domain="field service operations",
        source_system="field_service_scheduler_export",
        rule_pack_name="service_appointments_cleanup_pack",
        description=(
            "Clean a field-service appointments export by standardizing timestamps, normalizing "
            "technician and service-line labels, reviewing imputed values, resolving risky duplicate "
            "conflicts, and publishing an audited service scheduling table."
        ),
        input_path=_task_file("hard_appointments_cleanup", "raw.csv"),
        expected_path=_task_file("hard_appointments_cleanup", "ground_truth.csv"),
        expected_columns=[
            "appointment_id",
            "customer_name",
            "service_line",
            "technician",
            "appointment_time",
            "status",
            "notes",
            "updated_at",
        ],
        required_columns=[
            "appointment_id",
            "customer_name",
            "service_line",
            "technician",
            "appointment_time",
            "status",
            "notes",
            "updated_at",
        ],
        primary_key=["appointment_id"],
        date_columns={"appointment_time": True, "updated_at": True},
        normalization_hints={
            "service_line": {
                "delivery": "Delivery",
                "install": "Installation",
                "installation": "Installation",
                "return pickup": "Returns",
                "returns": "Returns",
            },
            "technician": {
                "alex cole": "Alex Cole",
                "sam reed": "Sam Reed",
                "jo park": "Jo Park",
            },
        },
        fill_defaults={"technician": "TBD", "notes": "UNKNOWN"},
        case_columns={"customer_name": CaseMode.TITLE},
        recommended_sort=("appointment_id",),
        validation_rules={
            "required_fields_present": "All required field-service scheduling fields are populated.",
            "schema_matches": "The cleaned table matches the published service scheduling schema.",
            "duplicates_resolved": "Appointment business keys are unique.",
            "timestamps_canonical": "Appointment and update timestamps are canonical ISO values.",
            "technician_assignments_valid": "Technician fields are populated or intentionally reviewed placeholders.",
        },
        risky_action_types=COMMON_RISKY_ACTIONS,
        default_export_destination="service_schedule_ready_json",
        max_steps=15,
        duplicate_rule=DuplicateRule(
            key_fields=["appointment_id"],
            completeness_fields=[
                "customer_name",
                "service_line",
                "technician",
                "appointment_time",
                "status",
                "notes",
                "updated_at",
            ],
            latest_timestamp_field="updated_at",
        ),
    ),
}


def _parse_csv_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if INTEGER_RE.match(text):
        return int(text)
    if FLOAT_RE.match(text):
        return float(text)
    return value


def load_table(path: Path, infer_numbers: bool = False) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not infer_numbers:
        return rows
    return [
        {key: _parse_csv_scalar(value) for key, value in row.items()}
        for row in rows
    ]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def load_task_input(task_id: str) -> List[Dict[str, Any]]:
    return load_table(get_task(task_id).input_path, infer_numbers=False)


def load_task_expected(task_id: str) -> List[Dict[str, Any]]:
    return load_table(get_task(task_id).expected_path, infer_numbers=True)


def load_task_metadata(task_id: str) -> Dict[str, Any]:
    path = _task_file(task_id, "metadata.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
