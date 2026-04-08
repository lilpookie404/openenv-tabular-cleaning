"""Typed models for the tabular cleaning environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from .openenv_compat import Action, Observation, State


class ActionType(str, Enum):
    INSPECT_TABLE = "inspect_table"
    INSPECT_COLUMN = "inspect_column"
    RENAME_COLUMN = "rename_column"
    STRIP_WHITESPACE = "strip_whitespace"
    NORMALIZE_CASE = "normalize_case"
    REPLACE_VALUES = "replace_values"
    STANDARDIZE_DATE = "standardize_date"
    FILL_MISSING = "fill_missing"
    CAST_DTYPE = "cast_dtype"
    DROP_DUPLICATES = "drop_duplicates"
    SORT_ROWS = "sort_rows"
    SUBMIT = "submit"


class CaseMode(str, Enum):
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"


class TabularCleaningAction(Action):
    action_type: ActionType
    column: Optional[str] = None
    new_name: Optional[str] = None
    case_mode: Optional[CaseMode] = None
    replacements: Dict[str, str] = Field(default_factory=dict)
    fill_value: Optional[str] = None
    dtype: Optional[str] = None
    sort_by: List[str] = Field(default_factory=list)
    ascending: bool = True
    preview_rows: int = 5


class TabularCleaningObservation(Observation):
    task_id: str
    task_description: str
    table_columns: List[str]
    table_rows_preview: List[Dict[str, Any]]
    row_count: int
    issues_summary: List[str]
    last_action: Optional[Dict[str, Any]] = None
    last_action_error: Optional[str] = None
    steps_taken: int
    max_steps: int
    current_score_estimate: float
    available_actions: List[str]


class TabularCleaningState(State):
    task_id: str = ""
    current_table: List[Dict[str, Any]] = Field(default_factory=list)
    current_columns: List[str] = Field(default_factory=list)
    current_score: float = 0.0
    best_score_so_far: float = 0.0
    submitted: bool = False
    max_steps: int = 0
    task_rules: Dict[str, Any] = Field(default_factory=dict)
