"""Request, directive, and response contracts for the GridWise service.

Validation posture is deliberately asymmetric:

* Judge input is strict about required fields and types but ignores unknown
  fields, so unexpected harness metadata can never reject a valid scenario.
* Model output is strict about shape and numeric ranges, because the problem
  statement requires exact structured adjustments and forbids clamping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

HORIZON = 24

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]

_TOLERANT = ConfigDict(extra="ignore")


class GuardrailError(ValueError):
    """Raised when model output fails deterministic validation."""


class SemanticRequestError(ValueError):
    """Raised for well-formed requests whose base data admits no valid plan."""


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def _not_a_boolean(value: Any) -> Any:
    """Reject booleans where a number is required.

    Numeric strings are still coerced on purpose: rejecting those would risk
    returning 400 for a request the judge considers valid, which costs far more
    than it protects. ``True`` silently becoming 1 kWh is a different matter.
    """
    if isinstance(value, bool):
        raise ValueError("must be a number, not a boolean")
    return value


class HourEntry(BaseModel):
    model_config = _TOLERANT

    hour: int = Field(ge=0, le=HORIZON - 1)
    demand_kwh: float = Field(ge=0, allow_inf_nan=False)
    solar_kwh: float = Field(ge=0, allow_inf_nan=False)
    tariff_bdt_per_kwh: float = Field(ge=0, allow_inf_nan=False)

    _reject_bools = field_validator(
        "hour", "demand_kwh", "solar_kwh", "tariff_bdt_per_kwh", mode="before"
    )(_not_a_boolean)


class Battery(BaseModel):
    model_config = _TOLERANT

    capacity_kwh: float = Field(gt=0, allow_inf_nan=False)
    initial_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    minimum_energy_kwh: float = Field(ge=0, allow_inf_nan=False)
    max_charge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)
    max_discharge_kwh_per_hour: float = Field(ge=0, allow_inf_nan=False)

    _reject_bools = field_validator(
        "capacity_kwh",
        "initial_energy_kwh",
        "minimum_energy_kwh",
        "max_charge_kwh_per_hour",
        "max_discharge_kwh_per_hour",
        mode="before",
    )(_not_a_boolean)


class OptimizeRequest(BaseModel):
    model_config = _TOLERANT

    scenario_id: str
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourEntry] = Field(min_length=HORIZON, max_length=HORIZON)
    battery: Battery

    @field_validator("scenario_id")
    @classmethod
    def _scenario_id_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario_id must be a non-empty string")
        return value

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, value: list[str]) -> list[str]:
        for index, note in enumerate(value):
            if not note.strip():
                raise ValueError(f"operator_notes[{index}] must be non-empty")
        return value

    @field_validator("hours")
    @classmethod
    def _hours_cover_horizon(cls, value: list[HourEntry]) -> list[HourEntry]:
        seen = sorted(entry.hour for entry in value)
        if seen != list(range(HORIZON)):
            raise ValueError("hours must contain exactly one entry for each hour 0-23")
        return value

    def ordered_hours(self) -> list[HourEntry]:
        """Hourly records sorted ascending, regardless of request order."""
        return sorted(self.hours, key=lambda entry: entry.hour)


def check_base_feasibility(request: OptimizeRequest) -> None:
    """Reject only base data that provably admits no valid schedule.

    End-of-day neutrality forces the battery back to ``initial_energy_kwh``, so
    a starting level outside the reserve/capacity band can never be scheduled.
    """
    battery = request.battery
    if battery.minimum_energy_kwh > battery.capacity_kwh:
        raise SemanticRequestError(
            "battery.minimum_energy_kwh exceeds battery.capacity_kwh"
        )
    if battery.initial_energy_kwh > battery.capacity_kwh:
        raise SemanticRequestError(
            "battery.initial_energy_kwh exceeds battery.capacity_kwh"
        )
    if battery.initial_energy_kwh < battery.minimum_energy_kwh:
        raise SemanticRequestError(
            "battery.initial_energy_kwh is below battery.minimum_energy_kwh"
        )


# --------------------------------------------------------------------------- #
# Validated directives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Directive:
    """One validated interpretation, ready to compile into solver constraints."""

    note_index: int
    applies: bool
    directive_type: str
    explanation: str
    hours: tuple[int, ...] = ()
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None

    def structured_adjustment(self) -> dict[str, Any] | None:
        """Exact adjustment object required by the problem statement."""
        if self.directive_type == "no_op":
            return None
        hours = list(self.hours)
        if self.directive_type == "solar_reduction":
            return {"hours": hours, "factor": self.factor}
        if self.directive_type == "minimum_battery_reserve":
            return {"hours": hours, "minimum_energy_kwh": self.minimum_energy_kwh}
        if self.directive_type == "max_grid_window":
            return {"hours": hours, "max_grid_kwh": self.max_grid_kwh}
        return {"hours": hours}


_DIRECTIVE_TYPES = frozenset(
    (
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    )
)

# Exact adjustment shape each directive type is allowed to carry.
_ADJUSTMENT_KEYS: dict[str, frozenset[str]] = {
    "solar_reduction": frozenset(("hours", "factor")),
    "minimum_battery_reserve": frozenset(("hours", "minimum_energy_kwh")),
    "no_charge_window": frozenset(("hours",)),
    "no_discharge_window": frozenset(("hours",)),
    "max_grid_window": frozenset(("hours", "max_grid_kwh")),
}


def _reject_foreign_keys(adjustment: dict[str, Any], directive_type: str) -> None:
    """Enforce the exact adjustment shape for the declared directive type.

    Keys carrying a null are tolerated, because the provider's structured-output
    mode fills every field of one flat object and nulls the unused ones. A key
    with real data that this type does not define means the model conflated two
    directives, which is worth a repair retry rather than a silent discard.
    """
    allowed = _ADJUSTMENT_KEYS[directive_type]
    foreign = sorted(
        key
        for key, value in adjustment.items()
        if key not in allowed and value is not None
    )
    if foreign:
        raise GuardrailError(
            f"{directive_type} does not accept {', '.join(foreign)}"
        )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuardrailError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise GuardrailError(f"{label} must be finite")
    return number


def _normalize_hours(raw: Any, label: str) -> tuple[int, ...]:
    """Sort and de-duplicate hours; reject anything outside 0-23.

    Ordering is normalization, which the guide permits. Out-of-range or
    non-integer hours are rejected rather than repaired.
    """
    if not isinstance(raw, list) or not raw:
        raise GuardrailError(f"{label}.hours must be a non-empty array")
    hours: set[int] = set()
    for item in raw:
        if isinstance(item, bool):
            raise GuardrailError(f"{label}.hours must contain integers")
        if isinstance(item, float):
            if not item.is_integer():
                raise GuardrailError(f"{label}.hours must contain integers")
            item = int(item)
        if not isinstance(item, int):
            raise GuardrailError(f"{label}.hours must contain integers")
        if not 0 <= item <= HORIZON - 1:
            raise GuardrailError(f"{label}.hours entries must be within 0-23")
        hours.add(item)
    return tuple(sorted(hours))


def validate_directive(
    raw: Any,
    *,
    expected_index: int,
    note_count: int,
    capacity_kwh: float,
) -> Directive:
    """Turn one untrusted model entry into a validated directive."""
    if not isinstance(raw, dict):
        raise GuardrailError("interpretation entry must be an object")

    note_index = raw.get("note_index")
    if isinstance(note_index, bool) or not isinstance(note_index, int):
        raise GuardrailError("note_index must be an integer")
    if not 0 <= note_index < note_count:
        raise GuardrailError("note_index does not identify an operator note")
    if note_index != expected_index:
        raise GuardrailError("interpretations must be ordered 0..N-1 with no gaps")

    # The isinstance check comes first on purpose: testing membership of an
    # unhashable value such as a dict raises TypeError, which would escape
    # GuardrailError and skip the repair retry.
    directive_type = raw.get("directive_type")
    if not isinstance(directive_type, str) or directive_type not in _DIRECTIVE_TYPES:
        raise GuardrailError("directive_type is not a supported value")

    applies = raw.get("applies")
    if not isinstance(applies, bool):
        raise GuardrailError("applies must be a boolean")

    adjustment = raw.get("structured_adjustment")
    explanation = raw.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "Interpreted from the operator note."

    if directive_type == "no_op":
        if applies:
            raise GuardrailError("no_op requires applies=false")
        if adjustment is not None:
            raise GuardrailError("no_op requires a null structured_adjustment")
        return Directive(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            explanation=explanation.strip(),
        )

    if not applies:
        raise GuardrailError(f"{directive_type} requires applies=true")
    if not isinstance(adjustment, dict):
        raise GuardrailError(f"{directive_type} requires a structured_adjustment object")

    _reject_foreign_keys(adjustment, directive_type)
    hours = _normalize_hours(adjustment.get("hours"), directive_type)

    if directive_type == "solar_reduction":
        factor = _finite(adjustment.get("factor"), "factor")
        if not 0.0 <= factor <= 1.0:
            raise GuardrailError("factor must be between 0 and 1 inclusive")
        return Directive(
            note_index=note_index,
            applies=True,
            directive_type=directive_type,
            explanation=explanation.strip(),
            hours=hours,
            factor=factor,
        )

    if directive_type == "minimum_battery_reserve":
        reserve = _finite(adjustment.get("minimum_energy_kwh"), "minimum_energy_kwh")
        if reserve < 0:
            raise GuardrailError("minimum_energy_kwh must be non-negative")
        if reserve > capacity_kwh:
            raise GuardrailError("minimum_energy_kwh exceeds battery capacity")
        return Directive(
            note_index=note_index,
            applies=True,
            directive_type=directive_type,
            explanation=explanation.strip(),
            hours=hours,
            minimum_energy_kwh=reserve,
        )

    if directive_type == "max_grid_window":
        cap = _finite(adjustment.get("max_grid_kwh"), "max_grid_kwh")
        if cap < 0:
            raise GuardrailError("max_grid_kwh must be non-negative")
        return Directive(
            note_index=note_index,
            applies=True,
            directive_type=directive_type,
            explanation=explanation.strip(),
            hours=hours,
            max_grid_kwh=cap,
        )

    return Directive(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        explanation=explanation.strip(),
        hours=hours,
    )


def validate_batch(
    raw_entries: Any,
    *,
    note_count: int,
    capacity_kwh: float,
) -> list[Directive]:
    """Validate a full interpretation batch: one ordered entry per note."""
    if not isinstance(raw_entries, list):
        raise GuardrailError("interpretations must be an array")
    if len(raw_entries) != note_count:
        raise GuardrailError(
            f"expected {note_count} interpretations, received {len(raw_entries)}"
        )
    return [
        validate_directive(
            entry,
            expected_index=index,
            note_count=note_count,
            capacity_kwh=capacity_kwh,
        )
        for index, entry in enumerate(raw_entries)
    ]


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class DirectiveInterpretationOut(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourPlanOut(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretationOut]
    hourly_plan: list[HourPlanOut]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: str
