"""Operator-note interpretation through a language model.

The model is the only component that reads natural language. Everything it
returns is treated as untrusted data and must survive ``schemas.validate_batch``
before it can reach the optimizer. There is deliberately no phrase-matching
interpreter behind it: normalization and validation assist the model, they never
replace it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from openai import APIStatusError, AsyncOpenAI

from schemas import Directive, GuardrailError, validate_batch

logger = logging.getLogger("gridwise.llm")

DIRECTIVE_TYPES = (
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
)


class ProviderError(RuntimeError):
    """The model could not be reached or produced nothing usable."""


SYSTEM_PROMPT = """\
You convert short campus-operator notes into structured directives for an energy optimizer.

Return exactly one entry per note, in input order, using note_index 0..N-1.

SUPPORTED DIRECTIVE TYPES (no others exist)
- solar_reduction {hours, factor}: usable rooftop solar is reduced during those hours.
- minimum_battery_reserve {hours, minimum_energy_kwh}: stored battery energy must stay at or above a level during those hours.
- no_charge_window {hours}: battery charging is unavailable during those hours.
- no_discharge_window {hours}: battery discharging is unavailable during those hours.
- max_grid_window {hours, max_grid_kwh}: grid import per hour is capped during those hours.
- no_op: the note does not change today's 24-hour electricity schedule.

HOURS
Whole-hour integers 0-23, unique, sorted ascending. Windows are start-inclusive and end-exclusive.
  "1 PM to 3 PM"                -> [13, 14]
  "between 13:00 and 15:00"     -> [13, 14]
  "from 6 PM until 9 PM"        -> [18, 19, 20]
  "from noon until 2 PM"        -> [12, 13]          (noon is hour 12)
  "11 AM until 1 PM"            -> [11, 12]
  "from 2 AM until 5 AM"        -> [2, 3, 4]
  "midnight to 3 AM"            -> [0, 1, 2]         (midnight is hour 0)
  "10 PM to 2 AM"               -> [0, 1, 22, 23]    (wraps past midnight, still sorted ascending)
  "at 3 PM"                     -> [15]
  "for three hours from 2 PM"   -> [14, 15, 16]

SOLAR FACTOR
factor is the fraction of forecast solar that still REMAINS usable, from 0 to 1.
A stated reduction must be subtracted from 1. A stated remaining level is used directly.
  "drops to about 25%"                        -> 0.25
  "treat usable solar as roughly 25% of forecast" -> 0.25
  "an 80% reduction in solar"                 -> 0.2
  "solar reduced by 80%"                      -> 0.2
  "about half of the forecast output"         -> 0.5
  "solar will be halved"                      -> 0.5
  "roughly one-fifth of normal output"        -> 0.2
  "panels offline" or "no solar output"       -> 0.0

RESERVE
minimum_energy_kwh is absolute kWh. Convert relative wording using the stated battery capacity.
  "keep at least 120 kWh"                              -> 120
  "keep at least 50% of capacity" (capacity 200 kWh)   -> 100
  "keep the battery at least a quarter full" (240 kWh) -> 60

GRID CAP
max_grid_kwh is the per-hour ceiling on grid import, never a total for the window.
  "must not exceed 155 kWh in any hour" -> 155
  "the transformer limit is 180 kWh of grid import" -> 180

WHEN TO USE no_op
Use no_op when the note does not change the electricity schedule at all: room bookings,
cafeteria menus, registration deadlines, notices, staffing, sports or library news.
Also use no_op when the note is about energy but cannot be expressed as one of the five
types above, for example a change in the demand forecast, a tariff or price change, or
equipment news that does not affect solar, charging, discharging, reserve, or grid import.
Never invent a directive type and never force an unrelated note into one.

OUTPUT RULES
- Exactly one entry per note, ordered by note_index.
- no_op: applies = false and every structured_adjustment field null.
- Any other type: applies = true, fill only the fields that type requires, leave the rest null.
- Each note maps to exactly one directive type. If a note seems to mention two, choose the one it actually instructs.
- Never alter demand, tariff, or battery parameters.
- explanation: one short sentence.\
"""

_ADJUSTMENT_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "hours": {"type": ["array", "null"], "items": {"type": "integer"}},
        "factor": {"type": ["number", "null"]},
        "minimum_energy_kwh": {"type": ["number", "null"]},
        "max_grid_kwh": {"type": ["number", "null"]},
    },
    "required": ["hours", "factor", "minimum_energy_kwh", "max_grid_kwh"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {"type": "string", "enum": list(DIRECTIVE_TYPES)},
                    "structured_adjustment": _ADJUSTMENT_SCHEMA,
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "applies",
                    "directive_type",
                    "structured_adjustment",
                    "explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["interpretations"],
    "additionalProperties": False,
}


@dataclass
class Settings:
    base_url: str
    api_key: str
    model: str
    transport: str = "responses"
    timeout_seconds: float = 12.0
    max_output_tokens: int = 1200
    temperature: float | None = 0.0
    reasoning_effort: str | None = None
    fallback_model: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.getenv(
            "AZURE_AI_BASE_URL",
            "https://ai-for-security.services.ai.azure.com/openai/v1/",
        )
        api_key = os.getenv("AZURE_AI_API_KEY", "")
        if not api_key:
            raise RuntimeError("AZURE_AI_API_KEY is not configured")
        # The tested gpt-5.6-terra deployment rejects an explicit temperature, so
        # it is omitted by default; sending it would waste a round trip on the
        # first request of every process. Set LLM_TEMPERATURE to pin it on a
        # deployment that accepts sampling parameters.
        temperature_raw = os.getenv("LLM_TEMPERATURE", "")
        effort = os.getenv("LLM_REASONING_EFFORT") or None
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=os.getenv("AZURE_AI_MODEL", "gpt-5.6-terra"),
            transport=os.getenv("LLM_TRANSPORT", "responses").strip().lower(),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "12")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200")),
            temperature=None if temperature_raw == "" else float(temperature_raw),
            reasoning_effort=effort,
            fallback_model=os.getenv("AZURE_AI_FALLBACK_MODEL") or None,
        )


def build_user_prompt(notes: Sequence[str], capacity_kwh: float) -> str:
    listing = "\n".join(f"{index}: {note}" for index, note in enumerate(notes))
    return (
        f"Battery capacity: {capacity_kwh:g} kWh "
        "(use this to convert any percentage or fractional reserve into kWh).\n"
        f"Interpret these {len(notes)} operator note(s) and return "
        f"{len(notes)} interpretation(s):\n{listing}"
    )


def _strip_nulls(adjustment: Any) -> Any:
    """Collapse the flat union shape the model fills into the exact shape.

    The request schema keeps every adjustment field present so the provider's
    strict structured-output mode stays simple; unused fields come back null and
    are removed here before validation.
    """
    if not isinstance(adjustment, dict):
        return adjustment
    trimmed = {key: value for key, value in adjustment.items() if value is not None}
    return trimmed or None


def _coerce(payload: Any) -> Any:
    """Normalize the raw model payload into a list of interpretation entries."""
    if isinstance(payload, dict):
        entries = payload.get("interpretations", payload.get("directive_interpretation"))
    else:
        entries = payload
    if not isinstance(entries, list):
        raise GuardrailError("model output did not contain an interpretations array")
    coerced = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise GuardrailError("interpretation entry must be an object")
        item = dict(entry)
        # Applied uniformly, including to no_op. Force-nulling a no_op
        # adjustment would erase the evidence that the model filled in real
        # hours or a real cap while still labelling the note irrelevant, and
        # that contradiction is worth a repair retry.
        item["structured_adjustment"] = _strip_nulls(item.get("structured_adjustment"))
        coerced.append(item)
    return coerced


@dataclass
class Interpreter:
    settings: Settings
    _client: AsyncOpenAI | None = field(default=None, init=False, repr=False)
    _transport: str = field(default="", init=False, repr=False)
    _send_temperature: bool = field(default=True, init=False, repr=False)
    _send_reasoning: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
            timeout=self.settings.timeout_seconds,
            max_retries=0,
        )
        self._transport = self.settings.transport
        self._send_reasoning = self.settings.reasoning_effort is not None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def warm_up(self) -> None:
        """Establish DNS, TLS, and connection state before the first real request.

        Without this the first scored request pays the cold-connection cost,
        which measurably dominates p95 over a short test run. Failures are
        ignored: this is an optimization, not a readiness requirement.
        """
        try:
            await self._call(
                [
                    {"role": "system", "content": "Reply with the single word ok."},
                    {"role": "user", "content": "ok"},
                ],
                self.settings.model,
            )
            logger.info("model connection warmed")
        except Exception as error:  # noqa: BLE001 - warmup must never break startup
            logger.warning("model warmup skipped: %s", type(error).__name__)

    async def interpret(
        self, notes: Sequence[str], capacity_kwh: float
    ) -> list[Directive]:
        """Interpret every note in one call, with one bounded repair retry."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(notes, capacity_kwh)},
        ]
        last_error: Exception | None = None

        for attempt in (1, 2):
            try:
                raw = await self._call(messages, self.settings.model)
            except ProviderError as error:
                last_error = error
                if self.settings.fallback_model:
                    logger.warning("primary model failed, trying the fallback model")
                    try:
                        raw = await self._call(messages, self.settings.fallback_model)
                    except ProviderError as fallback_error:
                        raise fallback_error from error
                else:
                    raise
            try:
                entries = _coerce(json.loads(raw))
                return validate_batch(
                    entries,
                    note_count=len(notes),
                    capacity_kwh=capacity_kwh,
                )
            except (json.JSONDecodeError, GuardrailError) as error:
                last_error = error
                if attempt == 2:
                    break
                logger.warning("model output rejected, requesting one repair: %s", error)
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"That response was rejected: {error}. "
                            f"Return only valid JSON with exactly {len(notes)} "
                            "interpretations, note_index 0.."
                            f"{len(notes) - 1} in order, applies=false and all "
                            "adjustment fields null for no_op, and only the "
                            "fields each directive type requires otherwise."
                        ),
                    },
                ]

        raise ProviderError(f"model interpretation failed: {last_error}")

    async def _call(self, messages: list[dict[str, str]], model: str) -> str:
        assert self._client is not None
        if self._transport == "chat":
            return await self._call_chat(messages, model)
        try:
            return await self._call_responses(messages, model)
        except APIStatusError as error:
            if error.status_code in (404, 405) and self.settings.transport == "responses":
                logger.warning("responses transport unavailable, switching to chat")
                self._transport = "chat"
                return await self._call_chat(messages, model)
            raise ProviderError(f"provider returned {error.status_code}") from error
        except ProviderError:
            raise
        except Exception as error:  # noqa: BLE001 - never leak provider internals
            raise ProviderError(f"{type(error).__name__} calling the model") from error

    async def _call_responses(self, messages: list[dict[str, str]], model: str) -> str:
        assert self._client is not None
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "max_output_tokens": self.settings.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "operator_note_interpretation",
                    "schema": OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        }
        if self._send_temperature and self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature
        if self._send_reasoning and self.settings.reasoning_effort:
            payload["reasoning"] = {"effort": self.settings.reasoning_effort}

        try:
            response = await self._client.responses.create(**payload)
        except APIStatusError as error:
            if self._drop_unsupported(error, payload):
                return await self._call_responses(messages, model)
            raise

        text = (getattr(response, "output_text", "") or "").strip()
        if not text:
            raise ProviderError("model returned an empty interpretation")
        return text

    async def _call_chat(self, messages: list[dict[str, str]], model: str) -> str:
        assert self._client is not None
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": self.settings.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "operator_note_interpretation",
                    "schema": OUTPUT_SCHEMA,
                    "strict": True,
                },
            },
        }
        if self._send_temperature and self.settings.temperature is not None:
            payload["temperature"] = self.settings.temperature

        try:
            response = await self._client.chat.completions.create(**payload)
        except APIStatusError as error:
            if self._drop_unsupported(error, payload):
                return await self._call_chat(messages, model)
            raise ProviderError(f"provider returned {error.status_code}") from error
        except Exception as error:  # noqa: BLE001
            raise ProviderError(f"{type(error).__name__} calling the model") from error

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None:
            raise ProviderError("model returned no choices")
        if getattr(choice.message, "refusal", None):
            raise ProviderError("model refused the interpretation request")
        text = (choice.message.content or "").strip()
        if not text:
            raise ProviderError("model returned an empty interpretation")
        return text

    def _drop_unsupported(self, error: APIStatusError, payload: dict[str, Any]) -> bool:
        """Retry once without a sampling parameter the deployment rejects."""
        if error.status_code != 400:
            return False
        detail = str(getattr(error, "message", "") or error).lower()
        if "temperature" in detail and self._send_temperature:
            logger.warning("deployment rejects temperature, retrying without it")
            self._send_temperature = False
            payload.pop("temperature", None)
            return True
        if "reasoning" in detail and self._send_reasoning:
            logger.warning("deployment rejects reasoning effort, retrying without it")
            self._send_reasoning = False
            payload.pop("reasoning", None)
            return True
        return False
