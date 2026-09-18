#!/usr/bin/env python3
"""Run the ten public sample cases in either of two lanes.

``--offline`` injects each case's published directive interpretation and checks
that the optimizer reaches the published optimal cost. It needs no credentials
and isolates optimizer correctness from model behaviour.

``--live`` posts each case to a running service and checks the full pipeline:
directive semantics, schedule validity, reported totals, and latency.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from schemas import OptimizeRequest, check_base_feasibility, validate_batch
from solve import compile_directives, replay, solve

TOLERANCE = 0.01
DEFAULT_CASES = Path(__file__).with_name("public_cases.json")


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def run_offline(cases: list[dict[str, Any]]) -> int:
    failures = 0
    for case in cases:
        request = OptimizeRequest.model_validate(case["input"])
        check_base_feasibility(request)
        expected = case["expected_output"]
        directives = validate_batch(
            expected["directive_interpretation"],
            note_count=len(request.operator_notes),
            capacity_kwh=request.battery.capacity_kwh,
        )
        solution = solve(request, directives)
        reference = expected["total_cost_bdt"]
        delta = solution.totals.total_cost_bdt - reference
        ok = solution.stage == "all directives" and abs(delta) <= TOLERANCE
        failures += 0 if ok else 1
        print(
            f"{'PASS' if ok else 'FAIL'} {case['id']}  "
            f"cost={solution.totals.total_cost_bdt:>10.2f}  "
            f"reference={reference:>10.2f}  delta={delta:+.4f}  "
            f"grid={solution.totals.total_grid_kwh:.2f}  "
            f"peak={solution.totals.peak_grid_kwh:.2f}  stage={solution.stage}"
        )
    return failures


def _semantic_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Compare directives by meaning, ignoring free-text explanation wording."""
    adjustment = entry.get("structured_adjustment")
    if adjustment is None:
        payload: tuple[Any, ...] = ()
    else:
        payload = tuple(
            (key, round(value, 4) if isinstance(value, (int, float)) else value)
            for key, value in sorted(adjustment.items())
            if key != "hours"
        )
        payload = (tuple(adjustment.get("hours", ())),) + payload
    return (entry.get("note_index"), entry.get("applies"), entry.get("directive_type"), payload)


def post(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw.decode("utf-8", "replace")
    except json.JSONDecodeError as error:
        return 200, f"response was not valid JSON: {error}"
    except OSError as error:
        # Unreachable service, reset connection, or timeout: report it as a
        # failed case rather than aborting the whole run with a traceback.
        return 0, str(error)


def run_live(cases: list[dict[str, Any]], base_url: str, timeout: float) -> int:
    base_url = base_url.rstrip("/")
    failures = 0
    latencies: list[float] = []

    status, health = post_health(base_url, timeout)
    print(f"GET /health -> {status} {health}")
    if status != 200 or health != {"status": "ok"}:
        print("FAIL health endpoint did not report readiness")
        failures += 1

    for case in cases:
        started = time.perf_counter()
        status, body = post(f"{base_url}/optimize-energy", case["input"], timeout)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        problems: list[str] = []
        if status != 200:
            problems.append(f"http {status}: {body}")
        else:
            problems.extend(check_live_body(case, body))

        if problems:
            failures += 1
            print(f"FAIL {case['id']}  {elapsed:.2f}s")
            for problem in problems:
                print(f"       - {problem}")
        else:
            expected_cost = case["expected_output"]["total_cost_bdt"]
            delta = body["total_cost_bdt"] - expected_cost
            print(
                f"PASS {case['id']}  {elapsed:.2f}s  "
                f"cost={body['total_cost_bdt']:.2f}  "
                f"reference={expected_cost:.2f}  delta={delta:+.4f}"
            )

    if latencies:
        ordered = sorted(latencies)
        index = max(0, int(round(0.95 * len(ordered))) - 1)
        print(
            f"\nlatency  median={statistics.median(ordered):.2f}s  "
            f"p95={ordered[index]:.2f}s  max={ordered[-1]:.2f}s"
        )
    return failures


def post_health(base_url: str, timeout: float) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")
    except OSError as error:
        return 0, str(error)


def check_live_body(case: dict[str, Any], body: Any) -> list[str]:
    """Validate a live response the way the judge would."""
    problems: list[str] = []
    if not isinstance(body, dict):
        return ["response was not a JSON object"]

    expected = case["expected_output"]
    request = OptimizeRequest.model_validate(case["input"])

    if body.get("scenario_id") != case["input"]["scenario_id"]:
        problems.append("scenario_id was not echoed")

    interpretations = body.get("directive_interpretation")
    if not isinstance(interpretations, list) or len(interpretations) != len(
        request.operator_notes
    ):
        problems.append("wrong number of directive_interpretation entries")
        return problems

    for got, want in zip(interpretations, expected["directive_interpretation"]):
        if _semantic_key(got) != _semantic_key(want):
            problems.append(
                f"note {want['note_index']} expected "
                f"{want['directive_type']} {want['structured_adjustment']} "
                f"but received {got.get('directive_type')} "
                f"{got.get('structured_adjustment')}"
            )
        # Wording is not compared, but the field is required by the schema.
        explanation = got.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            problems.append(
                f"note {want['note_index']} is missing a non-empty explanation"
            )

    # Replay the returned schedule against the organizer ground-truth directives.
    truth = validate_batch(
        expected["directive_interpretation"],
        note_count=len(request.operator_notes),
        capacity_kwh=request.battery.capacity_kwh,
    )
    compiled = compile_directives(request, truth)
    try:
        plan = [PlanRow(entry) for entry in body.get("hourly_plan", [])]
        totals = replay(compiled, plan)
    except Exception as error:  # noqa: BLE001 - report any replay failure verbatim
        problems.append(f"schedule failed replay: {error}")
        return problems

    for field, derived in (
        ("total_grid_kwh", totals.total_grid_kwh),
        ("total_cost_bdt", totals.total_cost_bdt),
        ("peak_grid_kwh", totals.peak_grid_kwh),
    ):
        reported = body.get(field)
        # isfinite is checked explicitly: every comparison against NaN is False,
        # so a NaN total would otherwise slip through as a match.
        if (
            not isinstance(reported, (int, float))
            or isinstance(reported, bool)
            or not math.isfinite(reported)
            or abs(reported - derived) > TOLERANCE
        ):
            problems.append(f"{field} {reported} does not match replay {derived:.4f}")

    if abs(totals.total_cost_bdt - expected["total_cost_bdt"]) > TOLERANCE:
        problems.append(
            f"cost {totals.total_cost_bdt:.2f} is not optimal "
            f"(reference {expected['total_cost_bdt']:.2f})"
        )
    if not isinstance(body.get("plan_summary"), str) or not body["plan_summary"].strip():
        problems.append("plan_summary is missing or empty")
    return problems


class PlanRow:
    """Adapts a raw JSON plan entry to the attribute access replay expects."""

    __slots__ = (
        "hour",
        "grid_kwh",
        "solar_used_kwh",
        "battery_action",
        "battery_kwh",
        "battery_energy_after_kwh",
    )

    def __init__(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise ValueError("hourly_plan entries must be objects")
        for name in self.__slots__:
            if name not in raw:
                raise ValueError(f"hourly_plan entry is missing {name}")
            setattr(self, name, raw[name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--offline",
        action="store_true",
        help="inject published directives and verify optimizer cost (no API key)",
    )
    group.add_argument(
        "--live",
        metavar="BASE_URL",
        help="post every case to a running service, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.live:
        failures = run_live(cases, args.live, args.timeout)
    else:
        failures = run_offline(cases)

    print(f"\n{len(cases) - failures}/{len(cases)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
