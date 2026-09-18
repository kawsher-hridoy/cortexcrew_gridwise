#!/usr/bin/env python3
"""Independent GridWise acceptance suite. Defaults to OFFLINE; never imports app code.

Generate fixtures once: .venv/bin/python tests/run_hard_live.py --generate
Offline verification:   .venv/bin/python tests/run_hard_live.py
Authorized live run:    .venv/bin/python tests/run_hard_live.py --live http://20.220.24.121

Live execution is sequential, has no retries, and permits at most 30 POST attempts.
Each invocation is a new run: do not rerun live without a fresh approved budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path(__file__).with_name("hard_live_cases.json")
TOL = 0.01
FACTOR_TOL = 1e-6  # Dimensionless comparison policy, separate from kWh/BDT tolerance.
TYPES = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "max_grid_window": {"hours", "max_grid_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "no_op": set(),
}
TOP_KEYS = {"scenario_id", "directive_interpretation", "hourly_plan", "total_grid_kwh",
            "total_cost_bdt", "peak_grid_kwh", "plan_summary"}
NOTE_KEYS = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}
ROW_KEYS = {"hour", "grid_kwh", "solar_used_kwh", "battery_action", "battery_kwh",
            "battery_energy_after_kwh"}
METRICS_MARKER = b"\n__GRIDWISE_CURL_METRICS__"


def number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def strict_json(text: str | bytes) -> Any:
    def reject_constant(value):
        raise ValueError("non-finite JSON constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def directive(kind: str, hours=(), **values) -> dict:
    return {"note_index": 0, "applies": kind != "no_op", "directive_type": kind,
            "structured_adjustment": None if kind == "no_op" else {"hours": list(hours), **values},
            "explanation": "Independent expected interpretation for this synthetic test."}


def base_request() -> dict:
    hours = []
    for h in range(24):
        demand = 68 + ((h * 7) % 17) + (18 if 8 <= h <= 16 else 32 if 17 <= h <= 21 else 0)
        solar = max(0, 6 - abs(h - 12)) * 17.5
        tariff = 3.2 if h < 6 else 7.5 if h < 16 else 16.5 if h < 22 else 4.25
        hours.append({"hour": h, "demand_kwh": demand, "solar_kwh": solar,
                      "tariff_bdt_per_kwh": tariff})
    return {"scenario_id": "", "operator_notes": [], "hours": hours,
            "battery": {"capacity_kwh": 240, "initial_energy_kwh": 96,
                        "minimum_energy_kwh": 24, "max_charge_kwh_per_hour": 40,
                        "max_discharge_kwh_per_hour": 45}}


def make_cases() -> list[dict]:
    """Fresh synthetic profiles; no public IDs, schedules, or notes used to generate them."""
    cases = []

    def add(label, notes, directives, request=None):
        request = copy.deepcopy(request or base_request())
        case_id = f"H{len(cases) + 1:02d}"
        request.update(scenario_id=f"HARD-{case_id}", operator_notes=copy.deepcopy(notes))
        directives = copy.deepcopy(directives)
        for i, entry in enumerate(directives):
            entry["note_index"] = i
        cases.append({"id": case_id, "label": label, "input": request,
                      "expected_directives": directives})

    solar = directive("solar_reduction", range(12, 15), factor=0.18)
    add("Reduction-by wording", ["Inverter servicing from noon until 3 PM will reduce usable rooftop solar by 82%."], [solar])
    add("Equivalent remaining-fraction wording", ["During the 12:00 to 15:00 interval, only 18% of forecast rooftop solar remains usable."], [solar])
    add("Complete solar outage", ["No rooftop solar output is available from 09:00 until 13:00 while the inverter is disconnected."],
        [directive("solar_reduction", range(9, 13), factor=0)])
    reserve_note = "Maintain at least 37.5% of the battery's capacity in storage from 17:00 until 21:00."
    add("Percentage reserve, 240 kWh capacity", [reserve_note],
        [directive("minimum_battery_reserve", range(17, 21), minimum_energy_kwh=90)])
    r = base_request(); r["battery"]["capacity_kwh"] = 360
    add("Same percentage, changed capacity", [reserve_note],
        [directive("minimum_battery_reserve", range(17, 21), minimum_energy_kwh=135)], r)
    add("Cheap-hour charging outage", ["The battery charger is isolated from midnight until 06:00; charging is unavailable during that interval."],
        [directive("no_charge_window", range(0, 6))])
    add("Expensive-hour discharge restriction", ["Relay checks prohibit battery discharge from 17:00 until 21:00."],
        [directive("no_discharge_window", range(17, 21))])
    add("Intersecting battery outages", ["Battery charging is disabled from 09:00 until 15:00.",
        "Battery discharging is disabled from 12:00 until 18:00."],
        [directive("no_charge_window", range(9, 15)), directive("no_discharge_window", range(12, 18))])
    r = base_request(); r["hours"][19]["demand_kwh"] = 113
    cap_note = "For the interval from 19:00 until 20:00, grid intake must not exceed 68 kWh."
    cap = directive("max_grid_window", [19], max_grid_kwh=68)
    add("Exactly binding cap and discharge rate", [cap_note], [cap], r)
    r = base_request()
    for h, s in zip(range(12, 15), [100, 70, 25]):
        r["hours"][h].update(demand_kwh=70, solar_kwh=s)
    add("Zero-import interval", ["Disconnect grid intake from 12:00 until 15:00: the grid-import limit is zero kWh in each of those hours."],
        [directive("max_grid_window", range(12, 15), max_grid_kwh=0)], r)
    combined_notes = ["Keep at least 120 kWh stored in the battery from 17:00 until 22:00.",
                      "Between 18:00 and 21:00, grid import is limited to 80 kWh per hour."]
    combined = [directive("minimum_battery_reserve", range(17, 22), minimum_energy_kwh=120),
                directive("max_grid_window", range(18, 21), max_grid_kwh=80)]
    add("Reserve with simultaneous grid cap", combined_notes, combined)
    combined_notes += ["Battery charging is unavailable from 17:00 until 23:00."]
    combined += [directive("no_charge_window", range(17, 23))]
    add("Three constraints requiring advance charging", combined_notes, combined)
    r = base_request(); r["battery"]["initial_energy_kwh"] = 240
    for h in range(10, 16):
        r["hours"][h]["solar_kwh"] = 480 + 10 * h
    add("Forced curtailment with full battery", ["Charging is prohibited from 10:00 until 16:00.",
        "Keep the battery at its full 240 kWh capacity from 10:00 until 16:00."],
        [directive("no_charge_window", range(10, 16)),
         directive("minimum_battery_reserve", range(10, 16), minimum_energy_kwh=240)], r)
    r = base_request(); r["hours"][19]["demand_kwh"] = 113
    for row in r["hours"]:
        row["tariff_bdt_per_kwh"] = 0
    add("Zero tariffs but nontrivial grid constraint", [cap_note], [cap], r)
    r = base_request()
    for row in r["hours"]:
        row["tariff_bdt_per_kwh"] = 5
        row["solar_kwh"] *= 0.5
    add("Flat tariffs and equivalent optima", ["Keep a minimum of 90 kWh in the battery from 17:00 until 21:00."],
        [directive("minimum_battery_reserve", range(17, 21), minimum_energy_kwh=90)], r)
    r = base_request(); r["hours"][23]["tariff_bdt_per_kwh"] = 48
    add("Terminal neutrality under expensive final hour", ["Do not charge the battery from 20:00 until midnight at the end of this planning day."],
        [directive("no_charge_window", range(20, 24))], r)
    r = base_request(); r["battery"].update(initial_energy_kwh=240, max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0)
    add("Zero rates and full-capacity reserve", ["Keep 240 kWh stored in the battery throughout all 24 hours of this scenario."],
        [directive("minimum_battery_reserve", range(24), minimum_energy_kwh=240)], r)
    r = base_request(); r["battery"]["initial_energy_kwh"] = 24
    add("Initial reserve boundary and early reserve increase", ["At least 168 kWh must remain stored from 04:00 until 07:00."],
        [directive("minimum_battery_reserve", range(4, 7), minimum_energy_kwh=168)], r)
    r = base_request(); scale = 1.8058782404676035
    for row in r["hours"]:
        row["demand_kwh"] *= scale; row["solar_kwh"] *= scale
    for key in r["battery"]:
        r["battery"][key] *= scale
    reserve, cap_value = 120 * scale, 80 * scale
    add("Fractional three-constraint numerical regression", [
        f"Keep at least {reserve:.12f} kWh in the battery from 17:00 until 22:00.",
        f"Grid intake must stay at or below {cap_value:.12f} kWh per hour from 18:00 until 21:00.",
        "Charging is unavailable from 17:00 until 23:00."],
        [directive("minimum_battery_reserve", range(17, 22), minimum_energy_kwh=reserve),
         directive("max_grid_window", range(18, 21), max_grid_kwh=cap_value),
         directive("no_charge_window", range(17, 23))], r)
    r = base_request(); random.Random(20260918).shuffle(r["hours"])
    add("Shuffled hours, three notes, distractor", [
        "The athletics office is revising next month's volunteer roster.",
        "Only 35% of forecast solar remains usable from 11:00 until 14:00.",
        "Do not discharge the battery from 22:00 until 24:00."],
        [directive("no_op"), directive("solar_reduction", range(11, 14), factor=0.35),
         directive("no_discharge_window", range(22, 24))], r)
    return cases


def fixture_guard(case: dict) -> None:
    r = case["input"]; b = r["battery"]; ds = case["expected_directives"]
    if len(r["hours"]) != 24 or sorted(h["hour"] for h in r["hours"]) != list(range(24)):
        raise ValueError("fixture has invalid hour coverage")
    if not 1 <= len(r["operator_notes"]) <= 3 or len(ds) != len(r["operator_notes"]):
        raise ValueError("fixture has invalid note coverage")
    if any(not isinstance(n, str) or not n.strip() for n in r["operator_notes"]):
        raise ValueError("fixture has an empty note")
    if any(not number(v) or v < 0 for v in b.values()):
        raise ValueError("fixture has invalid battery numbers")
    if not 0 <= b["minimum_energy_kwh"] <= b["initial_energy_kwh"] <= b["capacity_kwh"] or b["capacity_kwh"] <= 0:
        raise ValueError("fixture has invalid battery bounds")
    for row in r["hours"]:
        if type(row["hour"]) is not int or any(not number(row[k]) or row[k] < 0 for k in ("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")):
            raise ValueError("fixture has invalid hourly numbers")
    for i, d in enumerate(ds):
        if d["note_index"] != i or d["directive_type"] not in TYPES or d["applies"] != (d["directive_type"] != "no_op"):
            raise ValueError("fixture has invalid interpretation")
        adj = d["structured_adjustment"]
        if d["directive_type"] == "no_op":
            if adj is not None: raise ValueError("fixture no_op is not null")
            continue
        if set(adj) != TYPES[d["directive_type"]] or not adj["hours"] or any(type(h) is not int or not 0 <= h < 24 for h in adj["hours"]) or adj["hours"] != sorted(set(adj["hours"])):
            raise ValueError("fixture adjustment has invalid shape/hours")
        for key in set(adj) - {"hours"}:
            if not number(adj[key]) or adj[key] < 0: raise ValueError("fixture has invalid adjustment number")
        if adj.get("factor", 0) > 1 or adj.get("minimum_energy_kwh", 0) > b["capacity_kwh"]:
            raise ValueError("fixture adjustment exceeds bounds")


def oracle(case: dict) -> dict:
    """Independent 120-variable LP: G,S,C,D,E; no production imports.

    C and D may overlap in the relaxation. With unit efficiency, canceling their
    common amount preserves ALL constraints and the objective. The emitted
    witness is normalized to one direction and separately replayed below.
    """
    fixture_guard(case)
    r = case["input"]; b = r["battery"]
    ordered = sorted(r["hours"], key=lambda x: x["hour"])
    bounds = []
    for row in ordered:
        bounds.extend([(0, None), (0, row["solar_kwh"]), (0, b["max_charge_kwh_per_hour"]),
                       (0, b["max_discharge_kwh_per_hour"]), (b["minimum_energy_kwh"], b["capacity_kwh"])])
    for d in case["expected_directives"]:
        a = d["structured_adjustment"]
        if not d["applies"]: continue
        for h in a["hours"]:
            i = 5 * h; kind = d["directive_type"]
            if kind == "solar_reduction": bounds[i+1] = (0, ordered[h]["solar_kwh"] * a["factor"])
            elif kind == "minimum_battery_reserve": bounds[i+4] = (max(bounds[i+4][0], a["minimum_energy_kwh"]), b["capacity_kwh"])
            elif kind == "max_grid_window": bounds[i] = (0, a["max_grid_kwh"] if bounds[i][1] is None else min(bounds[i][1], a["max_grid_kwh"]))
            elif kind == "no_charge_window": bounds[i+2] = (0, 0)
            elif kind == "no_discharge_window": bounds[i+3] = (0, 0)
    objective = np.zeros(120); equations = np.zeros((49, 120)); rhs = np.zeros(49)
    for h, row in enumerate(ordered):
        i = 5 * h; objective[i] = row["tariff_bdt_per_kwh"]
        equations[2*h, i:i+4] = [1, 1, -1, 1]; rhs[2*h] = row["demand_kwh"]
        equations[2*h+1, i+2:i+5] = [-1, 1, 1]
        if h: equations[2*h+1, i-1] = -1
        else: rhs[2*h+1] = b["initial_energy_kwh"]
    equations[48, 119] = 1; rhs[48] = b["initial_energy_kwh"]
    result = linprog(objective, A_eq=equations, b_eq=rhs, bounds=bounds, method="highs",
                     options={"time_limit": 10.0})
    if not result.success:
        raise ValueError(f"{case['id']}: independent oracle failed, status {result.status}")
    plan = []; energy = b["initial_energy_kwh"]
    for h in range(24):
        g, s, c, d, _ = [float(x) for x in result.x[5*h:5*h+5]]
        net = c - d
        if abs(net) < 1e-9: net = 0.0
        energy += net
        plan.append({"hour": h, "grid_kwh": max(0.0, g), "solar_used_kwh": max(0.0, s),
                     "battery_action": "charge" if net > 0 else "discharge" if net < 0 else "idle",
                     "battery_kwh": abs(net), "battery_energy_after_kwh": energy})
    totals = derived_totals(r, plan)
    if abs(totals["total_cost_bdt"] - float(result.fun)) > 1e-5:
        raise ValueError("oracle witness did not preserve the LP objective")
    return {"scenario_id": r["scenario_id"], "directive_interpretation": copy.deepcopy(case["expected_directives"]),
            "hourly_plan": plan, **totals, "plan_summary": "Independent optimal, lossless, end-neutral witness."}


def derived_totals(request: dict, plan: list) -> dict:
    prices = {h["hour"]: h["tariff_bdt_per_kwh"] for h in request["hours"]}
    return {"total_grid_kwh": math.fsum(h["grid_kwh"] for h in plan),
            "total_cost_bdt": math.fsum(h["grid_kwh"] * prices[h["hour"]] for h in plan),
            "peak_grid_kwh": max(h["grid_kwh"] for h in plan)}


def check_response(case: dict, body: Any) -> dict:
    errors = {k: [] for k in ("schema", "interpretation", "schedule", "totals", "optimization")}
    out = {"errors": errors, "derived_totals": None}

    def shape(obj, keys, label):
        if not isinstance(obj, dict) or set(obj) != keys:
            errors["schema"].append(f"{label}: incorrect object shape")
            return False
        return True

    if not shape(body, TOP_KEYS, "response"):
        return out
    request = case["input"]; battery = request["battery"]
    if type(body["scenario_id"]) is not str or body["scenario_id"] != request["scenario_id"]:
        errors["schema"].append("scenario_id was not echoed")
    if not isinstance(body["plan_summary"], str) or not body["plan_summary"].strip():
        errors["schema"].append("plan_summary missing or empty")
    interpretations = body["directive_interpretation"]
    if not isinstance(interpretations, list) or len(interpretations) != len(case["expected_directives"]):
        errors["schema"].append("incorrect interpretation coverage")
    else:
        for i, (got, expected) in enumerate(zip(interpretations, case["expected_directives"])):
            if not shape(got, NOTE_KEYS, f"note {i}"): continue
            if not isinstance(got["explanation"], str) or not got["explanation"].strip():
                errors["schema"].append(f"note {i}: missing explanation")
            if type(got["note_index"]) is not int or got["note_index"] != i:
                errors["schema"].append(f"note {i}: invalid index/order")
            kind = got["directive_type"]
            if type(kind) is not str or kind not in TYPES:
                errors["schema"].append(f"note {i}: unsupported directive type"); continue
            if type(got["applies"]) is not bool or got["applies"] != (kind != "no_op"):
                errors["schema"].append(f"note {i}: invalid applies semantics")
            a = got["structured_adjustment"]; want = expected["structured_adjustment"]
            if kind != expected["directive_type"] or got["applies"] != expected["applies"]:
                errors["interpretation"].append(f"note {i}: wrong relevance/type")
            if kind == "no_op":
                if a is not None: errors["schema"].append(f"note {i}: no_op adjustment is not null")
                continue
            if not shape(a, TYPES[kind], f"note {i} adjustment"): continue
            hours = a["hours"]
            if not isinstance(hours, list) or not hours or any(type(h) is not int or not 0 <= h < 24 for h in hours) or hours != sorted(set(hours)):
                errors["schema"].append(f"note {i}: invalid hours")
            if want is not None and kind == expected["directive_type"] and hours != want["hours"]:
                errors["interpretation"].append(f"note {i}: incorrect affected hours")
            for key in TYPES[kind] - {"hours"}:
                value = a[key]
                if not number(value) or value < 0 or (key == "factor" and value > 1) or (key == "minimum_energy_kwh" and value > battery["capacity_kwh"]):
                    errors["schema"].append(f"note {i}: invalid {key}")
                elif want is not None and kind == expected["directive_type"] and abs(value - want[key]) > (FACTOR_TOL if key == "factor" else TOL):
                    errors["interpretation"].append(f"note {i}: incorrect {key}")
    plan = body["hourly_plan"]
    if not isinstance(plan, list) or len(plan) != 24:
        errors["schema"].append("hourly_plan must have 24 entries"); return out
    usable = True
    for row in plan:
        if not shape(row, ROW_KEYS, "hourly row"):
            usable = False; continue
        if type(row["hour"]) is not int or not 0 <= row["hour"] < 24:
            errors["schema"].append("invalid hour type/range"); usable = False
        for key in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            if not number(row[key]) or row[key] < 0:
                errors["schema"].append(f"invalid non-negative finite {key}"); usable = False
        if type(row["battery_action"]) is not str or row["battery_action"] not in ("charge", "discharge", "idle"):
            errors["schema"].append("invalid battery_action"); usable = False
    if not usable: return out
    if sorted(row["hour"] for row in plan) != list(range(24)):
        errors["schema"].append("missing or duplicate plan hours"); return out
    # This replay evaluates every EXPECTED directive directly; no shared compiler.
    source = {row["hour"]: row for row in request["hours"]}
    energy = battery["initial_energy_kwh"]
    for row in sorted(plan, key=lambda h: h["hour"]):
        h = row["hour"]; original = source[h]
        charge = row["battery_kwh"] if row["battery_action"] == "charge" else 0
        discharge = row["battery_kwh"] if row["battery_action"] == "discharge" else 0
        solar_limit = original["solar_kwh"]; reserve = battery["minimum_energy_kwh"]
        charge_limit = battery["max_charge_kwh_per_hour"]; discharge_limit = battery["max_discharge_kwh_per_hour"]
        caps = []
        for d in case["expected_directives"]:
            a = d["structured_adjustment"]
            if not d["applies"] or h not in a["hours"]: continue
            kind = d["directive_type"]
            if kind == "solar_reduction": solar_limit = original["solar_kwh"] * a["factor"]
            elif kind == "minimum_battery_reserve": reserve = max(reserve, a["minimum_energy_kwh"])
            elif kind == "max_grid_window": caps.append(a["max_grid_kwh"])
            elif kind == "no_charge_window": charge_limit = 0
            elif kind == "no_discharge_window": discharge_limit = 0
        energy += charge - discharge
        violations = []
        if row["battery_action"] == "idle" and row["battery_kwh"] != 0: violations.append("idle magnitude is not zero")
        if charge > charge_limit + TOL: violations.append("charge limit/window")
        if discharge > discharge_limit + TOL: violations.append("discharge limit/window")
        if row["solar_used_kwh"] > solar_limit + TOL: violations.append("effective solar exceeded")
        if any(row["grid_kwh"] > cap + TOL for cap in caps): violations.append("grid cap exceeded")
        if abs(row["grid_kwh"] + row["solar_used_kwh"] + discharge - original["demand_kwh"] - charge) > TOL: violations.append("energy balance")
        if abs(row["battery_energy_after_kwh"] - energy) > TOL: violations.append("battery transition")
        if energy < reserve - TOL: violations.append("battery reserve")
        if energy > battery["capacity_kwh"] + TOL: violations.append("battery capacity")
        errors["schedule"].extend(f"hour {h}: {v}" for v in violations)
    if abs(energy - battery["initial_energy_kwh"]) > TOL:
        errors["schedule"].append("terminal neutrality")
    try:
        totals = derived_totals(request, plan)
    except (OverflowError, ValueError):
        errors["schema"].append("numeric aggregation overflow"); return out
    if not all(number(value) for value in totals.values()):
        errors["schema"].append("non-finite derived totals"); return out
    out["derived_totals"] = totals
    for key, computed in totals.items():
        if not number(body[key]) or body[key] < 0:
            errors["schema"].append(f"{key}: invalid finite non-negative number")
        elif abs(body[key] - computed) > TOL:
            errors["totals"].append(f"{key}: differs from independently replayed total")
    reference = case.get("expected_output")
    if reference and abs(totals["total_cost_bdt"] - reference["total_cost_bdt"]) > TOL:
        errors["optimization"].append(f"cost differs from independent optimum by {totals['total_cost_bdt'] - reference['total_cost_bdt']:.6f} BDT")
    return out


def passed(check: dict) -> bool:
    return not any(check["errors"].values())


def malformed_cases(cases: list) -> list[dict]:
    base = cases[0]["input"]
    missing = copy.deepcopy(base); del missing["battery"]
    duplicate = copy.deepcopy(base); duplicate["hours"][-1]["hour"] = 22
    empty = copy.deepcopy(base); empty["operator_notes"] = []
    invalid = copy.deepcopy(base); invalid["battery"]["initial_energy_kwh"] = invalid["battery"]["capacity_kwh"] + 1
    return [{"id": "M01", "label": "Malformed JSON", "raw_body": '{"scenario_id":', "expected_statuses": [400]},
            *[{"id": f"M{i:02d}", "label": label, "input": data, "expected_statuses": statuses}
              for i, label, data, statuses in [(2, "Missing battery", missing, [400]),
                  (3, "Duplicate hour", duplicate, [400]), (4, "Empty notes", empty, [400]),
                  (5, "Initial energy exceeds capacity", invalid, [400, 422])]]]


def generate(path: Path) -> None:
    if path.exists(): raise FileExistsError("Refusing to overwrite existing fixtures")
    cases = make_cases()
    for case in cases:
        case["expected_output"] = oracle(case)
        result = check_response(case, case["expected_output"])
        if not passed(result): raise ValueError(f"invalid reference {case['id']}: {result}")
    payload = {"_meta": {"title": "Independent GridWise hard live suite", "version": 1,
                 "profile_seed": 20260918, "energy_tolerance": TOL, "factor_tolerance": FACTOR_TOL,
                 "oracle": "Independent 120-variable G,S,C,D,E LP with cancellation witness",
                 "note": "Synthetic test fixtures only. No production special cases."},
               "cases": cases, "repeat_case_ids": ["H01", "H05", "H12", "H19", "H20"],
               "malformed_cases": malformed_cases(cases)}
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Generated and replayed 20 independent references: {path}")


def offline_checks(pack: dict) -> dict:
    cases = pack["cases"]; by_id = {c["id"]: c for c in cases}
    if set(by_id) != {f"H{i:02d}" for i in range(1, 21)} or len(cases) != 20:
        raise ValueError("Expected exactly H01-H20")
    for case in cases:
        current = oracle(case)
        if abs(current["total_cost_bdt"] - case["expected_output"]["total_cost_bdt"]) > 1e-5:
            raise ValueError(f"stale or incorrect reference cost for {case['id']}")
        for response in (current, case["expected_output"]):
            result = check_response(case, response)
            if not passed(result): raise ValueError(f"reference failed replay: {case['id']} {result}")
        print(f"OFFLINE PASS {case['id']} optimum={current['total_cost_bdt']:.6f}")
    if abs(by_id["H01"]["expected_output"]["total_cost_bdt"] - by_id["H02"]["expected_output"]["total_cost_bdt"]) > TOL:
        raise ValueError("equivalent paraphrases changed the reference problem")
    reference = by_id["H09"]["expected_output"]["hourly_plan"][19]
    if abs(reference["grid_kwh"] - 68) > TOL or abs(reference["battery_kwh"] - 45) > TOL:
        raise ValueError("H09 does not force both binding limits")
    if by_id["H14"]["expected_output"]["total_cost_bdt"] != 0:
        raise ValueError("zero-tariff oracle is incorrect")
    # Analytic calibration independent of any fixture cost: no storage -> exact bill.
    analytic = copy.deepcopy(by_id["H17"])
    for row in analytic["input"]["hours"]:
        row.update(demand_kwh=10, solar_kwh=0, tariff_bdt_per_kwh=2)
    if abs(oracle(analytic)["total_cost_bdt"] - 480) > 1e-6:
        raise ValueError("oracle failed analytic no-storage calibration")
    arb = {"id": "ANALYTIC", "input": base_request(), "expected_directives": [directive("no_op")]}
    arb["input"]["operator_notes"] = ["A non-energy notice."]
    arb["input"]["battery"].update(capacity_kwh=1, initial_energy_kwh=0, minimum_energy_kwh=0, max_charge_kwh_per_hour=1, max_discharge_kwh_per_hour=1)
    for row in arb["input"]["hours"]:
        row.update(demand_kwh=1 if row["hour"] == 1 else 0, solar_kwh=0,
                   tariff_bdt_per_kwh=1 if row["hour"] == 0 else 10)
    if abs(oracle(arb)["total_cost_bdt"] - 1) > 1e-6:
        raise ValueError("oracle failed analytic one-unit arbitrage calibration")
    mutations = [
        ("missing explanation", lambda x: x["directive_interpretation"][0].pop("explanation")),
        ("NaN total", lambda x: x.update(total_cost_bdt=float("nan"))),
        ("infinite grid", lambda x: x["hourly_plan"][0].update(grid_kwh=float("inf"))),
        ("missing hour", lambda x: x["hourly_plan"].pop()),
        ("duplicate hour", lambda x: x["hourly_plan"][-1].update(hour=22)),
        ("boolean hour", lambda x: x["hourly_plan"][0].update(hour=False)),
        ("boolean energy", lambda x: x["hourly_plan"][0].update(grid_kwh=True)),
        ("wrong index", lambda x: x["directive_interpretation"][0].update(note_index=1)),
        ("wrong applies", lambda x: x["directive_interpretation"][0].update(applies=False)),
        ("unexpected adjustment", lambda x: x["directive_interpretation"][0]["structured_adjustment"].update(factor=0.5)),
        ("object directive type", lambda x: x["directive_interpretation"][0].update(directive_type={})),
        ("shifted hours", lambda x: x["directive_interpretation"][0]["structured_adjustment"].update(hours=[1, 2])),
        ("grid cap", lambda x: x["hourly_plan"][19].update(grid_kwh=1000)),
        ("battery transition", lambda x: x["hourly_plan"][8].update(battery_energy_after_kwh=0)),
        ("idle magnitude", lambda x: x["hourly_plan"][0].update(battery_action="idle", battery_kwh=1)),
        ("terminal energy", lambda x: x["hourly_plan"][-1].update(battery_energy_after_kwh=0)),
        ("solar excess", lambda x: x["hourly_plan"][0].update(solar_used_kwh=10)),
        ("wrong total", lambda x: x.update(total_grid_kwh=x["total_grid_kwh"] + 1)),
    ]
    for name, mutate in mutations:
        body = copy.deepcopy(by_id["H12"]["expected_output"]); mutate(body)
        if passed(check_response(by_id["H12"], body)):
            raise ValueError(f"checker accepted deliberate corruption: {name}")
    for malformed in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
        try: strict_json(malformed)
        except ValueError: pass
        else: raise ValueError("strict JSON decoder accepted invalid JSON")
    # An idle schedule is another optimum for H15: no solar surplus, flat tariff.
    flat = by_id["H15"]; alternate = copy.deepcopy(flat["expected_output"])
    alternate["hourly_plan"] = [{"hour": h["hour"], "grid_kwh": h["demand_kwh"] - h["solar_kwh"],
        "solar_used_kwh": h["solar_kwh"], "battery_action": "idle", "battery_kwh": 0,
        "battery_energy_after_kwh": flat["input"]["battery"]["initial_energy_kwh"]} for h in flat["input"]["hours"]]
    alternate.update(derived_totals(flat["input"], alternate["hourly_plan"]))
    if not passed(check_response(flat, alternate)): raise ValueError("checker rejected an equivalent optimum")
    # Validate the organizer's ten reference schedules with this separate checker.
    public_path = ROOT / "BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    public_count = 0
    if public_path.exists():
        for case in strict_json(public_path.read_bytes())["cases"]:
            public = {**case, "expected_directives": case["expected_output"]["directive_interpretation"]}
            if not passed(check_response(public, public["expected_output"])):
                raise ValueError(f"checker rejected official reference {case['id']}")
            public_count += 1
    result = {"hard_references_passed": len(cases), "oracle_analytic_calibrations": 2,
              "corruption_tests_rejected": len(mutations), "invalid_json_tests_rejected": 3,
              "equivalent_optimum_accepted": True, "official_reference_replays_passed": public_count,
              "live_requests_sent": 0}
    print("OFFLINE SELF-CHECKS", json.dumps(result))
    return result


SENSITIVE_KEY = re.compile(r"^(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|cookie|set-cookie)$", re.I)
TOKEN_PATTERN = re.compile(r"(?i)(bearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|(?:api[_-]?key|password|secret)\s*[=:]\s*[^\s,;]+)")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict): return {k: "[REDACTED]" if SENSITIVE_KEY.match(k) else sanitize(v) for k, v in value.items()}
    if isinstance(value, list): return [sanitize(v) for v in value]
    if isinstance(value, str):
        if "Traceback (most recent call last)" in value or re.search(r'File "[^\n]+", line \d+', value):
            return "[REDACTED STACK TRACE]"
        return TOKEN_PATTERN.sub("[REDACTED TOKEN]", value)
    if isinstance(value, float) and not math.isfinite(value): return "[NON-FINITE NUMBER]"
    return value


def curl_call(url: str, payload: bytes | None, timeout: float) -> dict:
    """curl enforces a wall-clock timeout; no redirects, no retries, no key required."""
    cmd = ["curl", "--silent", "--show-error", "--connect-timeout", "5", "--max-time", str(timeout),
           "--retry", "0", "--max-filesize", "524288", "--write-out", METRICS_MARKER.decode() + "%{json}"]
    if payload is not None:
        cmd += ["--request", "POST", "--header", "Content-Type: application/json", "--data-binary", "@-"]
    cmd += [url]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=timeout + 3)
        elapsed = time.perf_counter() - started
        data, marker, suffix = proc.stdout.rpartition(METRICS_MARKER)
        metrics = strict_json(suffix) if marker else {}
        raw = data if marker else proc.stdout
        body = None; parse_error = None
        try: body = strict_json(raw)
        except (ValueError, UnicodeDecodeError): parse_error = "response is not strict JSON"
        status = int(metrics.get("http_code", 0))
        leak = bool(TOKEN_PATTERN.search(raw.decode("utf-8", "replace"))) or b"Traceback (most recent call last)" in raw
        return {"status": status, "elapsed_seconds": elapsed, "curl_seconds": metrics.get("time_total"),
                "content_type": metrics.get("content_type", ""), "curl_exit_code": proc.returncode,
                "timed_out": proc.returncode == 28, "transport_error": None if proc.returncode == 0 else f"curl exit {proc.returncode}",
                "json_error": parse_error, "body": body, "response_sha256": hashlib.sha256(raw).hexdigest(),
                "unsafe_error_content": leak}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        return {"status": 0, "elapsed_seconds": time.perf_counter() - started, "curl_seconds": None,
                "content_type": "", "curl_exit_code": None, "timed_out": isinstance(exc, subprocess.TimeoutExpired),
                "transport_error": type(exc).__name__, "json_error": "response unavailable", "body": None,
                "unsafe_error_content": False}


def trial_list(pack: dict) -> list[dict]:
    by_id = {c["id"]: c for c in pack["cases"]}
    return ([{"id": c["id"], "suite": "hard", "case": c} for c in pack["cases"]]
            + [{"id": "R-" + i, "suite": "repeat", "case": by_id[i]} for i in pack["repeat_case_ids"]]
            + [{"id": c["id"], "suite": "malformed", "case": c} for c in pack["malformed_cases"]])


def availability_failure(response: dict) -> bool:
    return bool(response["transport_error"] or response["status"] == 429 or response["status"] >= 500)


def transport_ok(response: dict) -> bool:
    return not response["transport_error"] and not response["json_error"] and str(response["content_type"]).lower().startswith("application/json")


def latency_summary(records: list) -> dict:
    values = sorted(r["response"]["elapsed_seconds"] for r in records)
    if not values: return {"count": 0, "p50_seconds": None, "p95_seconds": None, "max_seconds": None, "failures": 0}
    return {"count": len(values), "p50_seconds": statistics.median(values),
            "p95_seconds": values[math.ceil(0.95 * len(values)) - 1], "max_seconds": max(values),
            "failures": sum(not r["passed"] for r in records),
            "timeouts": sum(r["response"]["timed_out"] for r in records)}


def write_report(folder: Path, metadata: dict, records: list, trials: list, health: list, stopped: str | None) -> dict:
    valid = [r for r in records if r["suite"] != "malformed"]
    metrics = {name: latency_summary([r for r in records if r["suite"] == name]) for name in ("hard", "repeat", "malformed")}
    metrics["valid_combined"] = latency_summary(valid)
    health_ok = len(health) == 2 and all(h["passed"] for h in health)
    functional = len(records) == len(trials) and all(r["passed"] for r in records) and health_ok
    performance = (len(valid) == 25 and not any(r["response"]["timed_out"] for r in valid)
                   and metrics["valid_combined"]["p95_seconds"] <= 5
                   and metrics["valid_combined"]["max_seconds"] <= 30)
    summary = {"all_functional_tests_passed": functional, "performance_target_met": performance,
               "checks_complete": len(records) == len(trials) and len(health) == 2,
               "posts_attempted": len(records), "posts_planned": len(trials), "stopped_reason": stopped,
               "health_passed": health_ok, "metrics": metrics,
               "valid_interpretations_passed": sum(r["category_passes"]["interpretation"] for r in valid),
               "valid_schedules_passed": sum(r["category_passes"]["schedule"] for r in valid),
               "valid_optima_passed": sum(r["category_passes"]["optimization"] for r in valid),
               "limits": "Only this synthetic suite was tested; no hidden-score, cold-start, image-identity, or long-term reliability guarantee."}
    save_json(folder / "summary.json", summary)
    lines = ["# GridWise Hard Live-Test Report", "", f"- Started: {metadata['started_at']}",
             f"- Target: `{metadata['base_url']}`", f"- Source commit: `{metadata['source_commit']}`",
             "- Deployed image/source identity: not independently established by these API tests.",
             f"- POST attempts: {len(records)}/{len(trials)} (budget {metadata['max_posts']}; no retries).",
             f"- All functional tests passed: **{functional}**", f"- Performance target met: **{performance}**",
             f"- Checks complete: **{summary['checks_complete']}**", f"- Stop reason: {stopped or 'none'}", "",
             "## Results", "", "| Case | Suite | HTTP | Seconds | Result |", "|---|---|---:|---:|---|"]
    done = {r["id"]: r for r in records}
    for trial in trials:
        r = done.get(trial["id"])
        lines.append(f"| {trial['id']} | {trial['suite']} | {r['response']['status'] if r else '-'} | "
                     f"{format(r['response']['elapsed_seconds'], '.3f') if r else '-'} | {'PASS' if r and r['passed'] else 'FAIL' if r else 'NOT RUN'} |")
    lines += ["", "## Latency", "", "Nearest-rank p95; all attempted valid requests are included. Timeout durations are censored observations, not successful response latencies. Malformed timings are separate.", "",
              "| Suite | Count | p50 s | p95 s | Max s | Failures |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in metrics.items():
        fmt = lambda x: "-" if x is None else f"{x:.3f}"
        lines.append(f"| {name} | {m['count']} | {fmt(m['p50_seconds'])} | {fmt(m['p95_seconds'])} | {fmt(m['max_seconds'])} | {m['failures']} |")
    lines += ["", "## Failures and reproductions", ""]
    for r in records:
        if r["passed"]: continue
        lines += [f"### {r['id']}: {r['label']}", f"Evidence: `{r['id']}.json`; exact request: `{r['id']}.request.json`."]
        for error in r["problems"]: lines.append(f"- {error}")
        lines.append(f"Reproduce only with fresh budget approval: `curl --max-time 30 -H 'Content-Type: application/json' --data-binary @{r['id']}.request.json {metadata['base_url']}/optimize-energy`")
        lines.append("")
    if all(r["passed"] for r in records): lines.append("No completed POST test failed." if records else "No POST tests executed.")
    lines += ["", "## Scope and limitations", "", "- Oracle and replay are independent of application modules. Oracle uses installed SciPy/HiGHS with an independently constructed 120-variable model.",
              "- The 20 novel fixtures were feasible and their reference plans replayed before any live POST.",
              "- Exact notes/types/hours are checked; kWh/BDT tolerance is 0.01. Dimensionless solar-factor comparison uses 1e-6.",
              "- Equal-cost alternative schedules are accepted. No production or deployment changes were made.",
              "- A small sequential sample does not prove a hidden score, concurrency reliability, or 60-second cold-start readiness.",
              "- Source, fixture, and runner hashes are in metadata.json. Each attempt was journaled before network dispatch.",
              "- Rerunning the live command spends a NEW request budget; failed requests were not retried."]
    (folder / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def run_live(pack: dict, base_url: str, max_posts: int, timeout: float, offline: dict, case_path: Path) -> int:
    if not shutil.which("curl"): raise ValueError("curl is required for wall-clock-bounded HTTP tests")
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("--live requires a base URL without credentials, endpoint path, query, or fragment")
    base_url = base_url.rstrip("/")
    trials = trial_list(pack)
    if len(trials) != 30: raise ValueError("suite must contain exactly 30 planned attempts")
    if not 1 <= max_posts <= 30 or not 0 < timeout <= 30: raise ValueError("limits: 1-30 POSTs and timeout <=30 seconds")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = ROOT / "reports" / "hard-live" / stamp; folder.mkdir(parents=True, exist_ok=False)
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    metadata = {"started_at": datetime.now().astimezone().isoformat(), "base_url": base_url,
                "source_commit": git("rev-parse", "HEAD"), "working_tree": git("status", "--short"),
                "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "max_posts": max_posts, "timeout_seconds": timeout, "concurrency": 1,
                "command": sys.argv, "offline_checks": offline}
    save_json(folder / "metadata.json", metadata)
    records = []; health = []; stopped = None; consecutive = 0
    def check_health(label):
        response = curl_call(base_url + "/health", None, 10)
        item = {"label": label, "passed": transport_ok(response) and response["status"] == 200 and response["body"] == {"status": "ok"}, "response": sanitize(response)}
        health.append(item); save_json(folder / "health.json", health)
        print(f"HEALTH {label}: {'PASS' if item['passed'] else 'FAIL'} HTTP {response['status']}", flush=True)
        return item["passed"]
    if not check_health("before"):
        stopped = "Initial health check failed; no POSTs sent."
    else:
        with (folder / "attempts.jsonl").open("x") as journal:
            for trial in trials:
                if len(records) >= max_posts:
                    stopped = "Approved POST budget exhausted."; break
                case = trial["case"]
                payload = case["raw_body"].encode() if "raw_body" in case else json_bytes(case["input"])
                (folder / f"{trial['id']}.request.json").write_bytes(payload)
                journal.write(json.dumps({"event": "request_started", "attempt": len(records) + 1, "id": trial["id"], "at": datetime.now(timezone.utc).isoformat()}) + "\n"); journal.flush()
                response = curl_call(base_url + "/optimize-energy", payload, timeout)
                problems = []; categories = {k: False for k in ("interpretation", "schedule", "optimization")}
                check = None
                if not transport_ok(response): problems.append(response["transport_error"] or response["json_error"] or "Content-Type is not application/json")
                if response["unsafe_error_content"]: problems.append("Potential sensitive token or traceback in response (redacted).")
                if trial["suite"] == "malformed":
                    if response["status"] not in case["expected_statuses"]:
                        problems.append(f"Expected HTTP {case['expected_statuses']}; got {response['status']}")
                    if not isinstance(response["body"], dict): problems.append("Expected a JSON error object")
                else:
                    if response["status"] != 200: problems.append(f"Expected HTTP 200; got {response['status']}")
                    if response["status"] == 200 and not response["json_error"]:
                        check = check_response(case, response["body"])
                        for category, errors in check["errors"].items():
                            problems.extend(f"{category}: {e}" for e in errors)
                        schema_ok = not check["errors"]["schema"] and transport_ok(response)
                        categories["interpretation"] = schema_ok and not check["errors"]["interpretation"]
                        categories["schedule"] = schema_ok and not check["errors"]["schedule"]
                        categories["optimization"] = categories["schedule"] and not check["errors"]["optimization"]
                    if response["elapsed_seconds"] > 30: problems.append("Client-observed request exceeded 30 seconds")
                record = {"id": trial["id"], "suite": trial["suite"], "label": case["label"],
                          "passed": not problems, "problems": problems, "category_passes": categories,
                          "response": sanitize(response), "check": sanitize(check),
                          "expected": case.get("expected_output", {"http_statuses": case.get("expected_statuses")})}
                save_json(folder / f"{trial['id']}.json", record); records.append(record)
                journal.write(json.dumps({"event": "request_completed", "id": trial["id"], "passed": record["passed"]}) + "\n"); journal.flush()
                print(f"{'PASS' if record['passed'] else 'FAIL'} {trial['id']} HTTP {response['status']} {response['elapsed_seconds']:.3f}s {case['label']}", flush=True)
                for problem in problems[:6]: print("  " + problem, flush=True)
                consecutive = consecutive + 1 if availability_failure(response) else 0
                write_report(folder, metadata, records, trials, health, stopped)
                if consecutive >= 3:
                    stopped = "Three consecutive availability failures; remaining cases not run."; break
    check_health("after")
    summary = write_report(folder, metadata, records, trials, health, stopped)
    print("REPORT", folder / "report.md", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["all_functional_tests_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="create fixtures; refuse to overwrite")
    mode.add_argument("--live", metavar="BASE_URL", help="explicitly opt into up to 30 paid-provider-backed POSTs")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--max-posts", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.generate:
        generate(args.cases)
    pack = strict_json(args.cases.read_bytes())
    offline = offline_checks(pack)
    if args.live:
        return run_live(pack, args.live, args.max_posts, args.timeout, offline, args.cases)
    print("OFFLINE ONLY: no live requests sent.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"ERROR: {sanitize(str(error))}", file=sys.stderr)
        raise SystemExit(2)
