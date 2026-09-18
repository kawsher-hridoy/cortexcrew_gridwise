# GridWise Implementation Review

**Review date:** September 18, 2026  
**Scope:** `implementation_plan.md`, the official participant documents, all ten public sample cases, and the implementation inspected during the review.

> This document records the previously completed review. It is not a re-verification of changes made afterward. Concurrent edits to `app.py` and `llm.py` were observed during the review; code references describe the inspected snapshot.

## Overall verdict

**The architecture is well chosen, but the reviewed plan is not submission-ready.** The main problems are not the optimizer choice: they are constraint relaxation, numerical postprocessing, and incomplete guardrails.

### Sources of truth

- [Official problem statement](BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Preliminary_Problem_Statement_GridWise_LLM.pdf): directives, schemas, guardrails, battery behavior, energy accounting, and optimization validity.
- [Participant guide and evaluation rubric](BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Participant_Guide_&_Evaluation_Rubric_GridWise_LLM.pdf): scoring, performance, deployment, repository, and submission requirements.
- [Public sample cases](BUP_CSE_FEST_2026_Participant_Docs/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json): ten worked examples, not the hidden judge set.

The working `public_cases.json` was confirmed to match the official sample JSON semantically.

## Findings, highest priority first

### 1. Critical: the relaxation ladder violates the challenge contract

**References:** `implementation_plan.md:83-94`, `solve.py:397-467`; Problem Statement sections 5 and 8; Participant Guide sections 7 and 9.

The plan intentionally drops grid caps, reserve requirements, and battery windows when solving fails. That produces a schedule for a **different problem**, not a valid fallback.

The official documents explicitly establish that:

- Applicable directives are hard constraints.
- Valid hidden scoring scenarios are feasible.
- Judges replay schedules against organizer ground truth, not merely the team's reported interpretation.
- A directive violation eliminates optimization credit for that case.

There is an additional inconsistency: `summarize()` still says directives were applied as hard constraints after the solver dropped them.

**Recommendation:** Remove constraint-dropping from successful responses. Retry numerical failures without changing the constraints. Return a controlled error if no verified schedule can be produced. A fallback schedule is acceptable only if it passes replay against **all original directives**.

### 2. Critical: rounding already causes failures on feasible inputs

**References:** `implementation_plan.md:73-81`, `solve.py:207-285`.

This is experimentally confirmed, not just a theoretical concern.

| Verification performed | Result |
| --- | ---: |
| Original public cases, offline | **10/10 passed**, matching reference costs |
| Feasible energy-scaled variants | 300 tested |
| Variants triggering fallback after numerical postprocessing | **77** |
| Returned plans violating original directives beyond 0.01 tolerance | **24** |

For these variants, demand, solar, battery energy/rate limits, and directive kWh values were multiplied by the same positive factor. This preserves feasibility; tariffs, solar-reduction factors, and directive hours were unchanged.

The test used `random.Random(8247)` to generate 30 scale factors with `uniform(0.1, 4.0)`, applying each factor to all ten public cases. Original-constraint violations were checked with separate replay logic at the official 0.01 tolerance.

One reproduction: scaling SAMPLE-10 by `1.8058782404676035` triggered the passive fallback, exceeding the hour-19 grid cap by approximately **72.24 kWh**.

The failure chain is:

1. The LP finds a solution.
2. Six-decimal rounding creates small accumulated discrepancies.
3. Replay rejects them using a `1e-6` tolerance.
4. The ladder drops constraints or substitutes an idle schedule.
5. A tiny numerical discrepancy becomes a substantial operational violation.

Also, `_repair_neutrality()` checks charge/discharge bounds but not all affected subsequent energy bounds and grid caps.

**Recommendation:** Preserve more numerical precision, coordinate rounding with replay tolerances, and verify every repair against the complete constraint set. Never treat a rounding failure as permission to relax directives.

These 300 transformed examples are useful regression material, **not an estimate of hidden-test failure rates**.

### 3. High: the actual guardrails are weaker than the plan claims

**References:** `implementation_plan.md:55-62,172-180`, `llm.py:209-227`, `schemas.py:189-294`.

The following discrepancies were reproduced:

| Model output | Actual behavior |
| --- | --- |
| `no_charge_window` with an extra non-null `factor` | Accepted; extra field discarded |
| `no_op` containing non-null hours/grid-cap data | `_coerce()` replaces the adjustment with `null`, then accepts it |
| Object instead of a string in `directive_type` | Raises `TypeError`, bypassing the intended guardrail-repair handling |

The `no_op` case is particularly important: direct validator tests reject it, but the **real model-output path accepts it** because preprocessing erases the violation.

Request validation is also not strict about JSON types: boolean hours, boolean demand, and numeric strings were accepted through coercion.

**Recommendation:** Validate type-specific key sets and values deterministically. Permit removal of known unused **null** fields, but reject unexpected non-null fields. Test raw model output through the complete adapter-and-validator path, not just `validate_batch()`.

### 4. High: verification evidence does not yet establish full-pipeline correctness

**References:** `implementation_plan.md:134-213`, `run_cases.py:150-208`.

The offline results are genuine and were reproduced, but they inject organizer interpretations and therefore do not test the mandatory LLM interpretation step.

Other gaps:

- The test runner imports production `compile_directives()` and `replay()`. Shared bugs can pass both production and verification.
- The live-response checker accepts a missing interpretation `explanation`.
- It also accepts a reported `NaN` total because its numeric comparison lacks an explicit finiteness check.
- No persistent regression-test files were present for the extensive edge-case claims.
- The plan records no completed live semantic or latency results.

**Recommendation:** Add an independently implemented acceptance checker and persistent regression tests. Prioritize:

- Every directive, paraphrases, percentages, and distractors.
- Compatible overlapping constraints.
- Fractional and boundary-value scenarios.
- Malformed model output through the actual interpretation path.
- Provider failures, repeated requests, and measured end-to-end latency.

### 5. Medium: readiness, deadlines, and logging claims need correction

**References:** `implementation_plan.md:20,191-202,267`; `app.py`.

Three concrete issues:

- **False readiness:** With interpreter configuration unavailable, `/health` still returns 200 while a valid optimization request returns 500. This was reproduced.
- **Partial deadline:** `REQUEST_BUDGET_SECONDS` wraps interpretation only. Solver execution, replay, and response construction are outside it; the synchronous solver also runs on the event loop. A local fault-injection test confirmed that delayed response construction can exceed the configured budget while still returning 200.
- **Unsafe exception logging:** Unexpected-error handlers use `logger.exception()`. A synthetic fault test confirmed that exception messages and full tracebacks reach logs, despite documentation claiming otherwise. No real secret was used in the test.

**Recommendation:** Make readiness reflect essential initialization, enforce an end-to-end time budget, and use sanitized error logging. Readiness does not need to make a paid model call on every health request.

### 6. Submission requirements need explicit acceptance checks

**References:** `implementation_plan.md:206-256`; Participant Guide sections 2, 4, 5, and 11.

The outstanding Docker and deployment work is correctly identified, but it needs stronger completion criteria:

- Exact, pullable image tag or digest, not a placeholder.
- Clean-machine pull/run test.
- Both endpoints tested externally.
- At least one successful model-backed optimization inside the deployed container.
- Repository visibility changed after the submission deadline.
- Provider access, quota, and required configuration documented.

**Important clarification:** The three-minute video has no base-score points, but it is still a **required submission item**, not optional.

There is also documentation drift: the plan says temperature is pinned to zero, while the inspected `Settings.from_env()` omits temperature by default.

## What to keep

Several decisions are strong:

- **Signed battery-flow LP:** Correct for this lossless model; no MILP is needed merely to prevent simultaneous charging and discharging.
- **Real LLM interpretation path:** The model produces the directives consumed by optimization.
- **One batched interpretation call:** Sensible latency design.
- **Deterministic summary:** Avoids an unnecessary second model call.
- **Cost-only objective:** Correctly avoids inventing efficiency losses, degradation penalties, or peak minimization.
- **Separate offline/live testing modes:** Useful structure once the verification gaps are fixed.

## Recommended priority

**Fix constraint relaxation -> fix numerical postprocessing -> strengthen guardrails -> establish live and independent verification -> finish deployment artifacts.**

Interpretation and constraint correctness account for **50 points**, and optimization credit depends on validity. Further optimizer sophistication is much less valuable than addressing these failures.

**Bottom line:** Keep the architecture. Correct its failure behavior and strengthen the evidence before submission.

## Review scope and limitations

- The review was read-only; implementation files were not changed.
- No live model-provider calls or deployment changes were made as part of the review.
- Tests used offline fixtures, in-process HTTP clients, synthetic faults, and temporary in-memory patches.
- Live model accuracy, deployed latency, registry pullability, and external endpoint reachability were not verified.
- Concurrent edits to `app.py` and `llm.py` were observed. Findings describe the inspected snapshot and should be rechecked after fixes.
- This file was created afterward to preserve the review at the user's request.
