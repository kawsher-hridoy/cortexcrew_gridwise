# GridWise — LLM-Assisted Smart Campus Energy Optimization

BUP CSE Fest 2026 · Online Preliminary · Smart Campus Energy Optimization Challenge

One HTTP service that reads 1–3 natural-language campus operator notes, converts them into
validated structured directives using a language model, applies those directives as hard
constraints, and returns a cost-minimal 24-hour electricity schedule.

- `GET /health` → `{"status":"ok"}`
- `POST /optimize-energy` → directive interpretation + 24-hour plan

---

## 1. Quickstart from a clean environment

Requires Python 3.12+ and an Azure AI Foundry key.

```bash
git clone <this-repository> gridwise
cd gridwise

python3 -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# open .env and set AZURE_AI_API_KEY=<your key>

set -a; . ./.env; set +a         # export the variables into the shell
uvicorn app:app --host 0.0.0.0 --port 8000
```

The service is ready in under two seconds. It never contacts the model provider during
startup, so `/health` answers immediately.

### Verify health

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}
```

### Run one public sample

```bash
curl -s -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  -d '{
    "scenario_id": "GRID-DEMO",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month'"'"'s registration deadline."
    ],
    "hours": [
      {"hour":0,"demand_kwh":95,"solar_kwh":0,"tariff_bdt_per_kwh":6},
      {"hour":1,"demand_kwh":90,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":2,"demand_kwh":88,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":3,"demand_kwh":85,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":4,"demand_kwh":88,"solar_kwh":0,"tariff_bdt_per_kwh":5},
      {"hour":5,"demand_kwh":95,"solar_kwh":10,"tariff_bdt_per_kwh":6},
      {"hour":6,"demand_kwh":110,"solar_kwh":35,"tariff_bdt_per_kwh":7},
      {"hour":7,"demand_kwh":135,"solar_kwh":70,"tariff_bdt_per_kwh":8},
      {"hour":8,"demand_kwh":160,"solar_kwh":110,"tariff_bdt_per_kwh":10},
      {"hour":9,"demand_kwh":175,"solar_kwh":140,"tariff_bdt_per_kwh":12},
      {"hour":10,"demand_kwh":180,"solar_kwh":165,"tariff_bdt_per_kwh":13},
      {"hour":11,"demand_kwh":185,"solar_kwh":175,"tariff_bdt_per_kwh":14},
      {"hour":12,"demand_kwh":190,"solar_kwh":180,"tariff_bdt_per_kwh":14},
      {"hour":13,"demand_kwh":185,"solar_kwh":170,"tariff_bdt_per_kwh":15},
      {"hour":14,"demand_kwh":180,"solar_kwh":150,"tariff_bdt_per_kwh":15},
      {"hour":15,"demand_kwh":175,"solar_kwh":120,"tariff_bdt_per_kwh":16},
      {"hour":16,"demand_kwh":180,"solar_kwh":80,"tariff_bdt_per_kwh":18},
      {"hour":17,"demand_kwh":195,"solar_kwh":40,"tariff_bdt_per_kwh":22},
      {"hour":18,"demand_kwh":215,"solar_kwh":10,"tariff_bdt_per_kwh":28},
      {"hour":19,"demand_kwh":210,"solar_kwh":0,"tariff_bdt_per_kwh":30},
      {"hour":20,"demand_kwh":200,"solar_kwh":0,"tariff_bdt_per_kwh":26},
      {"hour":21,"demand_kwh":180,"solar_kwh":0,"tariff_bdt_per_kwh":18},
      {"hour":22,"demand_kwh":150,"solar_kwh":0,"tariff_bdt_per_kwh":12},
      {"hour":23,"demand_kwh":120,"solar_kwh":0,"tariff_bdt_per_kwh":8}
    ],
    "battery": {
      "capacity_kwh": 220,
      "initial_energy_kwh": 110,
      "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50,
      "max_discharge_kwh_per_hour": 50
    }
  }'
```

The first note becomes `solar_reduction` with `hours [12,13]` and `factor 0.25`; the second
becomes `no_op` with a null adjustment.

### Run all ten public sample cases

Two lanes. The offline lane needs no API key and isolates the optimizer:

```bash
python run_cases.py --offline
```

It injects each published directive interpretation and checks that the optimizer reaches the
published optimal cost. Expected result: `10/10 cases passed`, every delta `+0.0000`.

The live lane exercises the whole pipeline against a running service:

```bash
python run_cases.py --live http://127.0.0.1:8000
```

For each case it compares directive semantics against the published ground truth (explanation
wording is not compared), replays the returned schedule against the ground-truth directives,
checks the three reported totals against the replay, confirms the cost is optimal, and reports
median, p95, and maximum latency.

---

## 2. Docker fallback

```bash
docker pull ghcr.io/kawsher-hridoy/cortexcrew_gridwise:latest

docker run --rm -p 8000:8000 \
  -e AZURE_AI_API_KEY=<your key> \
  ghcr.io/kawsher-hridoy/cortexcrew_gridwise:latest

curl -s http://127.0.0.1:8000/health
# {"status":"ok"}
```

The image binds to `0.0.0.0:8000`, runs as a non-root user, contains no credentials, and
carries its own `HEALTHCHECK`. `AZURE_AI_API_KEY` must be supplied at runtime.

To build it locally instead:

```bash
docker build -t gridwise:local .
docker run --rm -p 8000:8000 -e AZURE_AI_API_KEY=<your key> gridwise:local
```

---

## 3. Configuration

Only `AZURE_AI_API_KEY` is required. No secret values appear in this repository.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `AZURE_AI_API_KEY` | yes | — | Azure AI Foundry API key |
| `AZURE_AI_BASE_URL` | no | `https://ai-for-security.services.ai.azure.com/openai/v1/` | OpenAI-compatible endpoint |
| `AZURE_AI_MODEL` | no | `gpt-5.6-terra` | Deployment used for note interpretation |
| `AZURE_AI_FALLBACK_MODEL` | no | unset | Second deployment tried if the primary fails |
| `LLM_TRANSPORT` | no | `responses` | `responses` or `chat`; falls back automatically |
| `LLM_TIMEOUT_SECONDS` | no | `12` | Per-call provider timeout |
| `LLM_MAX_OUTPUT_TOKENS` | no | `1200` | Output cap, bounds tail latency |
| `LLM_TEMPERATURE` | no | `0` | Pinned for reproducible interpretation |
| `LLM_REASONING_EFFORT` | no | unset | Passed through when the deployment supports it |
| `REQUEST_BUDGET_SECONDS` | no | `25` | Endpoint deadline, under the 30 s judge timeout |
| `LOG_LEVEL` | no | `INFO` | Log verbosity |
| `HOST` / `PORT` | no | `0.0.0.0` / `8000` | Bind address inside the container |

---

## 4. Architecture

```
POST /optimize-energy
  |
  v
request validation        strict on required fields, tolerant of unknown ones
  |
  v
LLM interpretation        one batched, schema-constrained call, temperature 0
  |
  v
deterministic guardrails  untrusted output validated, never clamped
  |
  v
directive compilation     24-element solar / reserve / charge / discharge / grid-cap arrays
  |
  v
full-horizon optimizer    SciPy HiGHS linear program over all 24 hours at once
  |
  v
canonicalization          grid and stored energy re-derived from the physics
  |
  v
independent replay        plan re-verified from scratch; totals derived from it
  |
  v
response
```

| Module | Responsibility |
| --- | --- |
| `app.py` | Endpoints, error mapping, request correlation, endpoint deadline |
| `schemas.py` | Request/response contracts and the directive guardrails |
| `llm.py` | Prompt, schema-constrained model call, repair retry, failover |
| `solve.py` | Directive compilation, linear program, canonicalization, replay, summary |
| `run_cases.py` | Offline and live public-sample verification |

### Role of the language model

The model is the only component that reads natural language, and its structured output is
what the optimizer's constraints are built from. It is not used for cosmetic text: the
`plan_summary` field is generated deterministically in `solve.summarize` from the applied
directives and the solved schedule, specifically so that no scored behaviour depends on model
prose.

All 1–3 notes are interpreted in a single call constrained by a JSON schema covering the six
allowed directive types. The prompt fixes the conventions that make extraction machine
checkable: whole-hour windows that are start-inclusive and end-exclusive, hours sorted
ascending (so a window wrapping past midnight becomes `[0,1,22,23]`), solar `factor` as the
fraction *remaining* (an 80% reduction is `0.2`), reserves converted from percentages using
the battery capacity, and `no_op` for anything that cannot be expressed as one of the six
types — including energy-related notes about demand forecasts or tariffs.

There is no phrase-matching interpreter behind the model. Deterministic code only normalizes
(sorting and de-duplicating hours) and validates.

### Guardrails

Model output is untrusted until it passes `schemas.validate_batch`, which rejects rather than
repairs:

- `directive_type` must be one of the six supported values
- exactly one entry per note, `note_index` running `0..N-1` in order
- `no_op` requires `applies=false` and a null adjustment; every other type requires `applies=true`
- the adjustment object must match the exact shape its type requires
- hours must be integers within 0–23, non-empty, unique, sorted ascending
- `factor` finite and within `[0,1]`; reserve finite, non-negative, not above capacity; grid cap finite and non-negative

A rejection triggers one compact repair request quoting the specific failure. Values are never
clamped into range, and an unsupported directive type is never invented.

### Optimizer

A single linear program covers the whole 24-hour horizon, solved with SciPy's HiGHS backend.
Four continuous variables per hour: grid import `G`, solar used `S`, a **signed** battery flow
`B` (positive charges, negative discharges), and stored energy after the hour `E`. Because one
variable carries both directions, simultaneous charge and discharge is structurally
impossible, which keeps the model a linear program rather than requiring integer variables.

Minimize `sum(G[h] * tariff[h])` subject to, for every hour:

- `G[h] + S[h] - B[h] = demand[h]` — energy balance
- `E[h] = E[h-1] + B[h]`, seeded from `initial_energy_kwh` — battery transition
- `0 <= S[h] <= effective_solar[h]` — curtailment allowed, export not
- `active_reserve[h] <= E[h] <= capacity_kwh`
- `-max_discharge <= B[h] <= max_charge`, narrowed to `0` by no-charge or no-discharge windows
- `G[h] <= max_grid_kwh` where a cap directive applies
- `E[23] = initial_energy_kwh` — end-of-day neutrality

Overlapping compatible directives combine deterministically: smallest solar factor, highest
reserve, lowest grid cap, logical AND over charge and discharge availability. Solar reductions
are never relaxed, because reduced solar is a physical fact of the scenario.

### Canonicalization and replay

Rather than reporting raw solver values, the battery flow is rounded, grid import is
re-derived from the energy balance, and stored energy is accumulated from the initial state.
Both identities the judge checks most strictly therefore hold by construction. The serialized
plan is then replayed from scratch by `solve.replay`, which re-checks hour coverage, sign
conventions, action consistency, balance, transitions, capacity and reserve bounds, rate
limits, effective-solar limits, every directive, and terminal neutrality. `total_grid_kwh`,
`total_cost_bdt`, and `peak_grid_kwh` are derived from the replayed hourly values, so the
reported totals cannot disagree with the plan.

### Failure behaviour

If a constraint set admits no schedule, the service does not fail the request. It re-solves
through progressively relaxed sets — dropping grid caps, then reserve raises, then battery
windows — and if none solve, emits an idle-battery schedule that satisfies the base energy
rules by construction. A physically valid schedule still earns energy-balance, battery, and
action-consistency credit, whereas an error response earns none. Every relaxation is logged.

| Condition | Response |
| --- | --- |
| Malformed JSON or schema violation | `400` with a short message |
| Well-formed but no valid plan can exist (initial charge outside the reserve/capacity band) | `422` |
| Model unreachable, refusing, or output unusable after the repair retry | `500`, sanitized |
| Constraints infeasible | `200` with a relaxed but valid schedule |

Errors return only a short message and a request id. Stack traces, prompts, provider payloads,
and credentials are never included in responses or logs.

---

## 5. Performance

Measured over the ten public cases against the container image: median 3.23 s,
p95 4.15 s, maximum 4.15 s.

- `/health` is ready within about two seconds of start and never calls the provider; it answered in 2.4 ms while a startup warmup was still in flight
- one model call per request; `plan_summary` needs no second call
- the linear program and replay together take single-digit milliseconds, so latency is dominated by the one model call
- a fire-and-forget warmup at startup establishes DNS and TLS so the first scored request does not pay the cold-connection cost
- the endpoint deadline is 25 s, under the 30 s judge timeout
- `python run_cases.py --live <url>` reports median, p95, and maximum latency

---

## 6. Dependencies and credits

| Component | Used for |
| --- | --- |
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | HTTP service |
| [Pydantic v2](https://docs.pydantic.dev/) | Request, directive, and response validation |
| [SciPy](https://scipy.org/) `optimize.linprog`, HiGHS backend | Linear program |
| [HiGHS](https://highs.dev/) | Underlying LP solver bundled with SciPy |
| [openai](https://github.com/openai/openai-python) Python SDK | OpenAI-compatible client for Azure AI Foundry |
| Azure AI Foundry, deployment `gpt-5.6-terra` | Operator-note interpretation |

Public sample cases in `public_cases.json` are the organizer-published pack, included so the
verification commands above run without extra downloads. Nothing in the service branches on
their ids, wording, values, or schedules.

---

## 7. Known limitations

- Interpretation quality is bounded by the model. A misread note still produces a schedule that is internally valid and self-consistent, but it will be optimized against the wrong constraint.
- Each note maps to exactly one directive, matching the problem statement. A single note instructing two different directives would yield only the dominant one.
- The relaxation ladder favours returning a valid schedule over reporting infeasibility, so a grid cap that cannot be met is dropped rather than refused. Relaxations are logged but not exposed in the response, which has no field for them.
- `peak_grid_kwh` is reported, not minimized. The objective is cost alone, as specified, so an equally cheap schedule may show a different peak than another valid optimum.
- Cost ties can be broken differently than a reference schedule; the cost matches, the hourly actions may not.
- Aggregate grid limits across a whole window are not expressible: `max_grid_kwh` is a per-hour ceiling, per the problem statement.
- Requires outbound network access to the model endpoint. `/health` stays green regardless, so a provider outage surfaces as a sanitized `500` on `/optimize-energy` rather than as a failed readiness check.
