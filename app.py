"""GridWise HTTP service: GET /health and POST /optimize-energy.

Request flow: validate -> model interprets every note -> deterministic
guardrails -> compile directives -> optimize the full horizon -> replay the
serialized plan -> serialize the response.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm import Interpreter, ProviderError, Settings
from schemas import (
    DirectiveInterpretationOut,
    GuardrailError,
    HealthResponse,
    HourPlanOut,
    OptimizeRequest,
    OptimizeResponse,
    SemanticRequestError,
    check_base_feasibility,
)
from solve import solve, summarize

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gridwise.api")

# Leaves headroom under the judge's 30 second per-request timeout.
REQUEST_BUDGET_SECONDS = float(os.getenv("REQUEST_BUDGET_SECONDS", "25"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared model client once, without blocking readiness on it."""
    try:
        app.state.interpreter = Interpreter(Settings.from_env())
        logger.info("model interpreter ready")
    except Exception as error:  # noqa: BLE001 - never print the key or a traceback
        app.state.interpreter = None
        logger.error("model interpreter unavailable: %s", type(error).__name__)
    try:
        yield
    finally:
        interpreter = getattr(app.state, "interpreter", None)
        if interpreter is not None:
            await interpreter.close()


app = FastAPI(
    title="GridWise Smart Campus Energy Optimizer",
    version="1.0.0",
    debug=False,
    lifespan=lifespan,
)


def _error(code: int, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"error": message, "request_id": request_id},
    )


@app.middleware("http")
async def correlate(request: Request, call_next) -> Response:
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error [%s] %s", request_id, request.url.path)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error", request_id
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if request.url.path != "/health":
        logger.info(
            "[%s] %s %s -> %s in %.0f ms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Malformed JSON and structural schema violations are 400, not 422.

    The problem statement reserves 422 for requests that are well formed but
    semantically invalid, which is handled inside the endpoint.
    """
    request_id = getattr(request.state, "request_id", "-")
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    detail = first.get("msg", "invalid request")
    message = f"invalid request: {location or 'body'}: {detail}".strip()
    logger.info("[%s] rejected request: %s", request_id, message)
    return _error(status.HTTP_400_BAD_REQUEST, message, request_id)


@app.exception_handler(Exception)
async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("[%s] unhandled %s", request_id, type(exc).__name__)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error", request_id)


@app.get("/health", response_model=HealthResponse)
async def health() -> dict[str, str]:
    """Readiness for the judge harness; never calls the model provider."""
    return {"status": "ok"}


def _build_response(
    request_model: OptimizeRequest, directives: list[Any]
) -> OptimizeResponse:
    solution = solve(request_model, directives)
    return OptimizeResponse(
        scenario_id=request_model.scenario_id,
        directive_interpretation=[
            DirectiveInterpretationOut(
                note_index=directive.note_index,
                applies=directive.applies,
                directive_type=directive.directive_type,
                structured_adjustment=directive.structured_adjustment(),
                explanation=directive.explanation,
            )
            for directive in directives
        ],
        hourly_plan=[
            HourPlanOut(
                hour=entry.hour,
                grid_kwh=entry.grid_kwh,
                solar_used_kwh=entry.solar_used_kwh,
                battery_action=entry.battery_action,
                battery_kwh=entry.battery_kwh,
                battery_energy_after_kwh=entry.battery_energy_after_kwh,
            )
            for entry in solution.plan
        ],
        total_grid_kwh=solution.totals.total_grid_kwh,
        total_cost_bdt=solution.totals.total_cost_bdt,
        peak_grid_kwh=solution.totals.peak_grid_kwh,
        plan_summary=summarize(
            directives, solution.plan, solution.totals, solution.compiled
        ),
    )


@app.post(
    "/optimize-energy",
    response_model=OptimizeResponse,
    responses={400: {}, 422: {}, 500: {}},
)
@app.post("/optimize-energy/", include_in_schema=False, response_model=OptimizeResponse)
async def optimize_energy(payload: OptimizeRequest, request: Request) -> Any:
    request_id = getattr(request.state, "request_id", "-")

    try:
        check_base_feasibility(payload)
    except SemanticRequestError as error:
        logger.info("[%s] semantically invalid request: %s", request_id, error)
        return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error), request_id)

    interpreter: Interpreter | None = getattr(request.app.state, "interpreter", None)
    if interpreter is None:
        logger.error("[%s] interpreter is not configured", request_id)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "note interpretation is unavailable",
            request_id,
        )

    try:
        directives = await asyncio.wait_for(
            interpreter.interpret(payload.operator_notes, payload.battery.capacity_kwh),
            timeout=REQUEST_BUDGET_SECONDS,
        )
    except (ProviderError, GuardrailError) as error:
        logger.error("[%s] interpretation failed: %s", request_id, error)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "could not interpret the operator notes",
            request_id,
        )
    except asyncio.TimeoutError:
        logger.error("[%s] interpretation exceeded the request budget", request_id)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "note interpretation timed out",
            request_id,
        )

    applied = [d.directive_type for d in directives if d.applies]
    logger.info(
        "[%s] scenario=%s notes=%d applied=%s",
        request_id,
        payload.scenario_id,
        len(payload.operator_notes),
        applied or "none",
    )

    # The solver and replay are pure CPU work measured in single-digit
    # milliseconds, so they run inline rather than on a worker thread.
    return _build_response(payload, directives)
