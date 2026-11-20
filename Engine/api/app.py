"""Local HTTP adapter for era_engine.batch.simulate_batch.

This module does not implement or change simulation formulas, ratings,
calibration, or era rules. A JSON object is forwarded unchanged to
simulate_batch, and that function's dictionary is returned as JSON.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from era_engine.batch import BatchRequestError, simulate_batch

app = FastAPI(title="ERA Basketball batch API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/v1/simulate")
async def simulate(request: Request) -> JSONResponse:
    raw = await request.body()
    if not raw.strip():
        return _error(400, "malformed_request", "Request body must be a JSON object.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _error(400, "malformed_request", "Request body must be valid JSON.")
    if not isinstance(payload, dict):
        return _error(400, "malformed_request", "The request must be a JSON object.")
    try:
        result = await asyncio.to_thread(simulate_batch, payload)
    except BatchRequestError as exc:
        return _error(400, exc.code, str(exc))
    return JSONResponse(content=result)


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": code, "message": message})
