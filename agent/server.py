"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from agent.graph import MAX_ITERATIONS, AgentState, graph  # noqa: E402
from agent.schema import SAMPLE_VALUES  # noqa: E402

# Langfuse tracing. If keys are set we import it; failures are NOT swallowed -
# a misconfigured Langfuse should not silently produce zero traces. Missing keys
# would only degrade the SDK to a NoOpTracer, which is exactly the silent
# nothing this guard exists to avoid.
_LANGFUSE_ON = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
if _LANGFUSE_ON:
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Flush pending spans on the way out.

    The OTel batch processor ships every 5s, and SIGTERM from a supervisor beats
    the SDK's atexit hook, so without this the last trace of every run arrives
    truncated: no root span, hence no trace name and a short observation list.
    """
    yield
    if _LANGFUSE_ON:
        get_client().flush()


app = FastAPI(lifespan=lifespan)


def _trace_config(tags: dict[str, str]) -> dict[str, Any]:
    """LangChain config carrying the Langfuse trace attributes.

    Langfuse reads `langfuse_tags` / `langfuse_trace_name` only off the *root*
    run's metadata, and only with the exact types it expects - hand it a dict
    where it wants a list and the tags are dropped without an error. A Langfuse
    tag is a flat string, so the caller's dict is flattened to "key:value" for
    the trace list's tag column while the dict itself stays in metadata, where
    it keeps its structure for querying.
    """
    metadata: dict[str, Any] = dict(tags)
    metadata["langfuse_tags"] = [f"{k}:{v}" for k, v in sorted(tags.items())]
    # Without this every trace is named "LangGraph", which makes the trace list
    # unreadable.
    metadata["langfuse_trace_name"] = "answer"
    return {
        # One handler per request: it keeps mutable per-run state on self, and
        # constructing one is free - it just grabs the process-wide client.
        "callbacks": [CallbackHandler()] if _LANGFUSE_ON else [],
        "metadata": metadata,
    }


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    # Both are read from the environment at import, so they are properties of
    # this process. A run comparing two configurations has no other way to tell
    # which one it is talking to, and getting that wrong invalidates the run -
    # reusing an already-running agent is exactly how that happens.
    return {
        "status": "ok",
        "schema_values": "on" if SAMPLE_VALUES else "off",
        "max_iterations": str(MAX_ITERATIONS),
        "langfuse": "on" if _LANGFUSE_ON else "off",
    }


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    """Async on purpose. A sync def here is run by FastAPI in anyio's threadpool,
    which holds one of its 40 threads for the whole run - every sequential model
    call included. Measured at 20 rps: the agent's thread count pegged at 65
    (26 idle + the 40-token limiter) within 5 seconds and never moved, while
    established connections climbed to 483 and p50 hit 20s - with vLLM at
    0.62s/call and 9 of 16 cores idle. A run is ~2.6 network waits in a row; it
    does not need to own a thread to do them.
    """
    state = AgentState(question=req.question, db_id=req.db)
    try:
        final = await graph.ainvoke(state, config=_trace_config(req.tags))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sql = final.get("sql", "")
    iteration = final.get("iteration", 0)
    history = final.get("history", [])
    execution = final.get("execution")

    if execution is None:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error="agent produced no execution result",
            history=history,
        )
    if not execution.ok:
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=iteration,
            ok=False,
            error=execution.error,
            history=history,
        )

    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=iteration,
        ok=True,
        history=history,
    )
