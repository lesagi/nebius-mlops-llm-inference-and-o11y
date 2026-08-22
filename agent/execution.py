"""SQL execution helper (provided complete, bounded in Phase 6).

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.

Execution is bounded by wall clock. Phase 6 found a single model-written query
that ran for 512 seconds inside one request - sqlite3.connect(timeout=...) bounds
lock acquisition, not query runtime, so nothing stopped it. It held an anyio
threadpool thread and a core for the whole eight minutes, which is how a p50 of
1.1s coexisted with a p99 of 38s: the queue forms in front of the handler, where
neither the dashboard nor the trace can see it.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass

from agent.schema import db_path

# Real BIRD queries finish in milliseconds on these files; anything past this is
# a cross join or a scan the model did not intend. Aborting is better than
# waiting: the verifier is told the query was too slow and revise gets to write
# a cheaper one, which is the loop doing its job.
QUERY_BUDGET_SECONDS = float(os.environ.get("QUERY_BUDGET_SECONDS", "2.0"))


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(self, max_rows: int = 10) -> str:
        """Compact text rendering for prompt context."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."
        cols = ", ".join(self.columns or [])
        # SQL NULL, not Python's None: the verifier is asked to treat a NULL
        # aggregate as "the filters matched nothing", and it cannot do that if
        # the value arrives spelled like a string.
        preview = "\n".join(
            " | ".join("NULL" if c is None else str(c) for c in row)
            for row in (self.rows or [])[:max_rows]
        )
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        return f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0,
                budget_seconds: float = QUERY_BUDGET_SECONDS) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    path = db_path(db_id)
    try:
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            # The only way to bound a running query: sqlite calls this every N
            # VM instructions and aborts when it returns true. A signal-based
            # timeout would not work here - this runs in a worker thread.
            deadline = time.monotonic() + budget_seconds
            # Every 1M VM instructions, not every 10K: the callback takes the GIL,
            # and this runs in a worker thread beside every other request. 1M still
            # gives tens of checks per second on a scan, which is ample for a
            # budget measured in seconds.
            conn.set_progress_handler(lambda: time.monotonic() > deadline, 1_000_000)
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except sqlite3.OperationalError as e:
        # "interrupted" is the progress handler firing. Say so in words the
        # verifier can act on, rather than leaking sqlite's internal wording.
        if "interrupted" in str(e):
            return ExecutionResult(
                ok=False,
                error=f"the query was still running after {budget_seconds:.0f}s and was "
                      "cancelled - it is too expensive, so scan less data",
            )
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
