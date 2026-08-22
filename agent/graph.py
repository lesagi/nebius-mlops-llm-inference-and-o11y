"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# Ships at 3, which is the brief's range (3-5) and what produced
# results/eval_baseline.json. Env-driven because 2 is the better value and Phase 6
# needed to A/B it with a restart rather than an edit: Phase 5 measured the third
# attempt as worthless (per-iteration pass rate 33.3 / 36.7 / 36.7 - nothing
# anywhere was fixed at iteration 2, while 6 of 30 questions paid two extra
# sequential model calls for it), and Phase 6 measured what dropping it buys -
# client p95 6.04 -> 3.78s at 10 rps, pass rate unchanged at 36.67%. So the SLO
# numbers in REPORT.md 3 are all at MAX_ITERATIONS=2.
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "3"))

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# Revise samples instead of decoding greedily. At temperature 0 a revise that
# reproduces the previous SQL is a dead end by construction: same query, same
# rows, same verdict, same context next time round, so every remaining iteration
# repeats it. Measured on the H100 - the financial A14/A15 question revised twice
# and returned byte-identical SQL both times, 1.5s for nothing. Generate stays
# greedy so eval runs stay reproducible.
REVISE_TEMPERATURE = 0.7


def llm(temperature: float = 0.0) -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=temperature,
        # Serving is decode-bound (REPORT.md 1) and 16.6% of Phase 2 requests
        # finished on `length`, truncating SQL mid-query. A cap bounds the tail;
        # 512 is far more than one SELECT or one JSON verdict needs.
        max_tokens=512,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.

    Reasoning blocks are dropped first. The H100 checkpoint is the non-thinking
    Instruct-2507, but the small CPU stand-ins used for graph work are hybrid
    models that emit <think> by default, and that prose would otherwise be
    handed to sqlite as SQL.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = await llm().ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


# execute_node stays sync on purpose: sqlite is blocking work, and LangGraph runs
# a sync node in a worker thread when the graph is driven with ainvoke. So the
# blocking part gets a thread for the milliseconds it needs it, instead of one
# thread being held for the whole 2-6 sequential model calls of a run.
def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def _parse_verdict(text: str) -> tuple[bool, str]:
    """Pull {"ok": bool, "issue": str} out of a verifier reply.

    Non-greedy so a fenced or prose-wrapped object still matches. A reply we
    cannot read is not evidence of a bug, so it fails open: passing costs
    nothing, whereas a speculative revise costs two more vLLM calls and can
    turn a correct answer into a wrong one.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            verdict = json.loads(match.group(0))
            return bool(verdict.get("ok", True)), str(verdict.get("issue", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
    # ponytail: fail open on an unreadable verdict. Watch it if revise never
    # fires - a stricter parse or a retry only earns its place once Phase 5
    # shows the loop paying for itself.
    return True, ""


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    A sqlite error needs no model to judge, so that case short-circuits: it
    saves a full vLLM call out of the 5s end-to-end SLO budget, and the error
    text is a better revise hint than anything the verifier would paraphrase.
    Every case that needs actual judgement - including zero rows - goes to the
    model.
    """
    result = state.execution
    if result is None or not result.ok:
        issue = (result.error if result else None) or "the query did not execute"
        return {
            "verify_ok": False,
            "verify_issue": issue,
            "history": state.history + [{"node": "verify", "ok": False, "issue": issue}],
        }

    response = await llm().ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result.render(max_rows=5),
        )),
    ])
    ok, issue = _parse_verdict(response.content)
    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


async def revise_node(state: AgentState) -> dict:
    """Rewrite the SQL to fix state.verify_issue. Same shape as generate_sql_node."""
    response = await llm(temperature=REVISE_TEMPERATURE).ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=state.execution.render(max_rows=5) if state.execution else "(not executed)",
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [
            {"node": "revise", "sql": sql, "issue": state.verify_issue},
        ],
    }


def route_after_verify(state: AgentState) -> str:
    """Loop into revise, or stop. Literals key the add_conditional_edges map."""
    if state.verify_ok or state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
