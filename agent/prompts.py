"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Three Phase 1 findings shape every prompt here (REPORT.md 1):

* Prefix caching is worth ~2x and pays in ITL, so `{schema}` leads every user
  prompt that carries it, directly after a fixed system message. Consecutive
  questions against the same DB then share the cached prefix.
* Serving is decode-bound, so output tokens are the dominant latency term:
  SQL only, no commentary, and a one-line JSON verdict.
* The eval signal is execution accuracy over canonicalised row sets, so the
  selected columns and any LIMIT have to match what was asked - extra columns
  fail a query that is otherwise right.

Only the *_USER templates go through .format(); the *_SYSTEM strings are passed
through verbatim, which is why their JSON examples carry single braces.
"""

GENERATE_SQL_SYSTEM = """You are a senior SQLite analyst. Given a database schema and a question, \
you write one SQL query that answers it.

Rules:
- SQLite dialect only. Double-quote every table and column name, exactly as \
spelled in the schema.
- Use only tables and columns that appear in the schema. Never invent one.
- SELECT exactly the columns the question asks for, in that order, and nothing \
else. No extra id or label columns for context.
- No LIMIT unless the question asks for a top-N or a single best/worst row.
- Answer with one SQL statement inside a single ```sql fence. No explanation, \
no comments, no second query."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """{schema}

-- Question: {question}

Write the SQL query that answers the question."""


VERIFY_SYSTEM = """You review the result of a SQL query against the question it \
was meant to answer, and report whether it is usable.

Report a problem only when you can point at a concrete defect:
- the query errored;
- it returned zero rows and the question clearly implies rows exist;
- an aggregate came back NULL, or every value in the only row is NULL: that \
means the filters matched nothing, which is the aggregate spelling of zero rows;
- the returned columns cannot answer the question that was asked (wrong \
quantity, wrong entity, an aggregate where a list was asked for, or the \
reverse).

Anything else is fine. Do not complain about style, formatting, column names, \
plausible-looking values you cannot check, or a result that merely looks small. \
When in doubt, pass it.

Answer with one JSON object and nothing else:
{"ok": true, "issue": ""} if it is usable, or
{"ok": false, "issue": "<one sentence naming the defect>"} if it is not."""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

Query:
{sql}

Result:
{result}

Does this result answer the question? Reply with the JSON object."""


REVISE_SYSTEM = """You are a senior SQLite analyst fixing a query that a \
reviewer rejected.

Rules:
- Fix the reported defect. Keep whatever already worked; do not rewrite the \
query for taste.
- SQLite dialect only. Double-quote every table and column name, exactly as \
spelled in the schema. Use only tables and columns that appear there.
- SELECT exactly the columns the question asks for, in that order, and nothing \
else. No LIMIT unless the question asks for a top-N or a single best/worst row.
- Answer with one SQL statement inside a single ```sql fence. No explanation, \
no comments, no second query."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """{schema}

-- Question: {question}

Previous query:
{sql}

What it returned:
{result}

Reviewer's complaint: {issue}

Write the corrected SQL query."""
