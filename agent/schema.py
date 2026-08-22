"""Schema-rendering helper (provided complete, extended in Phase 5).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.

Low-cardinality text columns can also carry their distinct values as a comment
(`SCHEMA_SAMPLE_VALUES=1`, off by default). It supplies the literals the model
kept guessing - `molecule.label = 'carcinogenic'` where the values are '+'/'-' -
and Phase 5 measured it as a net loss anyway. Kept because the negative result
is the evidence.
"""
from __future__ import annotations

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


# Default off: measured worse. results/eval_schema_values.json scored 30.0%
# against the baseline's 36.7% - see REPORT.md 2 for the mechanism, but the short
# version is that annotating only the low-cardinality columns tells the model
# those columns matter more, and that two questions the greedy decode had right
# went wrong on exactly that. SCHEMA_SAMPLE_VALUES=1 turns it back on.
# Process-wide rather than a parameter: render_schema is lru_cached on db_id
# alone, and the comparison is between two agent processes.
SAMPLE_VALUES = os.environ.get("SCHEMA_SAMPLE_VALUES", "0") != "0"
# Both the probe and the cardinality test: fewer rows back than the limit means
# that IS the complete distinct set, so no COUNT(DISTINCT) full scan is needed.
# 26 keeps atom.element, which has 21.
SAMPLE_LIMIT = 26
# Longer than this is prose (a post body, a description), not a category.
MAX_VALUE_CHARS = 40


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _value_comment(conn: sqlite3.Connection, table: str, column: str) -> str:
    """`-- values: ...` for a low-cardinality text column, or "" for the rest."""
    values = [
        str(r[0]) for r in conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL LIMIT {SAMPLE_LIMIT}"
        )
    ]
    if len(values) >= SAMPLE_LIMIT or any(len(v) > MAX_VALUE_CHARS for v in values):
        return ""
    return "  -- values: " + ", ".join(repr(v) for v in values)


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            # (line, trailing comment) - the comment has to come after the comma,
            # or the separator ends up inside it and the DDL reads as malformed.
            col_lines: list[tuple[str, str]] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                text_column = "CHAR" in (ctype or "").upper() or "TEXT" in (ctype or "").upper()
                comment = _value_comment(conn, t, name) if SAMPLE_VALUES and text_column else ""
                col_lines.append((line, comment))
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                # `to` is NULL when the FK targets the referenced table's implicit
                # primary key; emit the bare table reference in that case.
                ref = _q(fk[2]) + (f"({_q(fk[4])})" if fk[4] is not None else "")
                col_lines.append((f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {ref}", ""))
            last = len(col_lines) - 1
            parts.append("\n".join(
                f"{line}{',' if i < last else ''}{comment}"
                for i, (line, comment) in enumerate(col_lines)
            ))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
