import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, label TEXT, started TEXT, seconds REAL,
    input_tokens INTEGER, output_tokens INTEGER, cost REAL);
CREATE TABLE IF NOT EXISTS spans (
    run_id TEXT, span_id TEXT PRIMARY KEY, parent_id TEXT, name TEXT, kind TEXT,
    ms REAL, error TEXT, input_tokens INTEGER, output_tokens INTEGER, cost REAL);
"""


def save(tracer, db_path, label):
    """Append one run and its spans. Saving the same run twice replaces it."""
    roots = tracer.roots()
    if not roots:
        return None
    run_id = roots[0].span_id
    tokens_in, tokens_out = tracer.tokens()
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(roots[0].start_ns / 1e9))
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?)",
                   (run_id, label, started, tracer.seconds(), tokens_in, tokens_out,
                    tracer.cost()))
        db.executemany("INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
                       [(run_id, s.span_id, s.parent_id, s.name, s.kind, s.ms, s.error,
                         s.input_tokens, s.output_tokens, s.cost) for s in tracer.ordered()])
    return run_id


def _query(db_path, sql, args=()):
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        return db.execute(sql, args).fetchall()


def recent(db_path, limit=10):
    """The last runs, newest first: label, when, tokens, cost, seconds."""
    return _query(db_path, "SELECT label, started, input_tokens, output_tokens, cost, seconds "
                           "FROM runs ORDER BY started DESC LIMIT ?", (limit,))


def totals(db_path):
    """How many runs are stored, and what they add up to."""
    rows = _query(db_path, "SELECT COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost) "
                           "FROM runs")
    runs, tokens_in, tokens_out, cost = rows[0]
    return runs, tokens_in or 0, tokens_out or 0, cost or 0.0


def slowest(db_path, limit=5):
    """The slowest spans ever recorded, whichever run they came from."""
    return _query(db_path, "SELECT name, kind, ms FROM spans ORDER BY ms DESC LIMIT ?", (limit,))
