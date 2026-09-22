"""SQLite engine and session helpers.

One engine per database path so tests can point at a temp file without
leaking into the real ``data/visualresearcher.db``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from . import models  # noqa: F401  (import registers the tables on SQLModel.metadata)
from .logging_setup import get_logger

__all__ = ["get_engine", "init_db", "session_scope", "reset_engines"]

log = get_logger("db")
#: One engine per database path. Guarded by a lock: the worker pool runs
#: jobs in threads, so two of them can ask for an engine at the same
#: moment, and a reset can land while another thread is still creating one.
_ENGINES: dict[str, object] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(db_path: Path, *, echo: bool = False):
    key = str(Path(db_path).resolve())
    with _ENGINES_LOCK:
        engine = _ENGINES.get(key)
    if engine is not None:
        return engine

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{key}",
        echo=echo,
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):  # pragma: no cover - driver callback
        cur = dbapi_conn.cursor()
        # WAL lets the worker write while the web UI reads (§19).
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    with _ENGINES_LOCK:
        # Another thread may have built one while we were not holding the
        # lock. Theirs wins, and ours is discarded rather than leaked.
        existing = _ENGINES.get(key)
        if existing is not None:
            engine.dispose()
            return existing
        _ENGINES[key] = engine
    return engine


def init_db(db_path: Path) -> None:
    engine = get_engine(db_path)
    SQLModel.metadata.create_all(engine)
    log.debug("database ready at %s", db_path)


@contextmanager
def session_scope(db_path: Path) -> Iterator[Session]:
    """Commit on success, roll back on error. Every write goes through here."""
    engine = get_engine(db_path)
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engines() -> None:
    """Drop cached engines. Tests call this between temp databases.

    Snapshots under the lock before disposing. Iterating the live dict raised
    ``dictionary changed size during iteration`` when a worker thread created
    an engine mid-reset -- which a threaded worker pool makes easy to hit.
    """
    with _ENGINES_LOCK:
        engines = list(_ENGINES.values())
        _ENGINES.clear()
    for engine in engines:
        try:
            engine.dispose()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - best effort
            pass
