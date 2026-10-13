from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class Registry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    def close(self) -> None:
        self._conn.close()
