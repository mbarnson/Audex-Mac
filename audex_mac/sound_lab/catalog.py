"""Durable local catalog and blind public views for Audex Sound Lab."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SoundLabCatalog:
    """Own Sound Lab persistence behind one SQLite-backed interface."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_job(
        self,
        *,
        job_id: str,
        requested_brief: str,
        requested_count: int,
        model_repo: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, requested_brief, requested_count, model_repo,
                    state, revealed, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'planning', 0, ?, ?)
                """,
                (
                    job_id,
                    requested_brief,
                    requested_count,
                    model_repo,
                    _now(),
                    _now(),
                ),
            )

    def add_candidate(
        self,
        *,
        asset_id: str,
        job_id: str,
        blind_label: str,
        caption: str,
        difference: str,
        seed: int,
        recipe: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO candidates (
                    asset_id, job_id, blind_label, caption, difference_note,
                    seed, recipe, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    asset_id,
                    job_id,
                    blind_label,
                    caption,
                    difference,
                    seed,
                    recipe,
                    _now(),
                    _now(),
                ),
            )
            connection.execute(
                "UPDATE jobs SET state = 'rendering', updated_at = ? WHERE job_id = ?",
                (_now(), job_id),
            )

    def record_design_attempts(
        self,
        job_id: str,
        *,
        raw_attempts: tuple[str, ...],
        repair_used: bool,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET designer_attempts_json = ?, designer_repair_used = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(raw_attempts), int(repair_used), _now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab job: {job_id}")

    def job_diagnostics(self, job_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT designer_attempts_json, designer_repair_used, failure
                FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Sound Lab job: {job_id}")
        return {
            "designer_raw_attempts": json.loads(str(row[0])),
            "designer_repair_used": bool(row[1]),
            "failure": row[2],
        }

    def mark_candidate_ready(
        self,
        asset_id: str,
        *,
        wav_path: Path,
        duration_seconds: float,
        elapsed_seconds: float,
        seed_used: int,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE candidates
                SET state = 'ready', wav_path = ?, duration_seconds = ?,
                    elapsed_seconds = ?, seed = ?, updated_at = ?
                WHERE asset_id = ?
                """,
                (
                    str(Path(wav_path).resolve()),
                    duration_seconds,
                    elapsed_seconds,
                    seed_used,
                    _now(),
                    asset_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab asset: {asset_id}")

    def record_candidate_attempts(
        self,
        asset_id: str,
        *,
        attempts: tuple[Any, ...],
    ) -> None:
        payload = [
            {
                "seed": attempt.seed,
                "elapsed_seconds": attempt.elapsed_seconds,
                "frame_count": attempt.frame_count,
                "duration_seconds": attempt.duration_seconds,
                "reached_end_token": attempt.reached_end_token,
                "failures": list(attempt.failures),
            }
            for attempt in attempts
        ]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE candidates SET generation_attempts_json = ?, updated_at = ?
                WHERE asset_id = ?
                """,
                (json.dumps(payload, sort_keys=True), _now(), asset_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab asset: {asset_id}")

    def mark_candidate_generating(self, asset_id: str) -> None:
        self._set_candidate_state(asset_id, "generating")

    def mark_candidate_failed(self, asset_id: str, error: str) -> None:
        self._set_candidate_state(asset_id, "failed", failure=error)

    def finish_job(self, job_id: str, *, failed: bool = False, error: str = "") -> None:
        state = "failed" if failed else "complete"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET state = ?, failure = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (state, error.strip() or None, _now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab job: {job_id}")

    def record_preference(
        self,
        *,
        job_id: str,
        selected_label: str,
        rejected_labels: tuple[str, ...] = (),
        note: str = "",
    ) -> None:
        selected = selected_label.strip().upper()
        rejected = tuple(
            dict.fromkeys(label.strip().upper() for label in rejected_labels)
        )
        with self._connect() as connection:
            labels = {
                str(row[0])
                for row in connection.execute(
                    "SELECT blind_label FROM candidates WHERE job_id = ?", (job_id,)
                )
            }
            if selected not in labels:
                raise ValueError(f"unknown selected label for {job_id}: {selected}")
            unknown = set(rejected) - labels
            if unknown:
                raise ValueError(
                    f"unknown rejected labels for {job_id}: {sorted(unknown)}"
                )
            if selected in rejected:
                raise ValueError("selected candidate cannot also be rejected")
            connection.execute(
                """
                INSERT INTO preferences (
                    job_id, selected_label, rejected_labels_json, note, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    selected_label = excluded.selected_label,
                    rejected_labels_json = excluded.rejected_labels_json,
                    note = excluded.note,
                    created_at = excluded.created_at
                """,
                (job_id, selected, json.dumps(rejected), note.strip(), _now()),
            )

    def reveal_job(self, job_id: str) -> None:
        with self._connect() as connection:
            preference = connection.execute(
                "SELECT 1 FROM preferences WHERE job_id = ?", (job_id,)
            ).fetchone()
            if preference is None:
                raise ValueError(
                    "record a blind preference before revealing production details"
                )
            cursor = connection.execute(
                "UPDATE jobs SET revealed = 1, updated_at = ? WHERE job_id = ?",
                (_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab job: {job_id}")

    def audio_path(self, asset_id: str) -> Path:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT wav_path FROM candidates WHERE asset_id = ? AND state = 'ready'",
                (asset_id,),
            ).fetchone()
        if row is None or row[0] is None:
            raise KeyError(f"Sound Lab asset is not ready: {asset_id}")
        path = Path(str(row[0]))
        if not path.is_file():
            raise FileNotFoundError(f"Sound Lab WAV is missing: {path}")
        return path

    def public_snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            jobs = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC"
            ).fetchall()
            public_jobs: list[dict[str, Any]] = []
            for job in jobs:
                candidates = connection.execute(
                    """
                    SELECT * FROM candidates
                    WHERE job_id = ?
                    ORDER BY blind_label ASC
                    """,
                    (job["job_id"],),
                ).fetchall()
                public_candidates = [
                    self._public_candidate(candidate, revealed=bool(job["revealed"]))
                    for candidate in candidates
                ]
                preference = connection.execute(
                    "SELECT * FROM preferences WHERE job_id = ?",
                    (job["job_id"],),
                ).fetchone()
                public_job: dict[str, Any] = {
                    "job_id": job["job_id"],
                    "state": job["state"],
                    "requested_count": job["requested_count"],
                    "revealed": bool(job["revealed"]),
                    "created_at": job["created_at"],
                    "candidates": public_candidates,
                }
                if preference is not None:
                    public_job["preference"] = {
                        "selected_label": preference["selected_label"],
                        "rejected_labels": json.loads(
                            preference["rejected_labels_json"]
                        ),
                        "note": preference["note"],
                    }
                else:
                    public_job["preference"] = None
                public_jobs.append(public_job)
        return {"jobs": public_jobs}

    def _public_candidate(
        self,
        candidate: sqlite3.Row,
        *,
        revealed: bool,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "asset_id": candidate["asset_id"],
            "label": candidate["blind_label"],
            "state": candidate["state"],
            "duration_seconds": candidate["duration_seconds"],
            "elapsed_seconds": candidate["elapsed_seconds"],
            "audio_url": (
                f"/audio/{candidate['asset_id']}"
                if candidate["state"] == "ready"
                else None
            ),
        }
        if revealed:
            item.update(
                {
                    "caption": candidate["caption"],
                    "difference": candidate["difference_note"],
                    "seed": candidate["seed"],
                    "recipe": candidate["recipe"],
                    "generation_attempts": json.loads(
                        candidate["generation_attempts_json"]
                    ),
                }
            )
        return item

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    requested_brief TEXT NOT NULL,
                    requested_count INTEGER NOT NULL CHECK(requested_count BETWEEN 1 AND 5),
                    model_repo TEXT NOT NULL,
                    state TEXT NOT NULL,
                    failure TEXT,
                    designer_attempts_json TEXT NOT NULL DEFAULT '[]',
                    designer_repair_used INTEGER NOT NULL DEFAULT 0,
                    revealed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    asset_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    blind_label TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    difference_note TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    recipe TEXT NOT NULL,
                    state TEXT NOT NULL,
                    failure TEXT,
                    wav_path TEXT,
                    duration_seconds REAL,
                    elapsed_seconds REAL,
                    generation_attempts_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(job_id, blind_label)
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
                    selected_label TEXT NOT NULL,
                    rejected_labels_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """)
            _ensure_column(
                connection,
                "jobs",
                "designer_attempts_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            _ensure_column(
                connection,
                "jobs",
                "designer_repair_used",
                "INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                connection,
                "candidates",
                "generation_attempts_json",
                "TEXT NOT NULL DEFAULT '[]'",
            )

    def _set_candidate_state(
        self,
        asset_id: str,
        state: str,
        *,
        failure: str | None = None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE candidates SET state = ?, failure = ?, updated_at = ?
                WHERE asset_id = ?
                """,
                (state, failure, _now(), asset_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Sound Lab asset: {asset_id}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    existing = {
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
