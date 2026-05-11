"""Results store: per-run RunRecord, JSONL append, GCS sync.

Layout (canonical = GCS):
    gs://<bucket>/<method>/<run-uuid>/run_record.json
    gs://<bucket>/<method>/<run-uuid>/training_log.jsonl
    gs://<bucket>/<method>/<run-uuid>/trajectories.jsonl
    gs://<bucket>/<method>/<run-uuid>/checkpoints/...

Local mirror under `results/runs/<run-uuid>/` lives in the same shape so
`gsutil rsync` (or our `gcs.upload_dir`) preserves directory structure.

Concurrent-write safety: each run owns its own subdirectory. There is no
shared global file. To "query all runs," glob `*/*/run_record.json`.
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as cfg
from . import gcs

log = logging.getLogger("redteam_rlhf.results")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(s: str) -> str:
    """Make a filesystem-safe slug (alnum + dash)."""
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    return "".join(c if c in keep else "-" for c in s).strip("-")[:48] or "run"


@dataclass
class RunRecord:
    """The reproducibility-critical record for one run.

    Written once at start (`status="started"`) and re-written at end with
    `status="ok"|"failed"` and `final_metrics` populated.
    """
    run_id: str
    run_name: str
    method: str
    timestamp_started: str
    timestamp_finished: str | None
    status: str  # "started" | "ok" | "failed"

    git_sha: str
    git_dirty: bool
    config: dict
    config_hash: str
    seed: int

    hostname: str
    gpu_name: str | None
    cuda_version: str | None
    python_version: str
    platform: str

    final_metrics: dict = field(default_factory=dict)
    artifacts_dir: str = ""  # local path, populated by ResultsLogger

    @classmethod
    def new(
        cls,
        method: str,
        run_name: str | None,
        config: dict,
        env: dict,
    ) -> "RunRecord":
        rid = uuid.uuid4().hex[:12]
        name = _slug(run_name) if run_name else f"{method}-{rid}"
        return cls(
            run_id=rid,
            run_name=name,
            method=method,
            timestamp_started=_utc_now_iso(),
            timestamp_finished=None,
            status="started",
            git_sha=env["git_sha"],
            git_dirty=env["git_dirty"],
            config=config,
            config_hash=cfg.hash_config(config)[:16],
            seed=int(config.get("seed", 42)),
            hostname=socket.gethostname(),
            gpu_name=env.get("gpu_name"),
            cuda_version=env.get("cuda_version"),
            python_version=env["python_version"],
            platform=env["platform"],
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class ResultsLogger:
    """Owns the per-run output directory and any append-only files inside it.

    Public API:
      - write_initial(): create the dir, drop the initial run_record.json
      - log_jsonl(name, record): append one JSON object to <name>.jsonl
      - artifact_path(name): return a path inside the run dir
      - finalize(status, final_metrics): rewrite run_record.json and sync to GCS
    """

    def __init__(
        self,
        record: RunRecord,
        output_dir: Path,
        gcs_bucket: str | None,
    ):
        self.record = record
        self.output_dir = Path(output_dir) / record.run_id
        self.gcs_bucket = gcs_bucket  # e.g. "gs://results_043026"
        self.record.artifacts_dir = str(self.output_dir)

    # ── filesystem ──────────────────────────────────────────────────────────
    def write_initial(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_record()
        # Persist env-pip-freeze separately so the record file stays small-ish.
        if "pip_freeze" in self.record.config.get("_env", {}):
            (self.output_dir / "pip_freeze.txt").write_text(
                "\n".join(self.record.config["_env"]["pip_freeze"])
            )
        log.info("Run dir: %s", self.output_dir)

    def _write_record(self) -> None:
        (self.output_dir / "run_record.json").write_text(self.record.to_json())

    def log_jsonl(self, name: str, payload: dict) -> None:
        """Append one JSON line to <run_dir>/<name>.jsonl."""
        path = self.output_dir / f"{name}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def artifact_path(self, *parts: str) -> Path:
        p = self.output_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── lifecycle ───────────────────────────────────────────────────────────
    def finalize(self, status: str, final_metrics: dict[str, Any]) -> None:
        self.record.status = status
        self.record.final_metrics = final_metrics
        self.record.timestamp_finished = _utc_now_iso()
        self._write_record()
        log.info("Run %s finished: status=%s metrics=%s",
                 self.record.run_id, status, final_metrics)

        if self.gcs_bucket:
            try:
                dest = f"{self.gcs_bucket.rstrip('/')}/{self.record.method}/{self.record.run_id}"
                n = gcs.upload_dir(self.output_dir, dest)
                log.info("Synced %d files to %s", n, dest)
            except Exception as e:
                log.exception("GCS sync failed (artifacts remain local): %s", e)
