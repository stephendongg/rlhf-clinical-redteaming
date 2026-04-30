"""Thin Google Cloud Storage upload wrapper.

Uses the official `google-cloud-storage` Python client (justified extra dep
per project spec — needed to sync per-run artifacts to gs://results_043026/).
Auth in Colab: `from google.colab import auth; auth.authenticate_user()`.
Auth elsewhere: `gcloud auth application-default login`.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("redteam_rlhf.gcs")


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split `gs://bucket/some/prefix` into (`bucket`, `some/prefix`)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    rest = uri[5:]
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    return bucket, prefix.rstrip("/")


def upload_dir(local_dir: Path, gs_uri: str) -> int:
    """Recursively upload `local_dir` under `gs_uri`.

    Returns the number of files uploaded. Skips empty files. Raises on
    upload failure for any individual file.
    """
    from google.cloud import storage

    bucket_name, prefix = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    n = 0
    local_dir = Path(local_dir)
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        log.info("GCS upload: %s -> gs://%s/%s", p, bucket_name, blob_name)
        bucket.blob(blob_name).upload_from_filename(str(p))
        n += 1
    return n


def upload_file(local_path: Path, gs_uri: str) -> None:
    """Upload a single file. `gs_uri` is the *full* destination URI."""
    from google.cloud import storage

    bucket_name, prefix = parse_gs_uri(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    log.info("GCS upload: %s -> gs://%s/%s", local_path, bucket_name, prefix)
    bucket.blob(prefix).upload_from_filename(str(local_path))
