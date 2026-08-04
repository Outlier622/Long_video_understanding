"""
AWS Batch container entrypoint for the sharded VideoThinker worker.

Required environment variables:
    MODEL_S3_PREFIX
    MANIFEST_S3_URI
    OUTPUT_S3_PREFIX
    DYNAMODB_TABLE
    RUN_ID

Optional environment variables:
    NUM_SHARDS  # defaults to 1

Shard selection:
    AWS_BATCH_JOB_ARRAY_INDEX is used automatically for an AWS Batch array job.
    SHARD_INDEX can override it for local testing.

The container deliberately does not accept an AWS CLI profile. In AWS Batch,
boto3 uses the job role. For local Docker testing, mount ~/.aws and set
AWS_PROFILE in the container environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


WORKER_PATH = Path("/app/src/workers/inference_worker_v5.py")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value.strip()


def optional_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_bool(name: str, default: bool = False) -> bool:
    value = optional_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = optional_env(name)
    if value is None:
        return default
    return int(value)


def resolve_shard_index() -> int:
    explicit = optional_env("SHARD_INDEX")
    if explicit is not None:
        return int(explicit)

    array_index = optional_env("AWS_BATCH_JOB_ARRAY_INDEX")
    if array_index is not None:
        return int(array_index)

    return 0


def append_optional(
    command: List[str],
    flag: str,
    value: Optional[str],
) -> None:
    if value is not None and value != "":
        command.extend([flag, value])


def main() -> None:
    if not WORKER_PATH.exists():
        raise FileNotFoundError(f"Worker script is missing: {WORKER_PATH}")

    model_s3_prefix = require_env("MODEL_S3_PREFIX")
    manifest_s3_uri = require_env("MANIFEST_S3_URI")
    output_s3_prefix = require_env("OUTPUT_S3_PREFIX")
    dynamodb_table = require_env("DYNAMODB_TABLE")
    run_id = require_env("RUN_ID")
    num_shards = env_int("NUM_SHARDS", 1)
    shard_index = resolve_shard_index()

    if num_shards < 1:
        raise ValueError("NUM_SHARDS must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"Shard index {shard_index} is outside [0, {num_shards - 1}]."
        )

    region = (
        optional_env("AWS_REGION")
        or optional_env("AWS_DEFAULT_REGION")
        or "us-east-2"
    )

    model_path = optional_env("MODEL_PATH", "/work/model")
    output_dir = optional_env("OUTPUT_DIR", "/work/output")
    cache_dir = optional_env("CACHE_DIR", "/work/cache")

    command = [
        sys.executable,
        str(WORKER_PATH),
        "--model-path",
        str(model_path),
        "--model-s3-prefix",
        model_s3_prefix,
        "--manifest-path",
        manifest_s3_uri,
        "--output-dir",
        str(output_dir),
        "--output-s3-prefix",
        output_s3_prefix,
        "--cache-dir",
        str(cache_dir),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
        "--dtype",
        optional_env("DTYPE", "bf16") or "bf16",
        "--state-backend",
        "dynamodb",
        "--dynamodb-table",
        dynamodb_table,
        "--aws-region",
        region,
        "--run-id",
        run_id,
        "--lease-seconds",
        str(env_int("LEASE_SECONDS", 7200)),
        "--max-attempts",
        str(env_int("MAX_ATTEMPTS", 3)),
        "--max-new-tokens",
        str(env_int("MAX_NEW_TOKENS", 500)),
        "--max-keyframes-per-clip",
        str(env_int("MAX_KEYFRAMES_PER_CLIP", 12)),
    ]

    append_optional(command, "--clip-ids", optional_env("CLIP_IDS"))
    append_optional(command, "--worker-id", optional_env("WORKER_ID"))

    if env_bool("VERIFY_MODEL_SHA256"):
        command.append("--verify-model-sha256")
    if env_bool("USE_FIRST_FRAMES"):
        command.append("--use-first-frames")
    if env_bool("DRY_RUN"):
        command.append("--dry-run")

    print("AWS Batch entrypoint")
    print("Run ID:", run_id)
    print("Shard:", f"{shard_index}/{num_shards}")
    print("Region:", region)
    print("Dry run:", env_bool("DRY_RUN"))
    print("Worker command:")
    print(" ".join(command))

    os.execv(command[0], command)


if __name__ == "__main__":
    main()