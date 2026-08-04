r"""
Upload a local Hugging Face model directory to a deterministic S3 bundle.

The bundle contains:
    <model-prefix>/bundle_manifest.json
    <model-prefix>/files/<relative model path>

The manifest records file size and SHA-256 so an AWS Batch worker can resume
partial downloads and optionally verify integrity.

Example:
    python .\src\cloud\prepare_s3_model_bundle.py `
      --model-dir "D:\projects\VideoThinker-R1-3B" `
      --bucket "videothinker-longvideo-535534157295-us-east-2" `
      --model-id "videothinker-r1-3b" `
      --region "us-east-2" `
      --profile "videothinker-dev"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.s3_artifact_store import S3ArtifactStore


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    "evaluation",
    "src",
    "tests",
    "outputs",
    "episodes",
    "logs",
    "notebooks",
    ".venv",
    "venv",
    "env",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--prefix", default="models")
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directory name to exclude; may be repeated.",
    )
    parser.add_argument(
        "--force-upload",
        action="store_true",
        help="Upload even when an object with the same size already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the bundle plan without uploading.",
    )
    parser.add_argument(
        "--local-manifest-path",
        default=None,
        help="Optional path for the generated local bundle manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(
    root: Path,
    excluded_dirs: Set[str],
) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in excluded_dirs for part in relative.parts[:-1]):
            continue
        yield path


def human_gib(size_bytes: int) -> str:
    return f"{size_bytes / (1024 ** 3):.3f} GiB"


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    excluded_dirs = DEFAULT_EXCLUDED_DIRS | set(args.exclude_dir)
    files = list(iter_files(model_dir, excluded_dirs))
    if not files:
        raise ValueError(f"No model files found under: {model_dir}")

    base_prefix = (
        f"s3://{args.bucket}/"
        f"{args.prefix.strip('/')}/{args.model_id.strip('/')}"
    )
    local_manifest_path = (
        Path(args.local_manifest_path)
        if args.local_manifest_path
        else model_dir / f"{args.model_id}_bundle_manifest.json"
    )

    store = S3ArtifactStore(
        region_name=args.region,
        profile_name=args.profile,
    )

    entries: List[Dict] = []
    total_bytes = 0
    uploaded_count = 0
    skipped_count = 0

    print("Scanning and hashing model files...")
    for index, local_path in enumerate(files, start=1):
        relative_path = local_path.relative_to(model_dir).as_posix()
        size_bytes = local_path.stat().st_size
        sha256 = sha256_file(local_path)
        object_uri = f"{base_prefix}/files/{relative_path}"

        entry = {
            "relative_path": relative_path,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "s3_uri": object_uri,
        }
        entries.append(entry)
        total_bytes += size_bytes

        print(
            f"[{index}/{len(files)}] "
            f"{relative_path} ({size_bytes} bytes)"
        )

        if args.dry_run:
            continue

        remote_size = store.size(object_uri)
        if (
            not args.force_upload
            and remote_size is not None
            and remote_size == size_bytes
        ):
            skipped_count += 1
            continue

        content_type = (
            mimetypes.guess_type(local_path.name)[0]
            or "application/octet-stream"
        )
        store.upload_file(
            local_path,
            object_uri,
            content_type=content_type,
            metadata={"sha256": sha256},
        )
        uploaded_count += 1
        print("Uploaded:", object_uri)

    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "source_model_dir": str(model_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_s3_prefix": base_prefix,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024 ** 3), 6),
        "excluded_directory_names": sorted(excluded_dirs),
        "files": entries,
    }

    local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    local_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest_uri = f"{base_prefix}/bundle_manifest.json"
    if not args.dry_run:
        store.upload_file(
            local_manifest_path,
            manifest_uri,
            content_type="application/json",
        )

    print("\nModel bundle preparation complete.")
    print("Files:", len(entries))
    print("Total size:", human_gib(total_bytes))
    print("Uploaded files:", uploaded_count)
    print("Existing same-size files skipped:", skipped_count)
    print("Local manifest:", local_manifest_path)
    print("S3 model prefix:", base_prefix)
    print("S3 manifest:", manifest_uri)
    if args.dry_run:
        print("Dry run: no S3 objects were written.")


if __name__ == "__main__":
    main()