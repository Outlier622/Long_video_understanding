"""
Download and validate a VideoThinker model bundle from S3 without loading it.

Example:
    python .\src\cloud\materialize_s3_model_bundle.py `
      --model-s3-prefix "s3://bucket/models/videothinker-r1-3b" `
      --output-dir "D:\projects\longvideo\model_cache\videothinker-r1-3b" `
      --region "us-east-2" `
      --profile "videothinker-dev" `
      --verify-sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.s3_artifact_store import (
    S3ArtifactStore,
    join_s3_uri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--verify-sha256", action="store_true")
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store = S3ArtifactStore(
        region_name=args.region,
        profile_name=args.profile,
    )

    manifest_uri = join_s3_uri(
        args.model_s3_prefix,
        "bundle_manifest.json",
    )
    local_manifest = output_dir / ".bundle_manifest.json"

    print("Downloading manifest:", manifest_uri)
    store.download_file(manifest_uri, local_manifest)

    manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
    files = manifest.get("files") or []
    if not files:
        raise ValueError("Model bundle manifest contains no files.")

    downloaded = 0
    reused = 0

    for index, item in enumerate(files, start=1):
        relative_path = Path(str(item["relative_path"]))
        expected_size = int(item["size_bytes"])
        expected_sha256 = str(item.get("sha256") or "")
        source_uri = str(item["s3_uri"])
        local_path = output_dir / relative_path

        valid = (
            local_path.exists()
            and local_path.is_file()
            and local_path.stat().st_size == expected_size
        )

        if valid and args.verify_sha256 and expected_sha256:
            valid = sha256_file(local_path) == expected_sha256

        if valid:
            reused += 1
            print(f"[{index}/{len(files)}] Reused:", relative_path)
            continue

        print(f"[{index}/{len(files)}] Downloading:", source_uri)
        store.download_file(source_uri, local_path)

        if local_path.stat().st_size != expected_size:
            raise IOError(f"Wrong file size after download: {local_path}")

        if (
            args.verify_sha256
            and expected_sha256
            and sha256_file(local_path) != expected_sha256
        ):
            raise IOError(f"SHA-256 mismatch: {local_path}")

        downloaded += 1

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Downloaded bundle has no config.json: {output_dir}"
        )

    print("\nModel bundle materialization passed.")
    print("Model ID:", manifest.get("model_id"))
    print("Files:", len(files))
    print("Total GiB:", manifest.get("total_gib"))
    print("Downloaded:", downloaded)
    print("Reused:", reused)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()