"""
S3 artifact utilities shared by VideoThinker cloud preparation and workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def is_s3_uri(value: str) -> bool:
    return str(value).lower().startswith("s3://")


def parse_s3_uri(uri: str) -> S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")

    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI does not contain an object key: {uri}")

    return S3Location(bucket=parsed.netloc, key=key)


def join_s3_uri(prefix: str, *parts: str) -> str:
    location = parse_s3_uri(prefix.rstrip("/") + "/placeholder")
    base_key = location.key.rsplit("/", 1)[0]
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    key = "/".join([part for part in [base_key, *clean_parts] if part])
    return f"s3://{location.bucket}/{key}"


class S3ArtifactStore:
    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        profile_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self.client = session.client("s3", endpoint_url=endpoint_url)

    def head(self, uri: str) -> Optional[Dict]:
        location = parse_s3_uri(uri)
        try:
            return self.client.head_object(
                Bucket=location.bucket,
                Key=location.key,
            )
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def exists(self, uri: str) -> bool:
        return self.head(uri) is not None

    def size(self, uri: str) -> Optional[int]:
        metadata = self.head(uri)
        if metadata is None:
            return None
        return int(metadata["ContentLength"])

    def upload_file(
        self,
        local_path: Path,
        uri: str,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(local_path)

        location = parse_s3_uri(uri)
        extra_args: Dict[str, object] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        if metadata:
            extra_args["Metadata"] = metadata

        kwargs = {
            "Filename": str(local_path),
            "Bucket": location.bucket,
            "Key": location.key,
        }
        if extra_args:
            kwargs["ExtraArgs"] = extra_args

        self.client.upload_file(**kwargs)

    def download_file(
        self,
        uri: str,
        local_path: Path,
    ) -> Path:
        location = parse_s3_uri(uri)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(
            location.bucket,
            location.key,
            str(local_path),
        )
        return local_path

    def download_if_exists(
        self,
        uri: str,
        local_path: Path,
    ) -> bool:
        if not self.exists(uri):
            return False
        self.download_file(uri, local_path)
        return True

    def iter_prefix(self, prefix_uri: str) -> Iterable[str]:
        location = parse_s3_uri(prefix_uri.rstrip("/") + "/placeholder")
        prefix = location.key.rsplit("/", 1)[0].rstrip("/") + "/"
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=location.bucket,
            Prefix=prefix,
        ):
            for item in page.get("Contents", []):
                yield f"s3://{location.bucket}/{item['Key']}"