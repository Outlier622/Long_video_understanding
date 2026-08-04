"""
Create a private S3 bucket for VideoThinker artifacts.

Example:
    python .\src\cloud\setup_s3_bucket.py `
      --bucket videothinker-longvideo-535534157295-us-east-2 `
      --region us-east-2 `
      --profile videothinker-dev
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import ClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = boto3.Session(
        profile_name=args.profile,
        region_name=args.region,
    )
    client = session.client("s3")

    try:
        client.head_bucket(Bucket=args.bucket)
        print("Bucket already exists and is accessible:", args.bucket)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

        kwargs = {"Bucket": args.bucket}
        if args.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": args.region
            }

        print("Creating bucket:", args.bucket)
        client.create_bucket(**kwargs)
        waiter = client.get_waiter("bucket_exists")
        waiter.wait(Bucket=args.bucket)
        print("Bucket created:", args.bucket)

    client.put_public_access_block(
        Bucket=args.bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )

    client.put_bucket_encryption(
        Bucket=args.bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256"
                    },
                }
            ]
        },
    )

    client.put_bucket_tagging(
        Bucket=args.bucket,
        Tagging={
            "TagSet": [
                {"Key": "Project", "Value": "VideoThinker-LongVideo"},
                {"Key": "Purpose", "Value": "InferenceArtifacts"},
            ]
        },
    )

    region = client.get_bucket_location(Bucket=args.bucket).get(
        "LocationConstraint"
    ) or "us-east-1"

    print("Public access: blocked")
    print("Default encryption: AES256")
    print("Region:", region)
    print("S3 URI:", f"s3://{args.bucket}/")


if __name__ == "__main__":
    main()