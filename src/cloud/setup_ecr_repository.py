"""
Create or inspect the private ECR repository for the VideoThinker worker.
"""

from __future__ import annotations

import argparse
import json

import boto3
from botocore.exceptions import ClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-name",
        default="videothinker-worker",
    )
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = boto3.Session(
        profile_name=args.profile,
        region_name=args.region,
    )
    client = session.client("ecr")

    try:
        response = client.describe_repositories(
            repositoryNames=[args.repository_name]
        )
        repository = response["repositories"][0]
        print("Repository already exists.")
    except client.exceptions.RepositoryNotFoundException:
        response = client.create_repository(
            repositoryName=args.repository_name,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
            tags=[
                {"Key": "Project", "Value": "VideoThinker-LongVideo"},
                {"Key": "Purpose", "Value": "InferenceWorker"},
            ],
        )
        repository = response["repository"]
        print("Repository created.")

    lifecycle_policy = {
        "rules": [
            {
                "rulePriority": 1,
                "description": "Retain the newest 10 images",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 10,
                },
                "action": {"type": "expire"},
            }
        ]
    }
    client.put_lifecycle_policy(
        repositoryName=args.repository_name,
        lifecyclePolicyText=json.dumps(lifecycle_policy),
    )

    print("Repository:", repository["repositoryName"])
    print("URI:", repository["repositoryUri"])
    print("ARN:", repository["repositoryArn"])
    print("Image scan on push: enabled")
    print("Lifecycle: retain newest 10 images")


if __name__ == "__main__":
    main()