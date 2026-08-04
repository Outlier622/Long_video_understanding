"""
Create the DynamoDB table used by VideoThinker clip workers.

Example:
    python .\src\cloud\setup_dynamodb_job_table.py `
      --table-name videothinker-clip-jobs `
      --region us-east-2
"""

from __future__ import annotations

import argparse

import boto3
from botocore.exceptions import ClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-name",
        default="videothinker-clip-jobs",
    )
    parser.add_argument("--region", default="us-east-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = boto3.client("dynamodb", region_name=args.region)

    try:
        response = client.describe_table(TableName=args.table_name)
        table = response["Table"]
        print("Table already exists.")
        print("Table:", table["TableName"])
        print("Status:", table["TableStatus"])
        print("ARN:", table["TableArn"])
        return
    except client.exceptions.ResourceNotFoundException:
        pass

    print("Creating DynamoDB table:", args.table_name)
    client.create_table(
        TableName=args.table_name,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "clip_id", "AttributeType": "N"},
        ],
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "clip_id", "KeyType": "RANGE"},
        ],
        Tags=[
            {"Key": "Project", "Value": "VideoThinker-LongVideo"},
            {"Key": "Purpose", "Value": "ClipJobState"},
        ],
    )

    waiter = client.get_waiter("table_exists")
    waiter.wait(TableName=args.table_name)

    table = client.describe_table(TableName=args.table_name)["Table"]
    print("Table created.")
    print("Table:", table["TableName"])
    print("Status:", table["TableStatus"])
    print("ARN:", table["TableArn"])


if __name__ == "__main__":
    main()