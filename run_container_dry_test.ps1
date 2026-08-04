param(
    [string]$Image = "videothinker-worker:v1",
    [string]$Profile = "videothinker-dev",
    [string]$Region = "us-east-2",
    [string]$RunId = "ep02-container-dry-run"
)

$ErrorActionPreference = "Stop"

$AwsDirectory = Join-Path $HOME ".aws"

if (-not (Test-Path $AwsDirectory)) {
    throw "AWS configuration directory not found: $AwsDirectory"
}

docker run --rm `
  -e "AWS_PROFILE=$Profile" `
  -e "AWS_REGION=$Region" `
  -e "MODEL_S3_PREFIX=s3://videothinker-longvideo-535534157295-us-east-2/models/videothinker-r1-3b" `
  -e "MANIFEST_S3_URI=s3://videothinker-longvideo-535534157295-us-east-2/runs/ep02-bf16-cloud-v1/inputs/manifest/clip_manifest_flow_cloud.jsonl" `
  -e "OUTPUT_S3_PREFIX=s3://videothinker-longvideo-535534157295-us-east-2/runs/$RunId/outputs" `
  -e "DYNAMODB_TABLE=videothinker-clip-jobs" `
  -e "RUN_ID=$RunId" `
  -e "NUM_SHARDS=4" `
  -e "SHARD_INDEX=0" `
  -e "CLIP_IDS=1,3" `
  -e "DTYPE=bf16" `
  -e "DRY_RUN=true" `
  -v "${AwsDirectory}:/root/.aws:rw" `
  $Image