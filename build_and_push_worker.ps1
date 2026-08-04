param(
    [string]$Profile = "videothinker-dev",
    [string]$Region = "us-east-2",
    [string]$AccountId = "535534157295",
    [string]$RepositoryName = "videothinker-worker",
    [string]$Tag = "v1",
    [switch]$SkipPush,
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$CommandName failed with exit code $LASTEXITCODE."
    }
}

$Registry = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$RepositoryUri = "$Registry/$RepositoryName"
$RemoteImage = "${RepositoryUri}:$Tag"
$LocalImage = "videothinker-worker:$Tag"

Write-Host "Checking Docker..."
docker version
Assert-NativeSuccess "docker version"

$BuildArgs = @(
    "build",
    "--file", ".\Dockerfile.worker",
    "--tag", $LocalImage
)

if ($NoCache) {
    $BuildArgs += "--no-cache"
}

$BuildArgs += "."

Write-Host "`nBuilding local image: $LocalImage"
& docker @BuildArgs
Assert-NativeSuccess "docker build"

Write-Host "`nVerifying local image..."
docker image inspect $LocalImage *> $null
Assert-NativeSuccess "docker image inspect"

if ($SkipPush) {
    Write-Host "`nImage built and verified locally."
    Write-Host "Local image: $LocalImage"
    exit 0
}

Write-Host "`nLogging Docker into ECR..."
$Password = aws ecr get-login-password `
    --region $Region `
    --profile $Profile
Assert-NativeSuccess "aws ecr get-login-password"

$Password | docker login `
    --username AWS `
    --password-stdin $Registry
Assert-NativeSuccess "docker login"

Write-Host "`nTagging image: $RemoteImage"
docker tag $LocalImage $RemoteImage
Assert-NativeSuccess "docker tag"

Write-Host "`nPushing image..."
docker push $RemoteImage
Assert-NativeSuccess "docker push"

Write-Host "`nVerifying remote image..."
aws ecr describe-images `
    --repository-name $RepositoryName `
    --image-ids "imageTag=$Tag" `
    --region $Region `
    --profile $Profile *> $null
Assert-NativeSuccess "aws ecr describe-images"

Write-Host "`nBuild and push completed."
Write-Host "Image URI: $RemoteImage"