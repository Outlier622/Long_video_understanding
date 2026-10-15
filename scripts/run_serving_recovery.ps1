param(
    [string]$PythonExe = 'python',
    [string]$OutputDirectory = '',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$testName = 'videothinker-recovery-test'
$created = $false
$stoppedServing = $false

function Invoke-Docker {
    param([string[]]$Arguments)
    $commandOutput = & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed: $($Arguments -join ' ')"
    }
    return $commandOutput
}

Push-Location $projectRoot
try {
    $existingTest = Invoke-Docker -Arguments @('ps', '-aq', '--filter', "name=^/$testName`$")
    if ($existingTest) { throw 'An existing recovery test container must be cleaned up first.' }
    if (-not $OutputDirectory) {
        $OutputDirectory = Join-Path $projectRoot ('evaluation/runs/serving_recovery_' + (Get-Date -Format yyyyMMdd_HHmmss))
    }
    elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
        $OutputDirectory = Join-Path $projectRoot $OutputDirectory
    }
    $runDir = [IO.Path]::GetFullPath($OutputDirectory)
    New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null
    $faultDir = Join-Path $runDir 'faults'
    New-Item -ItemType Directory -Path $faultDir -ErrorAction Stop | Out-Null
    New-Item -ItemType File -Path (Join-Path $faultDir 'initialization.fail') | Out-Null
    $modelDir = if ($env:VIDEO_MODEL_DIR) { $env:VIDEO_MODEL_DIR } else { 'D:/projects/VideoThinker-R1-3B' }
    $dataDir = if ($env:VIDEO_DATA_DIR) { $env:VIDEO_DATA_DIR } else { 'D:/projects/longvideo' }
    if (-not (Test-Path -LiteralPath $modelDir -PathType Container)) { throw 'Model directory is missing.' }
    if (-not (Test-Path -LiteralPath $dataDir -PathType Container)) { throw 'Data directory is missing.' }
    if (-not $SkipBuild) {
        Invoke-Docker -Arguments @('build', '-f', 'Dockerfile.serving', '-t', 'videothinker-serving:local', '.') | Out-Host
    }
    $runningServing = Invoke-Docker -Arguments @('ps', '-q', '--filter', 'name=^/videothinker-serving$')
    if ($runningServing) {
        Invoke-Docker -Arguments @('compose', '-f', 'compose.serving.yml', 'stop') | Out-Host
        $stoppedServing = $true
    }
    $launcher = Join-Path $PSScriptRoot 'recovery_fault_runner.py'
    Invoke-Docker -Arguments @(
        'run', '-d', '--name', $testName, '--restart', 'unless-stopped',
        '--gpus', 'all', '--shm-size', '1g', '-p', '127.0.0.1:8001:8000',
        '-e', 'MODEL_PATH=/model', '-e', 'VIDEO_DATA_ROOT=/data',
        '--mount', "type=bind,source=$modelDir,target=/model,readonly",
        '--mount', "type=bind,source=$dataDir,target=/data,readonly",
        '--mount', "type=bind,source=$launcher,target=/verification/recovery_fault_runner.py,readonly",
        '--mount', "type=bind,source=$faultDir,target=/faults",
        '--entrypoint', 'python', 'videothinker-serving:local', '/verification/recovery_fault_runner.py'
    ) | Out-Host
    $created = $true
    & $PythonExe (Join-Path $PSScriptRoot 'verify_serving_recovery.py') --fault-dir $faultDir --output-dir (Join-Path $runDir 'http_checks')
    if ($LASTEXITCODE -ne 0) { throw 'HTTP/GPU recovery acceptance failed.' }
    $restartCount = Invoke-Docker -Arguments @('inspect', $testName, '--format', '{{.RestartCount}}')
    $restartCount | Set-Content -LiteralPath (Join-Path $runDir 'restart_count.txt')
    if ([int]$restartCount -ne 2) { throw "Expected exactly two automatic restarts, got $restartCount" }
}
finally {
    try {
        if ($created) {
            & docker logs $testName 2>&1 | Set-Content -LiteralPath (Join-Path $runDir 'container.log')
            Invoke-Docker -Arguments @('stop', $testName) | Out-Host
            Invoke-Docker -Arguments @('rm', $testName) | Out-Host
        }
    }
    finally {
        try {
            if ($stoppedServing) {
                Invoke-Docker -Arguments @('compose', '-f', 'compose.serving.yml', 'up', '-d') | Out-Host
            }
        }
        finally { Pop-Location }
    }
}
