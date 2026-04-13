param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [string]$Image = "",
    [string]$DataDir = "",
    [string]$Output = "backend/training_ct/predictions",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Image) -and [string]::IsNullOrWhiteSpace($DataDir)) {
    throw "Please provide either -Image or -DataDir."
}

if (-not [string]::IsNullOrWhiteSpace($Image) -and -not [string]::IsNullOrWhiteSpace($DataDir)) {
    throw "Please provide only one of -Image or -DataDir, not both."
}

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

if (-not (Test-Path -LiteralPath $Config)) {
    throw "Config not found: $Config"
}

if (-not [string]::IsNullOrWhiteSpace($Image) -and -not (Test-Path -LiteralPath $Image)) {
    throw "Image not found: $Image"
}

if (-not [string]::IsNullOrWhiteSpace($DataDir) -and -not (Test-Path -LiteralPath $DataDir)) {
    throw "DataDir not found: $DataDir"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
Set-Location $repoRoot

$argsList = @(
    "backend/training_ct/predict_ct.py",
    "--checkpoint", $Checkpoint,
    "--config", $Config,
    "--output", $Output,
    "--device", $Device
)

if (-not [string]::IsNullOrWhiteSpace($Image)) {
    $argsList += @("--image", $Image)
}
else {
    $argsList += @("--data_dir", $DataDir)
}

Write-Host "Running: python $($argsList -join ' ')" -ForegroundColor Cyan
& python @argsList

if ($LASTEXITCODE -ne 0) {
    throw "Prediction failed with exit code: $LASTEXITCODE"
}

Write-Host "Done. Results saved to: $Output" -ForegroundColor Green
