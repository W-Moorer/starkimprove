param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [switch]$SkipBuild,
    [double]$Dt = 0.001,
    [double]$EndTime = 2.0
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ExePath = Join-Path $RepoRoot "$BuildDir\examples\$Config\examples.exe"
$OutputBase = Join-Path $RepoRoot "output\paper_experiments"

if (-not $SkipBuild) {
    cmake --build (Join-Path $RepoRoot $BuildDir) --config $Config --target examples
}

Write-Host "[1/4] Running STARK exp6..."
& $ExePath exp6

Write-Host "[2/4] Running PyChrono double pendulum..."
conda run -n chrono-baseline python (Join-Path $PSScriptRoot "pychrono_double_pendulum.py") --dt $Dt --end-time $EndTime --output-base $OutputBase

Write-Host "[3/4] Comparing curves..."
python (Join-Path $PSScriptRoot "compare_double_pendulum_curves.py") `
    --stark-dir (Join-Path $OutputBase "exp6_double_pendulum_stark") `
    --chrono-dir (Join-Path $OutputBase "exp6_double_pendulum_pychrono")

Write-Host "[4/4] Done. Key outputs:"
Write-Host "  - $OutputBase\exp6_double_pendulum_stark"
Write-Host "  - $OutputBase\exp6_double_pendulum_pychrono"
Write-Host "  - $OutputBase\exp6_double_pendulum_compare"
