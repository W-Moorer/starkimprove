param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [switch]$Run,
    [switch]$SkipBuild,
    [string]$Cases = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $PSScriptRoot "phase0_baseline_runner.py"
$OutputBase = Join-Path $RepoRoot "output\paper_experiments"
$ExePath = Join-Path $RepoRoot "$BuildDir\examples\$Config\examples.exe"

if ($Run -and -not $SkipBuild) {
    cmake --build (Join-Path $RepoRoot $BuildDir) --config $Config --target examples
}

$Args = @(
    $Runner,
    "--output-base", $OutputBase
)

if ($Run) {
    $Args += @("--run", "--exe", $ExePath)
}

if ($Cases -ne "") {
    $Args += @("--cases", $Cases)
}

if ($Strict) {
    $Args += "--strict"
}

python @Args
