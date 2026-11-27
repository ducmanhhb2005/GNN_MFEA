param(
    [string]$Python = 'py.exe',
    [string[]]$PythonPrefix = @('-3.11'),
    [switch]$HoldAwake
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$project = (Get-Location).Path
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$folder = Join-Path $project 'results/idpc_matched_n3'
$logs = Join-Path $folder 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Test-Stage([string]$Module, [string[]]$Options) {
    & $Python @PythonPrefix -B -m $Module @Options
    if ($LASTEXITCODE -ne 0) { throw "$Module failed with exit code $LASTEXITCODE" }
}
function Start-Worker([string]$Method, [int[]]$Seeds, [string]$Label) {
    $stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $arguments = $PythonPrefix + @('-B','-m','experiments.run_idpc_matched_n3',
                                   '--method',$Method,'--seeds') + @($Seeds | ForEach-Object { [string]$_ })
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs ($Label+'_'+$stamp+'.out')) -RedirectStandardError (Join-Path $logs ($Label+'_'+$stamp+'.err'))
    [void]$process.Handle
    return $process
}

if ($HoldAwake) {
    Add-Type -Namespace MatchedN3 -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
"@
    if ([MatchedN3.Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'The host did not grant the temporary idle-sleep request.'
    }
}
try {
    Test-Stage 'experiments.run_idpc_matched_n3' @('--verify-only')
    $smokeGate = Join-Path $logs 'smoke_validation.json'
    if (-not (Test-Path -LiteralPath $smokeGate)) {
        $smoke = foreach ($method in @('ea','mfea','ektgl')) {
            Start-Worker $method @(4300) ('smoke_'+$method)
        }
        foreach ($worker in $smoke) {
            $worker.WaitForExit()
            $worker.Refresh()
            if ($worker.ExitCode -ne 0) { throw "Smoke worker failed with exit code $($worker.ExitCode)" }
        }
    }
    Test-Stage 'experiments.report_idpc_matched_n3' @('--smoke')
    $workers = @()
    $workers += Start-Worker 'ea' (4301..4319) 'ea_n3'
    $workers += Start-Worker 'ektgl' (4301..4319) 'ektgl_n3'
    $workers += Start-Worker 'mfea' (4301..4310) 'mfea_n3_part1'
    $workers += Start-Worker 'mfea' (4311..4319) 'mfea_n3_part2'
    $failed = $false
    foreach ($worker in $workers) {
        $worker.WaitForExit()
        $worker.Refresh()
        $failed = $failed -or $worker.ExitCode -ne 0
    }
    Test-Stage 'experiments.report_idpc_matched_n3' @()
    if ($failed) { throw 'A benchmark worker failed. Completed records are available in coverage.json.' }
    $coverage = Get-Content -LiteralPath (Join-Path $folder 'coverage.json') -Raw | ConvertFrom-Json
    if ($coverage.status -ne 'SOLVER_COMPLETE') { throw 'The original stopping rule produced incomplete comparable coverage.' }
    Test-Stage 'experiments.plot_idpc_matched_n3' @()
    Test-Stage 'experiments.review_idpc_matched_n3' @()
} finally {
    if ($HoldAwake) { [void][MatchedN3.Power]::SetThreadExecutionState([uint32]2147483648) }
}
