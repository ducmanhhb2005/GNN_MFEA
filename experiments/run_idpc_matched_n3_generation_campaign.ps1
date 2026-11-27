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
$generation = Join-Path $project 'results/idpc_n3_300gen'
$logs = Join-Path $generation 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
function Invoke-Module([string]$Module, [string[]]$Options) {
    & $Python @PythonPrefix -B -m $Module @Options
    if ($LASTEXITCODE -ne 0) { throw "$Module failed with exit code $LASTEXITCODE" }
}
function Start-GenerationWorker([string]$Method, [int[]]$Seeds, [string]$Label) {
    $stamp=[DateTimeOffset]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $arguments = $PythonPrefix + @('-B','-m','experiments.run_idpc_matched_n3_generation',
                                   '--method',$Method,'--seeds') + @($Seeds | ForEach-Object { [string]$_ })
    $worker = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs ($Label+'_'+$stamp+'.out')) -RedirectStandardError (Join-Path $logs ($Label+'_'+$stamp+'.err'))
    [void]$worker.Handle
    return $worker
}
if ($HoldAwake) {
    Add-Type -Namespace N3Generation -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
"@
    if ([N3Generation.Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        Write-Warning 'The host did not grant the temporary idle-sleep request.'
    }
}
try {
    Invoke-Module 'experiments.run_idpc_matched_n3_generation' @('--verify-only')
    $workers=@()
    $workers+=Start-GenerationWorker 'ea' (4300..4319) 'ea_n3_generation'
    $workers+=Start-GenerationWorker 'ektgl' (4300..4319) 'ektgl_n3_generation'
    $workers+=Start-GenerationWorker 'mfea' (4300..4309) 'mfea_n3_generation_part1'
    $workers+=Start-GenerationWorker 'mfea' (4310..4319) 'mfea_n3_generation_part2'
    $failed=$false
    foreach ($worker in $workers) {
        $worker.WaitForExit()
        $worker.Refresh()
        $failed=$failed -or ($worker.ExitCode -ne 0)
    }
    Invoke-Module 'experiments.report_idpc_matched_n3_generation' @()
    if ($failed) { throw 'A generation worker failed. Resume from the existing checkpoints.' }
    Invoke-Module 'experiments.plot_idpc_matched_n3_generation' @()
} finally {
    if ($HoldAwake) { [void][N3Generation.Power]::SetThreadExecutionState([uint32]2147483648) }
}
