param(
    [switch]$FullCampaign,
    [switch]$HoldAwake
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$project=(Get-Location).Path
$results=Join-Path $project 'results/idpc_n3_300gen_optimized'
$logs=Join-Path $results 'logs'
$stamp=[DateTimeOffset]::UtcNow.ToString('yyyyMMdd_HHmmss')
$module='experiments.run_idpc_n3_optimized'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'

function Invoke-Module([string]$Name, [string[]]$Options) {
    & py -3.11 -B -m $Name @Options
    if ($LASTEXITCODE -ne 0) { throw "$Name exited with code $LASTEXITCODE" }
}

function Start-Suites([string]$Method, [int[]]$Seeds, [string]$Label) {
    $options=@('-3.11','-B','-m',$module,'--method',$Method,'--seeds') + @($Seeds | ForEach-Object { [string]$_ })
    $worker=Start-Process -FilePath py.exe -ArgumentList $options -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs ($Label + '_' + $stamp + '.out')) -RedirectStandardError (Join-Path $logs ($Label + '_' + $stamp + '.err'))
    [void]$worker.Handle
    return $worker
}

if ($HoldAwake) {
    Add-Type -Namespace N3Optimized -Name Power -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
"@
    if ([N3Optimized.Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        Write-Warning 'Windows did not grant the temporary idle-sleep request.'
    }
}

try {
    Invoke-Module $module @('--verify-only')
    if ($FullCampaign) {
        $acceptance=Get-Content -LiteralPath (Join-Path $project 'results/idpc_n3_performance_audit/audit_gate.json') -Raw | ConvertFrom-Json
        $manifest=Get-Content -LiteralPath (Join-Path $results 'experiment_manifest.json') -Raw | ConvertFrom-Json
        if ($acceptance.status -ne 'PASS' -or $acceptance.fingerprint -ne $manifest.fingerprint) {
            throw 'The targeted N3 audit must pass before starting all seeds.'
        }
        $workers=@(
            (Start-Suites 'ea' (4301..4319) 'ea')
            (Start-Suites 'ektgl' (4301..4319) 'ektgl')
            (Start-Suites 'mfea' (4301..4309) 'mfea_part1')
            (Start-Suites 'mfea' (4310..4319) 'mfea_part2')
        )
        $failed=$false
        foreach ($worker in $workers) {
            $worker.WaitForExit()
            $worker.Refresh()
            $failed=$failed -or ($worker.ExitCode -ne 0)
        }
        if ($failed) { throw 'At least one worker failed. Check the optimized logs.' }
        Invoke-Module 'experiments.report_idpc_matched_n3_generation' @('--out',$results)
        Invoke-Module 'experiments.plot_idpc_matched_n3_generation' @('--out',$results)
    } else {
        Invoke-Module 'unittest' @('discover','-s','tests','-p','test_idpc_n3_fast.py')
        Invoke-Module 'experiments.audit_idpc_n3_performance' @('--compare','--steps','3')
        Invoke-Module 'experiments.audit_idpc_n3_performance' @('--lines','--methods','mfea','--steps','1')
        foreach ($method in @('mfea','ea','ektgl')) {
            Invoke-Module $module @('--method',$method,'--seeds','4300')
        }
        Invoke-Module 'experiments.verify_idpc_n3_performance' @()
    }
} finally {
    if ($HoldAwake) { [void][N3Optimized.Power]::SetThreadExecutionState([uint32]2147483648) }
}
