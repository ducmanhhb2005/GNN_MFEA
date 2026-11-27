param([int]$OriginalPrimaryProcess, [switch]$RunConfirmation)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$v3Temp = [IO.Path]::GetTempPath()
$env:PYTHONPATH = (Join-Path $v3Temp 'gnn_mfea_idpc_v2_torch_cu126') + ';' +
                  (Join-Path $v3Temp 'gnn_mfea_idpc_v2_numba') + ';' +
                  (Join-Path $v3Temp 'gnn_mfea_ektgl_scipy_py311')
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$research = 'results/idpc_v3_research'
$logs = Join-Path $research '12_logs'
$py = (Get-Command py.exe).Source
$sum = @('-3.11','-B','-m')

function Begin-Stage([string]$name,[string]$module,[array]$args3) {
    $seq = $sum + @($module) + $args3
    $out = Join-Path $logs ($name+'.out')
    $err = Join-Path $logs ($name+'.err')
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    if (Test-Path -LiteralPath $out) { Copy-Item -LiteralPath $out -Destination ($out+'.'+$stamp) }
    if (Test-Path -LiteralPath $err) { Copy-Item -LiteralPath $err -Destination ($err+'.'+$stamp) }
    $worker = Start-Process -FilePath $py -ArgumentList $seq -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    [void]$worker.Handle
    return $worker
}

function Finish-Stage([string]$name, $process) {
    $process.WaitForExit()
    $process.Refresh()
    if ($process.ExitCode -ne 0) { throw "$name returned $($process.ExitCode). See 12_logs." }
    [DateTimeOffset]::UtcNow.ToString('o') | Set-Content -LiteralPath (Join-Path $logs ($name+'.complete'))
}

function Stage([string]$name,[string]$module,[array]$args3) {
    if (Test-Path -LiteralPath (Join-Path $logs ($name+'.complete'))) { return }
    Finish-Stage $name (Begin-Stage $name $module $args3)
}

$run = 'experiments.run_idpc_v3'
$matched = @('--phase','pilot','--profile','matched_ls','--device','cuda','--resume')
$prior = Join-Path $logs 'pilot_primary.out'
if ($OriginalPrimaryProcess) {
    while (Get-Process -Id $OriginalPrimaryProcess -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 15
    }
}
if (!(Test-Path -LiteralPath $prior) -or !(Select-String -Quiet -Path $prior -Pattern '^D3 ektgl/4202 complete$')) {
    Stage 'pilot_primary' $run ($matched+@('--arms','A0','A1','D1','D2','D3'))
}

$init = @('--phase','pilot','--profile','matched_init_ls','--resume','--arms','A0')
$initEA = $null
if (!(Test-Path -LiteralPath (Join-Path $logs 'pilot_matched_init_ea.complete'))) {
    $initEA = Begin-Stage 'pilot_matched_init_ea' $run ($init+@('--device','cpu','--threads','2','--methods','ea',
                                '--out',"$research/11_checkpoints/matched_init_ea"))
}
Stage 'pilot_diagnostics' 'experiments.run_idpc_v3_diagnostics' ($matched+@('--arms','A0','--diagnostic',
                              '--out',"$research/11_checkpoints/detailed_diagnostics"))
Stage 'pilot_matched_init' $run ($init+@('--device','cuda','--methods','ektgl'))
if ($initEA) { Finish-Stage 'pilot_matched_init_ea' $initEA }

$prefix = 'experiments.run_idpc_v3_prefix'
$unchangedOrdinary = @('A2_32','A2_64','A3','A4','A5','A6','A8_identity','A8_independent','A8_shuffled',
                       'G1','G2','G3','G4')
$verifyLog = Join-Path $logs 'pilot_prefix_check.out'
try {
    Stage 'pilot_prefix_verification_runner' $prefix @('--arms','A0','--only-seeds','4200')
    Stage 'pilot_prefix_check' 'experiments._support.idpc_v3_prefix_check' @()
    Stage 'pilot_components_learning' $prefix (@('--arms') + $unchangedOrdinary)
} catch {
    $failure = $_
    if (Test-Path -LiteralPath (Join-Path $logs 'pilot_components_learning.out')) { throw $failure }
    Stage 'pilot_components_learning_fallback' $run ($matched+@('--arms')+$unchangedOrdinary)
}
Stage 'pilot_components_neighborhood' $run ($matched+@('--arms','N1','N2','N3','A9_legacy_32'))
Stage 'pilot_materialize' 'experiments.report_idpc_v3' @('--verify-paths')
& (Join-Path $PSScriptRoot 'run_idpc_v3_campaign.ps1') -RunConfirmation:$RunConfirmation
