param(
    [string]$V3Python = 'py.exe',
    [string[]]$PythonPrefix = @('-3.11'),
    [switch]$RunConfirmation
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$project = (Get-Location).Path
$research = Join-Path $project 'results/idpc_v3_research'
$logs = Join-Path $research '12_logs'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'

function Invoke-Research([string]$module, [string[]]$options) {
    & $V3Python @PythonPrefix -B -m $module @options
    if ($LASTEXITCODE -ne 0) {
        throw "$module returned exit code $LASTEXITCODE"
    }
}

function Write-Reports {
    Invoke-Research 'experiments.report_idpc_v3' @('--verify-paths')
    Invoke-Research 'experiments.select_idpc_v3' @()
    Invoke-Research 'experiments._support.idpc_v3_mechanisms' @()
    Invoke-Research 'experiments.summarize_idpc_v3' @()
    Invoke-Research 'experiments.plot_idpc_v3' @()
    Invoke-Research 'experiments.review_idpc_v3' @()
}

function Complete-MatchedCaps([string]$phase) {
    Invoke-Research 'experiments.report_idpc_v3' @('--verify-paths')
    $coverage = Get-Content -LiteralPath (Join-Path $research 'coverage.json') -Raw | ConvertFrom-Json
    $extensions = $coverage | Where-Object { $_.status -eq 'below_matched_cap' -and $_.phase -eq $phase } |
                              Group-Object -Property { Split-Path -Parent $_.artifact }
    foreach ($group in $extensions) {
        $seedIds = @($group.Group | ForEach-Object { [string]$_.seed })
        $origin = Join-Path $research $group.Name
        Invoke-Research 'experiments.continue_idpc_v3_cap' (@('--source', $origin, '--seeds') + $seedIds)
    }
}
Complete-MatchedCaps 'pilot'
Write-Reports

$frozen = Join-Path $research '13_configs/frozen_v3.json'
if (-not (Test-Path -LiteralPath $frozen)) {
    $rows = Import-Csv -LiteralPath (Join-Path $research '08_v3_pilot/selection_evidence.csv')
    $n3 = $rows | Where-Object { $_.arm -eq 'N3' -and $_.eligible -eq 'True' }
    if ($n3) {
        try {
            Invoke-Research 'experiments.run_idpc_v3_memoized' @('--phase','pilot','--profile','matched_ls',
              '--device','cuda','--arms','N3','--only-seeds','4200','--resume',
              '--out','results/idpc_v3_research/11_checkpoints/performance_validation')
            Invoke-Research 'experiments._support.idpc_v3_memo_check' @()
        } catch {
            Write-Warning 'Memoization parity is unavailable. The original implementation remains eligible.'
        }
    }
    Invoke-Research 'experiments.select_idpc_v3' @('--freeze')
}
if (-not $RunConfirmation) {
    return
}
if (-not (Test-Path -LiteralPath $frozen)) {
    throw 'Confirmation requires a selected candidate and complete pilot records.'
}

$frozenRelative = 'results/idpc_v3_research/13_configs/frozen_v3.json'
$checkpoint = 'results/idpc_v3_research/11_checkpoints'
$common = @('--phase', 'confirmation', '--resume', '--frozen-config', $frozenRelative)
$jobs = @()

function Begin-Research([string]$name, [string[]]$options, [string]$module='experiments.run_idpc_v3') {
    $stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMdd_HHmmss')
    $args3 = $PythonPrefix + @('-B', '-m', $module) + $options
    $paths = @{
        FilePath = $V3Python
        ArgumentList = $args3
        WorkingDirectory = $project
        WindowStyle = 'Hidden'
        PassThru = $true
        RedirectStandardOutput = Join-Path $logs ($name + '_' + $stamp + '.out')
        RedirectStandardError = Join-Path $logs ($name + '_' + $stamp + '.err')
    }
    $worker = Start-Process @paths
    [void]$worker.Handle
    return $worker
}

# Canonical MFEA results are shared by the two matched profiles.
foreach ($profile in @('native', 'matched_ls')) {
    foreach ($method in @('ea', 'mfea')) {
        $name = "confirmation_" + $profile + '_' + $method
        $options = @('--profile', $profile, '--device', 'cpu', '--threads', '2', '--arms', 'A0',
                     '--methods', $method, '--out', "$checkpoint/final_baselines/$method")
        $jobs += Begin-Research $name ($common + $options)
    }
}
$initOptions = @('--profile', 'matched_init_ls', '--device', 'cpu', '--threads', '2', '--arms', 'A0',
                 '--methods', 'ea', '--out', "$checkpoint/final_baselines")
$jobs += Begin-Research 'confirmation_matched_init_ea' ($common + $initOptions)

# A single CUDA worker owns the graph features at a time on this 4 GB device.
$gpuFailure = $false
$decisions = Get-Content -LiteralPath $frozen -Raw | ConvertFrom-Json
foreach ($profile in @('native', 'matched_ls', 'matched_init_ls')) {
    $options = @('--profile', $profile, '--device', 'cuda', '--threads', '4', '--arms', 'V3', '--methods', 'ektgl')
    $module = if ($profile -eq 'native' -and $decisions.native_subpath_memoization) {
        'experiments.run_idpc_v3_memoized'
    } else { 'experiments.run_idpc_v3' }
    $process = Begin-Research ('confirmation_' + $profile + '_ektgl') ($common + $options) $module
    $process.WaitForExit()
    $process.Refresh()
    $gpuFailure = $gpuFailure -or ($process.ExitCode -ne 0)
}
$cpuFailure = $false
foreach ($worker in $jobs) {
    $worker.WaitForExit()
    $worker.Refresh()
    $cpuFailure = $cpuFailure -or ($worker.ExitCode -ne 0)
}
Complete-MatchedCaps 'confirmation'
Write-Reports
if ($gpuFailure -or $cpuFailure) {
    throw 'One or more confirmation workers reported an error. See coverage.json and 12_logs.'
}
