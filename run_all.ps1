$ErrorActionPreference = 'Stop'
$python = $env:HW5_PYTHON
if (-not $python -or -not (Test-Path $python)) {
    $python = (Get-Command python.exe -ErrorAction Stop).Source
}
# 实验作业五 一键复现脚本（本机实测：Windows 11 + RTX 4060 Laptop 8GB + Python 3.12 + torch 2.6.0+cu124）
$root = $PSScriptRoot
$env:PYTHONPATH = Join-Path $root '.deps'
$env:MPLCONFIGDIR = Join-Path $root '.matplotlib'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

function Run-Step([string]$logName, [string[]]$argv) {
    Write-Host "=== $logName : python $($argv -join ' ')" -ForegroundColor Cyan
    & $python -u @argv *>&1 | Tee-Object -FilePath (Join-Path $root "logs\$logName.log")
    if ($LASTEXITCODE -ne 0) { throw "step failed: $logName" }
}

$mode = if ($args.Count -gt 0) { $args[0] } else { 'main' }

switch ($mode) {
    'main' {
        # 主流程：基线 → SFT → DPO（17 组偏好对）→ 两种口径三阶段评测 → 手算复核
        Run-Step 'gpu_make_data' @('hw5_dpo.py', '--mode', 'make_data')
        Run-Step 'gpu_infer'     @('hw5_dpo.py', '--mode', 'infer')
        Run-Step 'gpu_sft'       @('hw5_dpo.py', '--mode', 'sft')
        Run-Step 'gpu_dpo'       @('hw5_dpo.py', '--mode', 'dpo')
        Run-Step 'gpu_eval'      @('hw5_dpo.py', '--mode', 'eval')
        Run-Step 'gpu_eval_4bit' @('hw5_dpo.py', '--mode', 'eval', '--eval_4bit')
        Run-Step 'gpu_dpo_check' @('hw5_dpo.py', '--dpo_check', '1.2,0.8,0.9,0.6,0.1')
    }
    'sweep' {
        # 延伸任务 A：β 敏感性扫描（β=0.05 / 0.1(3 轮) / 0.5，各 17~51 步）+ 两种口径评测
        foreach ($b in @('0.05', '0.5')) {
            $tag = "beta$b"
            Run-Step "gpu_dpo_$tag"      @('hw5_dpo.py', '--mode', 'dpo', '--beta', $b, '--tag', $tag)
            Run-Step "gpu_eval_$tag"     @('hw5_dpo.py', '--mode', 'eval', '--tag', $tag)
            Run-Step "gpu_eval_${tag}_4bit" @('hw5_dpo.py', '--mode', 'eval', '--tag', $tag, '--eval_4bit')
        }
        Run-Step 'gpu_dpo_b01e3'      @('hw5_dpo.py', '--mode', 'dpo', '--beta', '0.1', '--epochs', '3', '--tag', 'b01e3')
        Run-Step 'gpu_eval_b01e3'     @('hw5_dpo.py', '--mode', 'eval', '--tag', 'b01e3')
        Run-Step 'gpu_eval_b01e3_4bit' @('hw5_dpo.py', '--mode', 'eval', '--tag', 'b01e3', '--eval_4bit')
    }
    'report' {
        Run-Step 'plot'   @('plot_curves.py')
        & $python make_screenshot.py params
        & $python make_screenshot.py dpo_check
        Run-Step 'report' @('build_report.py')
    }
    default { throw "unknown step: $mode (use main / sweep / report)" }
}
