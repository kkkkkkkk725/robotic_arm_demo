# 环境变量设置脚本 (Windows PowerShell): . .\env.ps1

if (-not $env:GRASPNET_ROOT) {
    $env:GRASPNET_ROOT = "C:\path\to\graspnet_data"
}

$env:GRASP_DEMO_ROOT = $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"

Write-Host "[env] GRASPNET_ROOT   = $env:GRASPNET_ROOT"
Write-Host "[env] GRASP_DEMO_ROOT = $env:GRASP_DEMO_ROOT"
