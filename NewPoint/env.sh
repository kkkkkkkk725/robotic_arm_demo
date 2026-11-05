#!/usr/bin/env bash
# 环境变量设置脚本: source env.sh
# Windows PowerShell 请使用 env.ps1

# GraspNet-1Billion 数据集根目录 (目录下应有 scenes/ grasp_labels/ camera/)
export GRASPNET_ROOT="${GRASPNET_ROOT:-/path/to/graspnet_data}"

# 工程根目录
export GRASP_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python 路径
export PYTHONPATH="${GRASP_DEMO_ROOT}:${PYTHONPATH}"

echo "[env] GRASPNET_ROOT   = ${GRASPNET_ROOT}"
echo "[env] GRASP_DEMO_ROOT = ${GRASP_DEMO_ROOT}"
