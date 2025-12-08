#!/usr/bin/env bash
set -euo pipefail

CPPAD_PY_TAG="${CPPAD_PY_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"

echo "=============================================="
echo "  ADMISER: 安装 CppAD / cppad_py 到当前环境"
echo "=============================================="
echo

# [0/x] 检查当前是否激活了虚拟环境
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "错误：当前未检测到已激活的 Python 虚拟环境 (VIRTUAL_ENV 未设置)。" >&2
  echo "请在安装有admiser的虚拟环境中通过 "admiser-install-cppad" 运行此脚本，例如：" >&2
  echo "（假设 admiser 已经安装在了虚拟环境 admiser_venv 中）"
  echo "先运行激活虚拟环境" >&2
  echo "  source admiser_venv/bin/activate" >&2
  echo "然后再运行：" >&2
  echo "  admiser-install-cppad" >&2
  exit 1
fi

echo "[0/6] 当前 Python 解释器："
python - << 'PY'
import sys
print("  sys.executable =", sys.executable)
PY
echo

# [1/6] 安装系统依赖
echo "[1/6] Install system deps (需要 sudo)..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
       build-essential g++ git cmake swig \
       python3-dev python3-venv python3-pip
else
  echo "警告：未检测到 apt-get，本脚本当前只对 Debian/Ubuntu/WSL 做自动处理。" >&2
  echo "请手动确保系统已安装：g++, cmake, git, swig, python3-dev, python3-venv, python3-pip。" >&2
fi
echo

# [2/6] 升级当前环境里的 pip / wheel / setuptools
echo "[2/6] Upgrade pip / wheel / setuptools in current environment..."
python -m pip install -U pip wheel setuptools
echo

# [3/6] 克隆或更新 cppad_py 仓库
echo "[3/6] Clone / update cppad_py..."
if [ ! -d cppad_py ]; then
  git clone https://github.com/bradbell/cppad_py.git
fi
cd cppad_py
git fetch --all
git checkout "${CPPAD_PY_TAG}"
git pull --ff-only || true
echo

# [4/6] 使用 cppad_py 自带脚本安装 CppAD
echo "[4/6] Install CppAD via cppad_py helper scripts..."
bin/system_depend.sh
bin/get_cppad.sh
echo

# [5/6] 构建并安装 cppad_py wheel
echo "[5/6] Build cppad_py wheel..."
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
python -m pip install -U build
python -m build
echo

echo "Install cppad_py wheel into current environment..."
python -m pip install dist/*.whl
echo

# [6/6] 快速数值检查
echo "[6/6] Quick check..."
python - << 'PY'
import numpy as np, cppad_py
x = np.array([0.0, 1.0], dtype=float)
ax = cppad_py.independent(x)
ay = np.array([ax[0]*ax[0] + 3.0*ax[1]], dtype=object)
f  = cppad_py.d_fun(ax, ay)
y  = f.forward(0, np.array([2.0, 4.0], dtype=float))
J  = f.jacobian(np.array([2.0, 4.0], dtype=float))
print("cppad_py check OK. y =", y, ", J =", J)
PY

echo
echo "========= 完成 ========="
echo "cppad_py 已成功安装到当前虚拟环境："
echo "  ${VIRTUAL_ENV}"
echo "你现在可以在此环境中使用 admiser 的 AD 功能。"
