#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CPPAD_PY_TAG="${CPPAD_PY_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"

echo "[0/7] Creat Python venv..."
if [ -z "${VENV_DIR:-}" ]; then
  read -rp "请输入要创建/使用的虚拟环境目录名（默认: admiser_venv）: " VENV_NAME_INPUT
  if [ -z "${VENV_NAME_INPUT}" ]; then
    VENV_DIR="admiser_venv"
  else
    VENV_DIR="${VENV_NAME_INPUT}"
  fi
fi

echo "[1/7] Install system deps..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
       build-essential g++ git cmake swig \
       python3-dev python3-venv python3-pip
fi

echo "[2/7] Create/activate venv: ${VENV_DIR}"
[ -d "$VENV_DIR" ] || ${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[3/7] Clone cppad_py..."
if [ ! -d cppad_py ]; then
  git clone https://github.com/bradbell/cppad_py.git
fi
cd cppad_py
git fetch --all
git checkout "${CPPAD_PY_TAG}"
git pull --ff-only || true

echo "[4/7] Install CppAD..."
bin/system_depend.sh
bin/get_cppad.sh

echo "[5/7] Build wheel..."
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
python -m pip install -U build
python -m build

echo "[6/7] Install wheel..."
python -m pip install dist/*.whl

echo "[7/7] Quick check..."
python - <<'PY'
import numpy as np, cppad_py
x = np.array([0.0, 1.0], dtype=float)
ax = cppad_py.independent(x)
ay = np.array([ax[0]*ax[0] + 3.0*ax[1]], dtype=object)
f  = cppad_py.d_fun(ax, ay)
y  = f.forward(0, np.array([2.0, 4.0], dtype=float))
J  = f.jacobian(np.array([2.0, 4.0], dtype=float))
print("OK y,J:", y, J)
PY

echo "cppad_py ready in venv: ${VENV_DIR}"
