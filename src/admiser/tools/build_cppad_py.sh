#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CPPAD_PY_TAG="${CPPAD_PY_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"

echo "[1/6] Install system deps..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
       build-essential g++ git cmake swig \
       python3-dev python3-venv python3-pip
fi

echo "[2/6] Create/activate venv: ${VENV_DIR}"
[ -d "$VENV_DIR" ] || ${PYTHON_BIN} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

echo "[3/6] Clone cppad_py..."
if [ ! -d cppad_py ]; then
  git clone https://github.com/bradbell/cppad_py.git
fi
cd cppad_py
git fetch --all
git checkout "${CPPAD_PY_TAG}"
git pull --ff-only || true

echo "[4/6] Build wheel..."
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
python -m pip install -U build
python -m build

echo "[5/6] Install wheel..."
python -m pip install dist/*.whl

echo "[6/6] Quick check..."
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
