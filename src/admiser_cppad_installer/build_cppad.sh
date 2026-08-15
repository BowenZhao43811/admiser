#!/usr/bin/env bash
set -euo pipefail

CPPAD_PY_TAG="${CPPAD_PY_TAG:-master}"
JOBS="${JOBS:-$(nproc)}"

echo "=============================================="
echo "  ADMISER: installing CppAD / cppad_py into the active environment"
echo "=============================================="
echo

# [0/6] check that a virtual environment is active
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "error: no active Python virtual environment detected (VIRTUAL_ENV is unset)." >&2
  echo "Run this through \"admiser-install-cppad\" from the virtual environment that has admiser installed, e.g.:" >&2
  echo "(assuming admiser is installed in a virtual environment called admiser_venv)"
  echo "first activate the environment:" >&2
  echo "  source admiser_venv/bin/activate" >&2
  echo "then run:" >&2
  echo "  admiser-install-cppad" >&2
  exit 1
fi

echo "[0/6] active Python interpreter:"
python - << 'PY'
import sys
print("  sys.executable =", sys.executable)
PY
echo

# [1/6] install system dependencies
echo "[1/6] Install system deps (requires sudo)..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
       build-essential g++ git cmake swig \
       python3-dev python3-venv python3-pip
else
  echo "warning: apt-get not found; this script only automates Debian/Ubuntu/WSL." >&2
  echo "Please install manually: g++, cmake, git, swig, python3-dev, python3-venv, python3-pip." >&2
fi
echo

# [2/6] upgrade pip / wheel / setuptools in the active environment
echo "[2/6] Upgrade pip / wheel / setuptools in current environment..."
python -m pip install -U pip wheel setuptools
echo

# [3/6] clone the cppad_py repository
# create a temporary build directory, e.g. /tmp/cppad_py_build_XXXXXX
CPPAD_PY_BUILD_DIR="$(mktemp -d -t cppad_py_build_XXXXXX)"
echo "  temporary build directory: ${CPPAD_PY_BUILD_DIR}"

# clean up the temporary directory when the script exits
cleanup() {
  if [ -n "${CPPAD_PY_BUILD_DIR:-}" ] && [ -d "${CPPAD_PY_BUILD_DIR}" ]; then
    echo "removing temporary build directory: ${CPPAD_PY_BUILD_DIR}"
    rm -rf "${CPPAD_PY_BUILD_DIR}"
  fi
}
trap cleanup EXIT

# shallow-clone the requested tag or branch to avoid pulling the full history
git clone \
  --branch "${CPPAD_PY_TAG}" \
  --depth 1 \
  https://github.com/bradbell/cppad_py.git \
  "${CPPAD_PY_BUILD_DIR}"

echo

# remember the current directory so we can return to it after the build
OLDPWD="$(pwd)"

# enter the temporary build directory
cd "${CPPAD_PY_BUILD_DIR}"

# [4/6] install CppAD using cppad_py's own helper scripts
echo "[4/6] Install CppAD via cppad_py helper scripts..."
bin/system_depend.sh
bin/get_cppad.sh
echo

# [5/6] build and install the cppad_py wheel
echo "[5/6] Build cppad_py wheel..."
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
python -m pip install -U build
python -m build
echo

echo "Install cppad_py wheel into current environment..."
python -m pip install dist/*.whl
echo

# return to the original directory
cd "${OLDPWD}"

# [6/6] quick numerical smoke test
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
echo "========= done ========="
echo "cppad_py has been installed into the active virtual environment:"
echo "  ${VIRTUAL_ENV}"
echo "The AD features of admiser are now available in this environment."
