from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    """Console entry point for `admiser-install-cppad`."""
    print("==============================================")
    print("  ADMISER CppAD / cppad_py 安装向导 (WSL/Ubuntu)")
    print("==============================================")

    # 简单平台提示（不强制退出，方便你后面扩展）
    if sys.platform != "linux":
        print("警告：当前平台不是 Linux（含 WSL）。")
        print("      本安装流程目前只在 Debian/Ubuntu/WSL 上测试通过。")
        print()

    # 通过 importlib.resources 找到打包进去的 bash 脚本
    try:
        script_path = files("admiser.tools") / "build_cppad_py.sh"
    except Exception as exc:  # 极端情况：包结构异常
        print(f"错误：无法定位 build_cppad_py.sh: {exc}", file=sys.stderr)
        raise SystemExit(1)

    script_path = Path(script_path)

    if not script_path.is_file():
        print(f"错误：找不到脚本文件：{script_path}", file=sys.stderr)
        raise SystemExit(1)

    # 确认 bash 存在（一般 WSL/Ubuntu 都有）
    bash_exe = os.environ.get("BASH", "bash")
    if not shutil.which(bash_exe) if "shutil" in globals() else False:
        # 为了避免多引入模块，也可以简单信任 "bash"
        pass

    print(f"将执行安装脚本：{script_path}")
    print("（提示：你可以通过环境变量覆盖设置，例如：")
    print("  VENV_DIR=myenv CPPAD_PY_TAG=master admiser-install-cppad")
    print("）")
    print()

    try:
        # 在当前工作目录下运行脚本，这样 cppad_py 会克隆到“用户当前目录”
        subprocess.run([bash_exe, str(script_path)], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"安装失败，退出码：{exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
