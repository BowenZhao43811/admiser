from __future__ import annotations

import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def main() -> None:
    """Console entry point for `admiser-install-cppad`."""

    print("==============================================")
    print("  ADMISER CppAD / cppad_py 安装向导")
    print("  (当前实现仅在 Debian/Ubuntu/WSL 上测试通过)")
    print("==============================================")
    print()

    # 通过本包名定位被打包进去的 bash 脚本
    try:
        script = files("admiser_cppad_installer") / "install_cppad_stack.sh"
    except Exception as exc:  # 理论上不该发生，除非包没装好
        print(f"错误：无法定位安装脚本：{exc}", file=sys.stderr)
        raise SystemExit(1)

    script_path = Path(script)
    if not script_path.is_file():
        print(f"错误：找不到脚本文件：{script_path}", file=sys.stderr)
        raise SystemExit(1)

    bash = os.environ.get("BASH", "bash")

    print(f"将执行安装脚本：{script_path}")
    print("提示：")
    print("  - 你可以直接回车使用默认虚拟环境名 admiser_venv；")
    print("  - 或者在运行前设置环境变量，例如：")
    print("      VENV_DIR=myenv admiser-install-cppad")
    print()

    env = os.environ.copy()

    try:
        subprocess.run([bash, str(script_path)], check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"安装失败，退出码：{exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
