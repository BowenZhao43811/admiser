from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Console entry point for `admiser-install-cppad`."""

    print("==============================================")
    print("  ADMISER CppAD / cppad_py 安装向导")
    print("  (当前实现仅在 Debian/Ubuntu/WSL 上测试通过)")
    print("==============================================")
    print()

    # 用 __file__ 所在目录定位 .sh 文件
    pkg_dir = Path(__file__).resolve().parent
    script_path = pkg_dir / "build_cppad.sh"

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
