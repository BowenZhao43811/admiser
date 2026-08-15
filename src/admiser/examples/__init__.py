from importlib.resources import files

#: 随包分发的示例 notebook 文件名
DEFAULT_NOTEBOOK = "main_solver_entrence.ipynb"


def get_notebook_path(name: str = DEFAULT_NOTEBOOK) -> str:
    """
    返回打包在 admiser.examples.notebooks 中的 notebook 的绝对路径。
    """
    path = files(__package__) / "notebooks" / name
    if not path.is_file():
        available = sorted(p.name for p in (files(__package__) / "notebooks").iterdir())
        raise FileNotFoundError(
            f"notebook {name!r} 不在包内；可用的有：{available}"
        )
    return str(path)