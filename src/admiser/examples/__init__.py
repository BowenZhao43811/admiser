from importlib.resources import files

def get_notebook_path(name: str = "main_solver.ipynb") -> str:
    """
    返回打包在 admiser.examples.notebooks 中的 notebook 的绝对路径。
    """
    return str(files(__package__) / "notebooks" / name)