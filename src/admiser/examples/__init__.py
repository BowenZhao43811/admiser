from importlib.resources import files

#: Name of the example notebook shipped with the package.
DEFAULT_NOTEBOOK = "main_solver_entrence.ipynb"


def get_notebook_path(name: str = DEFAULT_NOTEBOOK) -> str:
    """Absolute path of a notebook packaged under admiser.examples.notebooks."""
    path = files(__package__) / "notebooks" / name
    if not path.is_file():
        available = sorted(p.name for p in (files(__package__) / "notebooks").iterdir())
        raise FileNotFoundError(
            f"notebook {name!r} is not part of the package; available: {available}"
        )
    return str(path)
