"""
Defines a lazy module factory that allows to create a module's __getattr__ and __dir__ function from the absolute
name and exported module list.
"""

import importlib
import typing as T


def lazy_module_factory(name: str, modules: T.Iterable[str], /, extras: T.Optional[T.Iterable[str]] = None):
    """
    Creates a __getattr__ and __dir__ function for a module that lazily imports its submodules.

    Args:
        name: The absolute name of the module (e.g. __name__)
        modules: The list of submodules to import (e.g. __all__)

    Returns:
        A tuple of (__getattr__, __dir__)
    """

    attr = set(modules)

    def __getattr__(mod: str):
        if mod in attr:
            return importlib.import_module(f"{name}.{mod}")
        else:
            raise AttributeError(f"module {name!r} has no submodule {mod!r}, available are: {attr}")

    dir = list(modules)
    if extras is not None:
        dir.extend(extras)

    def __dir__():
        return dir

    return __getattr__, __dir__
