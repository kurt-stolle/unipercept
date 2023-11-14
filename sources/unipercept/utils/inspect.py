import inspect
import os
import sys
from typing import Optional


def calling_module_name(frames: int = 0, left: Optional[int] = None) -> str:
    """
    Return the name of the module that called the current function from the perspective
    of the function that calls this function.

    Parameters
    ----------
    frames
        The number of frames to go back in the stack, by default 0.
    left
        The number nested modules to return, e.g. when the module name is `a.b.c` and `left=1` the return value would
        be `a.b`. By default `None`, which returns the full module name.

    Returns
    -------
    The name of the module that called the current function.

    """
    frm = inspect.stack()[2 + frames]
    mod = inspect.getmodule(frm[0])
    if mod is None:
        raise ModuleNotFoundError(f"Could not find module for {frm.filename}, which was called from {frm.function}")

    name = mod.__name__
    if left is not None:
        name = ".".join(name.split(".")[: left + 1])

    return name


def caller_identity() -> tuple[str, tuple[str, int, str]]:  # type: ignore
    """
    Identify the caller by a name and hashable key.

    Returns
    -------
    tuple[str,str]
        Module name of the caller and a hashable key
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", __file__) not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "<main>"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back
