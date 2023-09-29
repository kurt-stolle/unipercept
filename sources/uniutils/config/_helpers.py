import os
import socket
import typing as T
from os import PathLike
from pathlib import Path

from detectron2.config import instantiate
from omegaconf import dictconfig
from unicore import file_io
from uniutils.time import get_timestamp

__all__ = ["infer_project_name", "parse_dict", "infer_session_name"]


def infer_project_name(config_path: str | Path | PathLike, *, relative_to="configs", env="UNI_PROJECT") -> str:
    """
    Infer the name of a configuration file by its path, where we assume that the path is relative to a folder name
    ``relative_to``, which is by default ``configs``.


    Parameters
    ----------
    config_path : str
        Path to the configuration file. When called from a configuration file, usually this can be set to the
        ``__file__`` variable.
    relative_to : str
        Name of the folder that is the root of the configuration files.
    env : str
        Name of the environment variable that contains the name of the project. If the variable is not set, the
        function will return the name of the configuration file.

    Examples
    --------
        >>> get_name("configs/cityscapes/multidvps_resnet50.py")
        "cityscapes/multidvps_resnet50"
    """

    project_name = os.getenv(env)
    if project_name is not None and project_name != "" and project_name != "None":
        return project_name

    path = file_io.Path(config_path)
    assert path.suffix == ".py", "The configuration file must be a Python file."

    name = [path.stem]

    for part in reversed(path.parts[:-1]):
        if part == relative_to:
            break
        name.append(part)

    name = reversed(name)

    return "/".join(name)


def infer_session_name(*, env: str = "UNI_SESSION") -> str:
    """
    Infer the name of the current session from the environment variable ``env``. If the variable is not set, a
    timestamp is returned instead.
    """
    name = os.getenv(env)
    if name is None or name != "" or name != "None":
        name = get_timestamp() + "@" + socket.gethostname()
    return name


def parse_dict(d: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    """
    Parse a DictConfig object to a regular Dict. This is useful for the instantiation of some models that have a Dict
    argument that was lazily initiated.

    Always returns a new dictionary.
    """
    if isinstance(d, dictconfig.DictConfig):
        return {k: instantiate(v) for k, v in d.items()}
    return {k: v for k, v in d.items()}
