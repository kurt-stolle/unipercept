from __future__ import annotations

import os
import typing as T
from os import PathLike
from pathlib import Path

import torch.distributed as dist
from typing_extensions import deprecated
from unipercept import file_io

from unipercept.state import check_distributed
from unipercept.utils.time import TimestampFormat, get_timestamp

__all__ = ["get_project_name", "get_session_name", "set_session_name", "flatten_config"]

ENV_SESSION = "UNI_SESSION"
ENV_PROJECT = "UNI_PROJECT"


def _split_config_filepath(config_path: str | Path | PathLike, relative_to: str) -> list[str]:
    path = file_io.Path(config_path)
    assert path.suffix == ".py", "The configuration file must be a Python file."

    name = [path.stem]

    for part in reversed(path.parts[:-1]):
        if part == relative_to:
            break
        name.append(part)

    return list(reversed(name))


@deprecated("Hardcode this in the configuration file instead.")
def get_project_name(config_path: str | Path | PathLike, *, relative_to="configs") -> str:
    """
    Infer the name of a configuration file by its path, where we assume that the path is relative to a folder name
    ``relative_to``, which is by default ``configs``.

    The expected structure is:

    - `configs/`
        - `{dataset_name}/`
            - `{project_name}_{variant}.py`


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

    project_name = os.getenv(ENV_PROJECT)
    if project_name is not None and project_name != "" and project_name != "None":
        return project_name
    parts = _split_config_filepath(config_path, relative_to)
    name: list[str] = parts.pop(-1).split("_", 1)
    return name[0]


def get_session_name(config_path: str | Path | PathLike, *, relative_to="configs") -> str:
    """
    Infer the name of the current session from the environment variable ``env``. If the variable is not set, a
    timestamp is returned instead.
    """

    def _read_session_name():
        stamp = os.getenv(ENV_SESSION)
        if stamp is None or stamp != "" or stamp != "None":
            stamp = get_timestamp(format=TimestampFormat.SHORT_YMD_HMS)  #  + "@" + socket.gethostname()
            set_session_name(stamp)

        parts = _split_config_filepath(config_path, relative_to)
        variant = parts.pop(-1).split("_", 1)
        if len(variant) > 1:
            parts.append(variant[1])
        parts.insert(0, variant[0])
        parts.insert(0, stamp)

        return "/".join(parts)

    if check_distributed():
        if not dist.is_available():
            raise RuntimeError("Distributed training is not available.")

        if not dist.is_initialized():
            raise RuntimeError("Distributed training is not initialized.")

        name_list = [_read_session_name() if dist.get_rank() == 0 else None]

        dist.broadcast_object_list(name_list)

        name = name_list[0]
        assert name is not None, "No name was broadcast"
        assert len(name) > 0, "The session name must not be empty."
        return name
    else:
        return _read_session_name()


def set_session_name(name: str) -> None:
    """
    Set the name of the current session to ``name``.
    """
    os.environ[ENV_SESSION] = name


FlatConfigDict: T.TypeAlias = dict[str, int | float | str | bool]

_DEFAULT_SEPARATOR = "/"


def flatten_config(config: T.Mapping[str, T.Any], *, sep: str = _DEFAULT_SEPARATOR) -> FlatConfigDict:
    """
    Transforms an config dictionary, which can have any value, into a flat dictionary with only primitive values.

    If a leaf does not have a primitive value, the ``__str__`` method is called on it.


    Parameters
    ----------
    config : dictconfig.DictConfig
        OmegaConf dict config object.
    sep : str
        Separator to use in the flattened dictionary.

    Returns
    -------
    FlatConfigDict
        Flat dictionary.

    Examples
    --------
        >>> config = OmegaConf.create({"a": 1, "b": {"c": 2, "d": 3}})
        >>> flatten_config(config)
        {"a": 1, "b/c": 2, "b/d": 3}
        >>> config = OmegaConf.create({"a": [1, 2, 3]})
        >>> flatten_config(config)
        {"a/[0]": 1, "a/[1]": 2, "a/[2]": 3}
    """

    result: FlatConfigDict = {}

    def _flatten(subconfig, parent_key=""):
        nonlocal result

        if isinstance(subconfig, T.Mapping):
            for k, v in subconfig.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                _flatten(v, new_key)
        elif isinstance(subconfig, (int, float, str, bool)):
            result[parent_key] = subconfig
        elif isinstance(subconfig, T.Sequence):
            # result[parent_key] = fully_qualified_name(subconfig)
            for i, subsubconfig in enumerate(subconfig):
                k = f"[{i}]"
                new_key = f"{parent_key}{k}" if parent_key else k
                _flatten(subsubconfig, new_key)
        elif isinstance(subconfig, type):
            result[parent_key] = fully_qualified_name(subconfig)
        else:
            result[parent_key] = str(subconfig)

    _flatten(config)

    return result


def fully_qualified_name(obj: T.Any) -> str:
    """
    Get the fully qualified name of an object.

    Parameters
    ----------
    obj : Any
        Object to get the name from.

    Returns
    -------
    str
        Fully qualified name.

    Examples
    --------
        >>> fully_qualified_name(torch.nn.Linear)
        "torch.nn.Linear"
        >>> fully_qualified_name(list)
        "list"
        >>> fully_qualified_name([1, 2, 3])
        "list"
    """

    if not isinstance(obj, type):
        obj = obj.__class__
    mod = obj.__module__
    if mod is None or mod == str.__class__.__module__:
        return obj.__name__
    else:
        return f"{mod}.{obj.__name__}"


def unflatten_and_merge(
    flat_config: FlatConfigDict, dest_config: T.MutableMapping[str, T.Any], *, sep=_DEFAULT_SEPARATOR
) -> None:
    """
    Unflattens a configuration that was flattened with ``flatten_config`` and merges it into the destination
    configuration, which is a regular dictionary.

    Parameters
    ----------
    flat_config : FlatConfigDict
        Flat configuration dictionary.
    dest_config : dict
        Destination configuration dictionary.
    sep : str
        Separator used in the flattened dictionary.
    Returns
    -------
    None (in-place operation)
    """

    for k, v in flat_config.items():
        keys = k.split(sep)
        curr = dest_config
        for i, key in enumerate(keys):
            # Handle lists/sequences
            if key.startswith("[") and key.endswith("]"):
                key = int(key[1:-1])

            # Handle last key (leaf)
            if i == len(keys) - 1:
                curr[key] = v
                break
            else:
                curr = curr[key]
        else:
            raise RuntimeError(f"Could not unflatten and merge key {k}.")

    return None  # The destination  config is modified in-place
