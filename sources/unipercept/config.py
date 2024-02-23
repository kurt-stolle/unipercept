"""
Lazy configuration system, inspired by and based on Detectron2 and Hydra.
"""

from __future__ import annotations

import ast
import builtins
import collections.abc as abc
import dataclasses as D
import enum
import os
import types
import typing as T
import uuid
from contextlib import contextmanager
from copy import deepcopy
from distutils.util import strtobool
from typing import Any, List

import omegaconf
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf, SCMode
from typing_extensions import override

import unipercept.file_io as file_io
from unipercept.utils.inspect import generate_path, locate_object

if T.TYPE_CHECKING:
    from unipercept.utils.typings import Pathable

__all__ = [
    "apply_overrides",
    "as_dict",
    "as_list",
    "as_set",
    "as_tuple",
    "bind",
    "call",
    "ConfigList",
    "ConfigSet",
    "ConfigTuple",
    "get_env",
    "instantiate",
    "LAZY_TARGET",
    "LazyCall",
    "LazyObject",
    "load_config",
    "make_dict",
    "make_list",
    "make_set",
    "make_tuple",
    "save_config",
    "KEY_VERSION",
    "KEY_MODEL",
    "KEY_DATASET",
    "KEY_NAME",
    "KEY_SESSION_ID",
]


######################
# Configuration keys #
######################

KEY_VERSION: T.Final = "VERSION"
KEY_MODEL: T.Final = "MODEL"
KEY_DATASET: T.Final = "DATASET"
KEY_NAME: T.Final = "name"
KEY_SESSION_ID: T.Final = "session_id"

####################
# Environment vars #
####################

_R = T.TypeVar("_R", int, str, bool)


class EnvFilter(enum.StrEnum):
    STRING = enum.auto()
    TRUTHY = enum.auto()
    FALSY = enum.auto()
    POSITIVE = enum.auto()
    NEGATIVE = enum.auto()
    NONNEGATIVE = enum.auto()
    NONPOSITIVE = enum.auto()

    @staticmethod
    def apply(f: EnvFilter | str, v: T.Any, /) -> bool:
        if v is None:
            return False
        match EnvFilter(f):
            case EnvFilter.STRING:
                assert isinstance(v, str)
                v = v.lower()
                return v != ""
            case EnvFilter.TRUTHY:
                return bool(v)
            case EnvFilter.FALSY:
                return not bool(v)
            case EnvFilter.POSITIVE:
                return v > 0
            case EnvFilter.NEGATIVE:
                return v < 0
            case EnvFilter.NONNEGATIVE:
                return v >= 0
            case EnvFilter.NONPOSITIVE:
                return v <= 0
            case _:
                msg = f"Invalid filter: {f!r}"
                raise ValueError(msg)


@T.overload
def get_env(
    __type: type[_R], /, *keys: str, default: _R, filter: EnvFilter = EnvFilter.TRUTHY
) -> _R:
    ...


@T.overload
def get_env(
    __type: type[_R],
    /,
    *keys: str,
    default: _R | None = None,
    filter: EnvFilter = EnvFilter.TRUTHY,
) -> _R | None:
    ...


def get_env(
    __type: type[_R],
    /,
    *keys: str,
    default: _R | None = None,
    filter: EnvFilter = EnvFilter.TRUTHY,
) -> _R | None:
    """
    Read an environment variable. If the variable is not set, return the default value.

    If no default is given, an error is raised if the variable is not set.
    """
    for k in keys:
        v = os.getenv(k)
        if v is None:
            continue
        if __type is bool:
            v = bool(strtobool(v))
        else:
            v = __type(v)
        if not EnvFilter.apply(filter, v):
            continue
        break
    else:
        v = default
    return T.cast(_R, v)


######################
# Lazy configuration #
######################

# Some global constants for lazy configuration
LAZY_TARGET: T.Final = "_target_"
_PACKAGE_PREFIX: T.Final = "_config_"
_OMEGA_DICT_FLAGS: T.Final = {"allow_objects": True}

# Type alias for a DictConfig that is used to store a lazy configuration.
LazyConfigDict: T.TypeAlias = DictConfig

# Type vars
_C = T.TypeVar("_C", DictConfig, ListConfig, contravariant=True)
_P = T.ParamSpec("_P")
_L = T.TypeVar("_L")


# See detectron2 implementation
class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, **kwargs):
        if D.is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = generate_path(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags=_OMEGA_DICT_FLAGS)


def _apply_recursive(cfg: _C, func: T.Callable[[_C], None]) -> None:
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _apply_recursive(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _apply_recursive(v, func)


def _validate_syntax(filename):
    """
    Validate the syntax of a Python-based configuration file.

    Adapted from: https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    """
    with file_io.open(filename, "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e


def _as_omegadict(obj: dict | DictConfig) -> DictConfig:
    if isinstance(obj, dict):
        return DictConfig(obj, flags=_OMEGA_DICT_FLAGS)
    return obj


def _generate_packagename(path: str):
    # generate a random package name when loading config files
    return _PACKAGE_PREFIX + str(uuid.uuid4())[:4] + "." + os.path.basename(path)


@contextmanager
def _patch_import():
    """
    Context manager that patches ``builtins.__import__`` to:
    - locate files purely based on relative location, regardless of packages.
            e.g. you can import file without having __init__
    - do not cache modules globally; modifications of module states has no side effect
    - support other storage system through file_io, so config files can be in the cloud
    - imported dict are turned into omegaconf.DictConfig automatically
    """
    import importlib.machinery
    import importlib.util

    # Reference the 'original' import function such that we can revert to normal behavior
    # after exiting the context manager.
    import_default = builtins.__import__

    def find_relative(original_file, relative_import_path, level):
        # NOTE: "from . import x" is not handled. Because then it's unclear
        # if such import should produce `x` as a python module or DictConfig.
        # This can be discussed further if needed.
        relative_import_err = (
            "Relative import of directories is not allowed within config files. "
            "Within a config file, relative import can only import other config files."
        )
        if not len(relative_import_path):
            raise ImportError(relative_import_err)

        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not file_io.isfile(cur_file):
            cur_file_no_suffix = cur_file[: -len(".py")]
            if file_io.isdir(cur_file_no_suffix):
                raise ImportError(
                    f"Cannot import from {cur_file_no_suffix}." + relative_import_err
                )
            else:
                raise ImportError(
                    f"Cannot import name {relative_import_path} from "
                    f"{original_file}: {cur_file} does not exist."
                )
        return cur_file

    def import_patched(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_PACKAGE_PREFIX)
        ):
            cur_file = find_relative(globals["__file__"], name, level)
            _validate_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(
                _generate_packagename(cur_file), None, origin=cur_file
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with file_io.open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = _as_omegadict(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return import_default(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = import_patched
    yield import_patched
    builtins.__import__ = import_default


def _filepath_to_name(path: Pathable) -> str | None:
    """
    Convert a file path to a module name.
    """

    configs_root = file_io.Path("//configs/").resolve()
    path = file_io.Path(path).resolve()
    try:
        # If the file is under "//configs", then we can use the relative path to generate a name
        name = path.relative_to(configs_root).parent.as_posix() + "/" + path.stem
    except Exception:
        # Otherwise, we use the absolute path and find the name using the filename without suffix
        name = "/".join([path.parent.stem, path.stem])

    name = name.replace("./", "")
    name = name.replace("//", "/")

    if name in {"__init__", "defaults", "unknown", "config", "configs"}:
        return None
    else:
        return name


def load_config_remote(path: str):
    from unipercept.integrations.wandb_integration import WANDB_RUN_PREFIX
    from unipercept.integrations.wandb_integration import read_run as wandb_read_run

    if path.startswith(WANDB_RUN_PREFIX):
        run = wandb_read_run(path)
        config = DictConfig(run.config)
        return config

    raise FileNotFoundError(path)


def load_config(path: str) -> DictConfig:
    """
    Load a config file.

    Parameters
    ----------
    path
        The path to the config file. The file extension must be either ".py" or ".yaml".

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the file does not exist.

    """
    from unipercept import __version__ as up_version

    path = file_io.get_local_path(path)
    if path is None or not file_io.isfile(path):
        return load_config_remote(path)

    ext = os.path.splitext(path)[1]
    match ext.lower():
        case ".py":
            _validate_syntax(path)

            with _patch_import():
                # Record the filename
                nsp = {
                    "__file__": path,
                    "__package__": _generate_packagename(path),
                }
                with file_io.open(path) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, path, "exec"), nsp)

            export = nsp.get(
                "__all__",
                (
                    k
                    for k, v in nsp.items()
                    if not k.startswith("_")
                    and (
                        isinstance(
                            v,
                            (dict, list, DictConfig, ListConfig, int, float, str, bool),
                        )
                        or v is None
                    )
                ),
            )
            obj: dict[str, T.Any] = {k: v for k, v in nsp.items() if k in export}
            obj.setdefault("name", _filepath_to_name(path))
            obj.setdefault(KEY_VERSION, up_version)

        case ".yaml":
            with file_io.open(path) as f:
                obj = yaml.unsafe_load(f)
            obj.setdefault("name", "unknown")
            obj.setdefault(KEY_VERSION, "unknown")
        case _:
            msg = "Unsupported file extension %s!"
            raise ValueError(msg, ext)

    return _as_omegadict(obj)


def save_config(cfg, path: str):
    """
    Save a config object to a yaml file.

    Parameters
    ----------
    cfg
        An omegaconf config object.
    filename
        The file name to save the config file.

    Notes
    -----
    When the config dictionary contains complex objects (e.g. lambda), it cannot be saved to yaml.
    In that case, an error will be printed and the config will be saved to a pkl file instead.
    """
    if not isinstance(cfg, DictConfig):
        cfg = _as_omegadict(D.asdict(cfg) if D.is_dataclass(cfg) else cfg)

    try:
        cfg = deepcopy(cfg)
    except Exception:
        pass
    else:
        # if it's deep-copyable, then...
        def _replace_type_by_name(x):
            if "_target_" in x and callable(x._target_):
                try:
                    x._target_ = generate_path(x._target_)
                except AttributeError:
                    pass

        # not necessary, but makes yaml looks nicer
        _apply_recursive(cfg, _replace_type_by_name)

    try:
        dict = OmegaConf.to_container(
            cfg,
            # Do not resolve interpolation when saving, i.e. do not turn ${a} into
            # actual values when saving.
            resolve=False,
            # Save structures (dataclasses) in a format that can be instantiated later.
            # Without this option, the type information of the dataclass will be erased.
            structured_config_mode=SCMode.INSTANTIATE,
        )
        dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True)
        with file_io.open(path, "w") as f:
            f.write(dumped)

        _ = yaml.unsafe_load(dumped)  # test that it is loadable
    except Exception as err:
        raise SyntaxError(f"Config file {path} cannot be saved to yaml!") from err


def apply_overrides(cfg, overrides: List[str]):
    """
    In-place override contents of cfg.

    Parameters
    ----------
    cfg
        An omegaconf config object
    overrides
        List of strings in the format of "a=b" to override configs.
        See: https://hydra.cc/docs/next/advanced/override_grammar/basic/

    Returns
    -------
    DictConfig
        Lazy configuration object
    """
    from hydra.core.override_parser.overrides_parser import OverridesParser

    def safe_update(cfg, key, value):
        parts = key.split(".")
        for idx in range(1, len(parts)):
            prefix = ".".join(parts[:idx])
            v = OmegaConf.select(cfg, prefix, default=None)
            if v is None:
                break
            if not OmegaConf.is_config(v):
                raise KeyError(
                    f"Trying to update key {key}, but {prefix} "
                    f"is not a config, but has type {type(v)}."
                )
        OmegaConf.update(cfg, key, value, merge=True)

    for o in OverridesParser.create().parse_overrides(overrides):
        key = o.key_or_group
        value = o.value()
        if o.is_delete():
            raise NotImplementedError("deletion is not yet a supported override")
        safe_update(cfg, key, value)

    return cfg


# HACK: This is a workaround for a bug in OmegaConf, where lazily called objects are incompatible with the structured
# config system. This is a temporary solution until the bug is fixed.
if T.TYPE_CHECKING:

    class LazyObject(T.Generic[_L]):
        def __getattr__(self, name: str) -> T.Any:
            ...

        @override
        def __setattr__(self, __name: str, __value: Any) -> None:
            ...

else:
    import types

    class LazyObject(T.Dict[str, T.Any]):
        def __class_getitem__(cls, item: T.Any) -> T.Dict[str, T.Any]:
            return types.GenericAlias(dict, (str, T.Any))


def bind(func: T.Callable[_P, _L], /) -> T.Callable[_P, LazyObject[_L]]:
    """
    The method ``lazy_bind`` is a wrapper around call with type hints that support use in OmegaConf's structured
    configuration system.
    """
    return call(func)  # type: ignore


class _LazyCall(T.Generic[_P, _L]):
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.
    """

    def __init__(self, target: T.Callable[_P, _L]):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(
                f"target of LazyCall must be a callable or defines a callable! Got {target}"
            )
        self._target = target

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> DictConfig:
        if D.is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = generate_path(self._target)
        else:
            target = self._target
        kwargs[LAZY_TARGET] = target

        return DictConfig(content=kwargs, flags=_OMEGA_DICT_FLAGS)


def call(func: T.Callable[_P, _L], /) -> T.Callable[_P, _L]:
    """
    Wrap a callable object so that it can be lazily called.

    Parameters
    ----------
    func : callable
        The callable object to be wrapped.

    Returns
    -------
    callable
        A lazily callable object.

    Notes
    -----
    The returned callable object can be called with arbitrary arguments
    and keyword arguments. The actual call to the wrapped function is
    deferred until the returned object is called.
    """
    return _LazyCall(func)  # type: ignore


_INST_SEQ_TYPEMAP: dict[type, type] = {
    omegaconf.ListConfig: list,
    list: list,
    tuple: tuple,
    set: set,
    frozenset: frozenset,
}


def migrate_target(v: T.Any) -> T.Any:
    if isinstance(v, str):
        match v:
            case "unipercept.utils.catalog.DataManager.get_info_at":
                return "unipercept.get_info_at"
            case _:
                pass
    return v


@T.overload
def instantiate(cfg: T.Sequence[LazyObject[_L]], /) -> T.Sequence[_L]:
    ...


@T.overload
def instantiate(cfg: LazyObject[_L], /) -> _L:
    ...


@T.overload
def instantiate(cfg: T.Mapping[T.Any, LazyObject[_L]], /) -> T.Mapping[T.Any, _L]:
    ...


def instantiate(cfg: T.Any, /) -> T.Any:
    """
    Recursively instantiate objects defined in dictionaries by "_target_" and
    arguments.

    Our version differs from Detectron2's in that it never returns a
    configuration object, but always returns the instantiated object (e.g. a
    ListConfig is always converted to a list).
    """
    if cfg is None or isinstance(
        cfg,
        (
            int,
            float,
            bool,
            str,
            set,
            frozenset,
            bytes,
            type,
            types.NoneType,
            types.FunctionType,
        ),
    ):
        return cfg  # type: ignore

    if isinstance(cfg, T.Sequence) and not isinstance(cfg, (T.Mapping, str, bytes)):
        cls = type(cfg)
        cls = _INST_SEQ_TYPEMAP.get(cls, cls)
        return cls(instantiate(x) for x in cfg)

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, omegaconf.DictConfig) and D.is_dataclass(
        cfg._metadata.object_type
    ):
        return omegaconf.OmegaConf.to_object(cfg)

    if isinstance(cfg, T.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = migrate_target(cls)
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate_object(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:  # noqa: B902, PIE786
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            return cls(**cfg)
        except TypeError as err:
            msg = f"Error instantiating {cls_name} with arguments {cfg}!"
            raise TypeError(msg) from err

    if isinstance(cfg, (dict, omegaconf.DictConfig)):
        return {k: instantiate(v) for k, v in cfg.items()}  # type: ignore

    if callable(cfg):
        return cfg

    err = f"Cannot instantiate {cfg}, type {type(cfg)}!"
    raise ValueError(err)


def make_dict(**kwargs) -> dict[str, T.Any]:
    return dict(**kwargs)


class ConfigSet(set):
    pass


def make_set(items) -> set[T.Any]:
    items = (
        omegaconf.OmegaConf.to_object(items)
        if isinstance(items, omegaconf.ListConfig)
        else items
    )
    return ConfigSet(i for i in items)  # type: ignore


class ConfigTuple(tuple):
    pass


_T = T.TypeVar("_T", bound=tuple, covariant=True)


def make_tuple(items) -> _T:
    items = (
        omegaconf.OmegaConf.to_object(items)
        if isinstance(items, omegaconf.ListConfig)
        else items
    )
    return ConfigTuple(i for i in items)  # type: ignore


class ConfigList(list):
    pass


def make_list(items) -> list[T.Any]:
    items = (
        omegaconf.OmegaConf.to_object(items)
        if isinstance(items, omegaconf.ListConfig)
        else items
    )
    return ConfigList(i for i in items)  # type: ignore


def wrap_on_result(func_wrap, func_next, **kwargs_next):
    """
    Run a function on the result of another function. Useful in configuration files when
    you want to wrap a function on the result of another function, without having to change
    the keys of the configuration file.
    """

    def wrapper(*args, **kwargs):
        return func_next(func_wrap(*args, **kwargs), **kwargs_next)

    return wrapper


def as_dict(**kwargs):
    return call(make_dict)(*kwargs)


def as_set(*args):
    return call(make_set)(items=args)


def as_tuple(*args):
    return call(make_tuple)(items=args)


def as_list(*args):
    return call(make_list)(items=args)
