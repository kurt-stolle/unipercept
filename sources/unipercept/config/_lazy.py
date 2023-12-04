"""
Lazy configuration system, inspired by and based on Detectron2 and Hydra.
"""


from __future__ import annotations

import dataclasses
import pydoc
import typing as T
from typing import Any

import omegaconf
import torch.nn as nn
from typing_extensions import override

import ast
import builtins
import collections.abc as abc
import importlib
import inspect
import logging
import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import is_dataclass
from typing import List, Tuple, Union
import cloudpickle
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf, SCMode
from unicore import file_io

__all__ = [
    "LazyCall",
    "LazyConfig",
    "call",
    "bind",
    "LazyObject",
    "instantiate",
    "locate",
    "use_norm",
    "use_activation",
    "as_dict",
    "as_set",
    "as_tuple",
    "as_list",
    "ConfigSet",
    "ConfigTuple",
    "ConfigList",
    "make_dict",
    "make_set",
    "make_tuple",
    "make_list",
    "LazyConfig",
]


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        if is_dataclass(self._target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            target = _convert_target_to_string(self._target)
        else:
            target = self._target
        kwargs["_target_"] = target

        return DictConfig(content=kwargs, flags={"allow_objects": True})


def _visit_dict_config(cfg, func):
    """
    Apply func recursively to all DictConfig in cfg.
    """
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _validate_py_syntax(filename):
    # see also https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/config.py
    with file_io.open(filename, "r") as f:
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Config file {filename} has syntax error!") from e


def _cast_to_config(obj):
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


_CFG_PACKAGE_NAME = "detectron2._cfg_loader"
"""
A namespace to put all imported config into.
"""


def _random_package_name(filename):
    # generate a random package name when loading config files
    return _CFG_PACKAGE_NAME + str(uuid.uuid4())[:4] + "." + os.path.basename(filename)


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. support other storage system through file_io, so config files can be in the cloud
    4. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        # NOTE: "from . import x" is not handled. Because then it's unclear
        # if such import should produce `x` as a python module or DictConfig.
        # This can be discussed further if needed.
        relative_import_err = """
Relative import of directories is not allowed within config files.
Within a config file, relative import can only import other config files.
""".replace(
            "\n", " "
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
                raise ImportError(f"Cannot import from {cur_file_no_suffix}." + relative_import_err)
            else:
                raise ImportError(
                    f"Cannot import name {relative_import_path} from " f"{original_file}: {cur_file} does not exist."
                )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            _validate_py_syntax(cur_file)
            spec = importlib.machinery.ModuleSpec(_random_package_name(cur_file), None, origin=cur_file)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with file_io.open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = _cast_to_config(module.__dict__[name])
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load_rel(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Similar to :meth:`load()`, but load path relative to the caller's
        source file.

        This has the same functionality as a relative import, except that this method
        accepts filename as a string, so more characters are allowed in the filename.
        """
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return LazyConfig.load(filename, keys)

    @staticmethod
    def load(filename: str, keys: Union[None, str, Tuple[str, ...]] = None):
        """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
        has_keys = keys is not None
        filename = filename.replace("/./", "/")  # redundant
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")
        if filename.endswith(".py"):
            _validate_py_syntax(filename)

            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _random_package_name(filename),
                }
                with file_io.open(filename) as f:
                    content = f.read()
                # Compile first with filename to:
                # 1. make filename appears in stacktrace
                # 2. make load_rel able to find its parent's (possibly remote) location
                exec(compile(content, filename, "exec"), module_namespace)

            ret = module_namespace
        else:
            with file_io.open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        if has_keys:
            if isinstance(keys, str):
                return _cast_to_config(ret[keys])
            else:
                return tuple(_cast_to_config(ret[a]) for a in keys)
        else:
            if filename.endswith(".py"):
                # when not specified, only load those that are config objects
                ret = DictConfig(
                    {
                        name: _cast_to_config(value)
                        for name, value in ret.items()
                        if isinstance(value, (DictConfig, ListConfig, dict)) and not name.startswith("_")
                    },
                    flags={"allow_objects": True},
                )
            return ret

    @staticmethod
    def save(cfg, filename: str):
        """
        Save a config object to a yaml file.
        Note that when the config dictionary contains complex objects (e.g. lambda),
        it can't be saved to yaml. In that case we will print an error and
        attempt to save to a pkl file instead.

        Args:
            cfg: an omegaconf config object
            filename: yaml file name to save the config file
        """
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass
        else:
            # if it's deep-copyable, then...
            def _replace_type_by_name(x):
                if "_target_" in x and callable(x._target_):
                    try:
                        x._target_ = _convert_target_to_string(x._target_)
                    except AttributeError:
                        pass

            # not necessary, but makes yaml looks nicer
            _visit_dict_config(cfg, _replace_type_by_name)

        save_pkl = False
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
            dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
            with file_io.open(filename, "w") as f:
                f.write(dumped)

            try:
                _ = yaml.unsafe_load(dumped)  # test that it is loadable
            except Exception:
                logger.warning(
                    "The config contains objects that cannot serialize to a valid yaml. "
                    f"{filename} is human-readable but cannot be loaded."
                )
                save_pkl = True
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            save_pkl = True

        if save_pkl:
            new_filename = filename + ".pkl"
            try:
                # retry by pickle
                with file_io.open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config is saved using cloudpickle at {new_filename}.")
            except Exception:
                pass

    @staticmethod
    def apply_overrides(cfg, overrides: List[str]):
        """
        In-place override contents of cfg.

        Args:
            cfg: an omegaconf config object
            overrides: list of strings in the format of "a=b" to override configs.
                See https://hydra.cc/docs/next/advanced/override_grammar/basic/
                for syntax.

        Returns:
            the cfg object
        """

        def safe_update(cfg, key, value):
            parts = key.split(".")
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])
                v = OmegaConf.select(cfg, prefix, default=None)
                if v is None:
                    break
                if not OmegaConf.is_config(v):
                    raise KeyError(
                        f"Trying to update key {key}, but {prefix} " f"is not a config, but has type {type(v)}."
                    )
            OmegaConf.update(cfg, key, value, merge=True)

        try:
            from hydra.core.override_parser.overrides_parser import OverridesParser

            has_hydra = True
        except ImportError:
            has_hydra = False

        if has_hydra:
            parser = OverridesParser.create()
            overrides = parser.parse_overrides(overrides)
            for o in overrides:
                key = o.key_or_group
                value = o.value()
                if o.is_delete():
                    # TODO support this
                    raise NotImplementedError("deletion is not yet a supported override")
                safe_update(cfg, key, value)
        else:
            # Fallback. Does not support all the features and error checking like hydra.
            for o in overrides:
                key, value = o.split("=")
                try:
                    value = eval(value, {})
                except NameError:
                    pass
                safe_update(cfg, key, value)
        return cfg

    @staticmethod
    def to_py(cfg, prefix: str = "cfg."):
        """
        Try to convert a config object into Python-like psuedo code.

        Note that perfect conversion is not always possible. So the returned
        results are mainly meant to be human-readable, and not meant to be executed.

        Args:
            cfg: an omegaconf config object
            prefix: root name for the resulting code (default: "cfg.")


        Returns:
            str of formatted Python code
        """
        import black

        cfg = OmegaConf.to_container(cfg, resolve=True)

        def _to_str(obj, prefix=None, inside_call=False):
            if prefix is None:
                prefix = []
            if isinstance(obj, abc.Mapping) and "_target_" in obj:
                # Dict representing a function call
                target = _convert_target_to_string(obj.pop("_target_"))
                args = []
                for k, v in sorted(obj.items()):
                    args.append(f"{k}={_to_str(v, inside_call=True)}")
                args = ", ".join(args)
                call = f"{target}({args})"
                return "".join(prefix) + call
            elif isinstance(obj, abc.Mapping) and not inside_call:
                # Dict that is not inside a call is a list of top-level config objects that we
                # render as one object per line with dot separated prefixes
                key_list = []
                for k, v in sorted(obj.items()):
                    if isinstance(v, abc.Mapping) and "_target_" not in v:
                        key_list.append(_to_str(v, prefix=prefix + [k + "."]))
                    else:
                        key = "".join(prefix) + k
                        key_list.append(f"{key}={_to_str(v)}")
                return "\n".join(key_list)
            elif isinstance(obj, abc.Mapping):
                # Dict that is inside a call is rendered as a regular dict
                return (
                    "{"
                    + ",".join(f"{repr(k)}: {_to_str(v, inside_call=inside_call)}" for k, v in sorted(obj.items()))
                    + "}"
                )
            elif isinstance(obj, list):
                return "[" + ",".join(_to_str(x, inside_call=inside_call) for x in obj) + "]"
            else:
                return repr(obj)

        py_str = _to_str(cfg, prefix=[prefix])
        try:
            return black.format_str(py_str, mode=black.Mode())
        except black.InvalidInput:
            return py_str


# HACK: This is a workaround for a bug in OmegaConf, where lazily called objects are incompatible with the structured
# config system. This is a temporary solution until the bug is fixed.
if T.TYPE_CHECKING:
    _P = T.ParamSpec("_P")
    _L = T.TypeVar("_L")

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
    from detectron2.config import LazyCall as _L

    return _L(func)  # type: ignore


def _convert_target_to_string(t: Any) -> str:
    """
    Inverse of ``locate()``.

    Args:
        t: any object with ``__module__`` and ``__qualname__``
    """
    module, qualname = t.__module__, t.__qualname__

    # Compress the path to this object, e.g. ``module.submodule._impl.class``
    # may become ``module.submodule.class``, if the later also resolves to the same
    # object. This simplifies the string, and also is less affected by moving the
    # class implementation.
    module_parts = module.split(".")
    for k in range(1, len(module_parts)):
        prefix = ".".join(module_parts[:k])
        candidate = f"{prefix}.{qualname}"
        try:
            if locate(candidate) is t:
                return candidate
        except ImportError:
            pass
    return f"{module}.{qualname}"


def locate(name: str) -> T.Any:
    """
    Dynamically locates and returns an object by its fully qualified name.

    Based on Detectron2's `locate` function.

    Parameters
    ----------
    name (str):
        The fully qualified name of the object to locate.

    Returns
    -------
    Any:
        The located object.

    Raises
    ------
    ImportError
        If the object cannot be located.
    """
    obj = pydoc.locate(name)

    # Some cases (e.g. torch.optim.sgd.SGD) not handled correctly
    # by pydoc.locate. Try a private function from hydra.
    if obj is None:
        try:
            # from hydra.utils import get_method - will print many errors
            from hydra.utils import _locate
        except ImportError as e:
            raise ImportError(f"Cannot dynamically locate object {name}!") from e
        else:
            obj = _locate(name)  # it raises if fails

    return obj


_INST_SEQ_TYPEMAP: dict[type, type] = {
    omegaconf.ListConfig: list,
    list: list,
    tuple: tuple,
    set: set,
    frozenset: frozenset,
}


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
    Recursively instantiate objects defined in dictionaries by "_target_" and arguments.

    Our version differs from Detectron2's in that it never returns a configuration object, but always
    returns the instantiated object (e.g. a ListConfig is always converted to a list).

    """
    if cfg is None or isinstance(
        cfg, (int, float, bool, str, set, frozenset, bytes, type, types.NoneType, types.FunctionType)
    ):
        return cfg  # type: ignore

    if isinstance(cfg, T.Sequence) and not isinstance(cfg, (T.Mapping, str, bytes)):
        cls = type(cfg)
        cls = _INST_SEQ_TYPEMAP.get(cls, cls)
        return cls(instantiate(x) for x in cfg)

    # If input is a DictConfig backed by dataclasses (i.e. omegaconf's structured config),
    # instantiate it to the actual dataclass.
    if isinstance(cfg, omegaconf.DictConfig) and dataclasses.is_dataclass(cfg._metadata.object_type):
        return omegaconf.OmegaConf.to_object(cfg)

    if isinstance(cfg, T.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            return cls(**cfg)
        except TypeError as err:
            raise TypeError(f"Error instantiating {cls_name} with arguments {cfg}!") from err

    if isinstance(cfg, (dict, omegaconf.DictConfig)):
        return {k: instantiate(v) for k, v in cfg.items()}  # type: ignore

    if callable(cfg):
        return cfg

    raise ValueError(f"Cannot instantiate {cfg}, type {type(cfg)}!")


def make_dict(**kwargs) -> dict[str, T.Any]:
    return dict(**kwargs)


class ConfigSet(set):
    pass


def make_set(items) -> set[T.Any]:
    items = omegaconf.OmegaConf.to_object(items) if isinstance(items, omegaconf.ListConfig) else items
    return ConfigSet(i for i in items)  # type: ignore


class ConfigTuple(tuple):
    pass


def make_tuple(items) -> tuple[T.Any, ...]:
    items = omegaconf.OmegaConf.to_object(items) if isinstance(items, omegaconf.ListConfig) else items
    return ConfigTuple(i for i in items)  # type: ignore


class ConfigList(list):
    pass


def make_list(items) -> list[T.Any]:
    items = omegaconf.OmegaConf.to_object(items) if isinstance(items, omegaconf.ListConfig) else items
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


def use_norm(name: str):
    from unipercept.nn.layers.utils import wrap_norm

    return call(wrap_norm)(name=name)


def use_activation(module: type[nn.Module], inplace: T.Optional[bool] = None, **kwargs):
    from unipercept.nn.layers.utils import wrap_activation

    if inplace is not None:
        kwargs["inplace"] = inplace
    return call(wrap_activation)(module=module, **kwargs)


def as_dict(**kwargs):
    return call(make_dict)(*kwargs)


def as_set(*args):
    return call(make_set)(items=args)


def as_tuple(*args):
    return call(make_tuple)(items=args)


def as_list(*args):
    return call(make_list)(items=args)
