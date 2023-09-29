"""
Lazy configuration system, inspired by and base don Detectron2 and Hydra.
"""


from __future__ import annotations

import dataclasses
import pydoc
import typing as T
from typing import Any

import omegaconf
import torch.nn as nn
from typing_extensions import override

__all__ = ["call", "bind", "LazyObject"]

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
    from unipercept.modeling.layers.utils import wrap_norm

    return call(wrap_norm)(name=name)


def use_activation(module: type[nn.Module], inplace: T.Optional[bool] = None, **kwargs):
    from unipercept.modeling.layers.utils import wrap_activation

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
