import abc
import dataclasses as D
import functools
import typing as T
import warnings

import omegaconf
import regex as re
import typing_extensions as TX

from unipercept.config import lazy
from unipercept.utils.inspect import generate_path, locate_object

__all__ = [
    "call",
    "bind",
    "node",
    "ref",
    "partial",
    "islazy",
    "Dict",
    "Set",
    "Tuple",
    "List",
]


# ------- #
# Typings #
# ------- #

LazyConfigDict: T.TypeAlias = omegaconf.DictConfig

# --------------- #
# Config file API #
# --------------- #


class _LazyCall[**_P, _L]:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.
    """

    def __init__(self, target: T.Callable[_P, _L]):
        if not callable(target) and not isinstance(target, (str, T.Mapping)):
            msg = (
                "Expected a callable object, string or configuration dictionary, "
                f"got {target=} (type {type(target)})!"
            )
            raise TypeError(msg)
        self.target = target

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> LazyConfigDict:
        if {lazy.LAZY_TARGET, lazy.LAZY_ARGS} & kwargs.keys():
            msg = f"Cannot use reserved keys {lazy.LAZY_TARGET} or {lazy.LAZY_ARGS} in kwargs!"
            raise ValueError(msg)

        node = {}
        if D.is_dataclass(self.target):
            # omegaconf object cannot hold dataclass type
            # https://github.com/omry/omegaconf/issues/784
            node[lazy.LAZY_TARGET] = generate_path(self.target)
        else:
            node[lazy.LAZY_TARGET] = self.target
        if args and len(args) > 0:
            node[lazy.LAZY_ARGS] = tuple(args)

        node.update(kwargs)

        return omegaconf.DictConfig(content=node, flags=lazy.OMEGA_DICT_FLAGS)


def islazy(obj: T.Any, checks: T.Iterable[T.Callable] | T.Callable) -> bool:
    """
    Check if an object is a lazy call to a target callable.

    Parameters
    ----------
    obj : any
        The object to check.
    target : callable or str
        The target callable to check against.

    Returns
    -------
    bool
        Whether the object is a lazy call to the target callable.
    """
    if not isinstance(obj, omegaconf.DictConfig):
        return False
    tgt = obj.get(lazy.LAZY_TARGET, None)
    if tgt is None:
        return False
    if isinstance(tgt, str):
        tgt = locate_object(tgt)
    if callable(checks):
        checks = [checks]
    return any(tgt is check for check in checks)


def call[**_P, _R](func: T.Callable[_P, _R], /) -> T.Callable[_P, _R]:
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


def bind[**_P, _R](func: T.Callable[_P, _R], /) -> T.Callable[_P, lazy.LazyObject[_R]]:
    """
    Wrapper around call with type hints that support use in OmegaConf's structured
    configuration system.

    Primary use is the definition of root nodes that are also lazy calls.
    """
    return call(func)  # type: ignore


def pairs(
    node: LazyConfigDict | T.Mapping,
) -> T.Iterator[tuple[str, T.Any]]:
    r"""
    Key-value pairs from a configuration node, where special keys are ignored.
    """
    if not isinstance(node, omegaconf.DictConfig):
        msg = f"Expected a configuration node, got {node=} (type {type(node)})!"
        raise TypeError(msg)
    for k, v in node.items():
        if k in {lazy.LAZY_TARGET, lazy.LAZY_ARGS}:
            continue
        assert isinstance(k, str), type(k)
        yield str(k), v


@T.dataclass_transform(kw_only_default=True)
class NodeSpec:
    pass


def node[_T: NodeSpec](_: type[_T], /) -> type[_T]:
    """
    Uses a template :class:`ConfigNode` to define an OmegaConf node that has the same
    fields as the template


    The type checker will treat the resulting object as a :class:`ConfigNode`, but
    at runtime the resulting object is a regular OmegaConf node based on the fields
    of the template.
    """

    def _create_fake_class(**kwargs):
        return omegaconf.DictConfig(kwargs, flags=lazy.OMEGA_DICT_FLAGS)

    return _create_fake_class  # type: ignore


_R = T.TypeVar("_R", covariant=True)


def ref(target: str) -> _R:
    """
    Reference to another variable in the configuration using an OmegaConf interpolation
    string.
    """
    return T.cast(_R, target)


def wrap_on_result(func_wrap, func_next, **kwargs_next):
    """
    Run a function on the result of another function. Useful in configuration files when
    you want to wrap a function on the result of another function, without having to change
    the keys of the configuration file.
    """

    def wrapper(*args, **kwargs):
        return func_next(func_wrap(*args, **kwargs), **kwargs_next)

    return wrapper


def _call_partial(
    *,
    _callable_: T.Callable[..., T.Any] | str | None = None,
    **kwargs,  # type: ignore
) -> T.Callable[..., T.Any]:
    if _callable_ is None:
        cb = kwargs.pop("_func_", None)  # legacy support
    else:
        cb = _callable_
    if isinstance(cb, str):
        cb = locate_object(cb)
    if not callable(cb):
        msg = f"Expected a callable object or location (str), got {_callable_} (type {type(_callable_)}"
        raise TypeError(msg)
    return functools.partial(cb, **kwargs)


def partial(func: T.Callable[..., T.Any], /) -> T.Callable[..., T.Any]:
    """
    Partially apply a function with keyword arguments.

    Parameters
    ----------
    func : callable
        The function to partially apply.

    Returns
    -------
    callable
        A lazy callable object that is forwarded to ``functools.partial``.
    """
    return lambda **kwargs: call(_call_partial)(_callable_=func, **kwargs)


# ------ #
# Macros #
# ------ #

PATTERN_CONFIG_KEY_STRICT = r"^[a-zA-Z_][a-zA-Z0-9_]*$"


def _check_valid_key(key: str, *, warn: bool = True) -> bool:
    if not re.match(PATTERN_CONFIG_KEY_STRICT, key):
        if warn:
            msg = (
                f"Key '{key}' may possibly lead to bad interoperability! "
                "It is recommended that keys start with a letter or underscore, "
                "and can only contain letters, numbers, and underscores."
            )
            warnings.warn(msg)
        return False
    return True


class _PositionalMacro[_R](metaclass=abc.ABCMeta):
    r"""
    A macro that generates a :func:`call` from positional arguments (accepts a sequence
    of items).
    """

    def __new__(cls, *args: T.Any) -> _R:
        return call(cls.target)(items=list(args))

    @classmethod
    @abc.abstractmethod
    def target(cls, items: list[T.Any]) -> _R:
        raise NotImplementedError("Method 'target' must be implemented in subclasses!")


class _KeywordMacro[_R](metaclass=abc.ABCMeta):
    r"""
    A macro that generates a :func:`call` from keyword arguments (accepts a mapping of
    key-value pairs).
    """

    def __new__(cls, **kwargs: T.Any) -> _R:
        for k in kwargs.keys():
            _check_valid_key(k)
        return call(cls.target)(**kwargs)

    @classmethod
    @abc.abstractmethod
    def target(cls, **kwargs: T.Any) -> _R:
        raise NotImplementedError("Method 'target' must be implemented in subclasses!")


class Dict[_T](_KeywordMacro[dict[str, _T]]):
    @TX.override
    @classmethod
    def target(cls, **kwargs: _T):
        return {k: v for k, v in kwargs.items()}


class DictFromItems[_T](_PositionalMacro[dict[str, _T]]):
    @classmethod
    @TX.override
    def target(cls, items: list[tuple[str, _T]]):
        return {k: v for k, v in items}


class Set[_T](_PositionalMacro[set[_T]]):
    @classmethod
    @TX.override
    def target(cls, items: list[_T]):
        if isinstance(items, omegaconf.ListConfig):
            items = omegaconf.OmegaConf.to_object(items)  # type: ignore
            assert isinstance(items, T.Sequence), type(items)
        return set(i for i in items)


class Tuple[_T](_PositionalMacro[tuple[_T]]):
    def __new__(cls, *args):
        return call(cls.target)(items=list(args))

    @classmethod
    @TX.override
    def target(cls, items: T.Any) -> tuple:
        items = (
            omegaconf.OmegaConf.to_object(items)
            if isinstance(items, omegaconf.ListConfig)
            else items
        )
        return tuple(i for i in items)


class List[_T](_PositionalMacro[list[_T]]):
    def __new__(cls, *args):
        return call(cls.target)(items=list(args))

    @classmethod
    @TX.override
    def target(cls, items: T.Any) -> list:
        if isinstance(items, omegaconf.ListConfig):
            items = omegaconf.OmegaConf.to_object(items)  # type: ignore
            assert isinstance(items, T.Sequence), type(items)
        return list(i for i in items)
