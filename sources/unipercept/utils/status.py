from __future__ import annotations

import contextlib
import enum as E
import functools
import typing as T

__all__ = ["assert_status", "modify_status", "pop_status", "put_status", "StatusDescriptor"]

_O_contra = T.TypeVar("_O_contra", bound=object, contravariant=True)
_P = T.ParamSpec("_P")
_R_co = T.TypeVar("_R_co", covariant=True)
_S = T.TypeVar("_S", bound=E.IntFlag, contravariant=True)
_StatusDecoFuncType: T.TypeAlias = T.Callable[T.Concatenate[_O_contra, _P], _R_co]
_StatusAttrType: T.TypeAlias = str | property | T.Any


def _get_error_bad_attr(attr: T.Any) -> TypeError:
    return TypeError(f"Invalid type {type(attr)} for status attribute.")


def _get_attr(obj: _O_contra, attr: _StatusAttrType, default=None) -> int:
    if isinstance(attr, property):
        value = attr.fget(obj)
    elif isinstance(attr, str):
        value = getattr(obj, attr, default)
    elif isinstance(attr, StatusDescriptor):
        value = attr.__get__(obj)
    else:
        raise _get_error_bad_attr(attr)
    if value is None:
        raise AttributeError(f"{obj} has no attribute {attr}.")
    return value


def _set_attr(obj: _O_contra, attr: _StatusAttrType, status: int) -> None:
    if isinstance(attr, property):
        attr.fset(obj, status)
    elif isinstance(attr, str):
        setattr(obj, attr, status)
    elif isinstance(attr, StatusDescriptor):
        attr.__set__(obj, status)
    else:
        raise _get_error_bad_attr(attr)


def _put_status(obj: _O_contra, attr, status) -> None:
    _set_attr(obj, attr, _get_attr(obj, attr) | status)


def _pop_status(obj: _O_contra, attr, status) -> None:
    value = _get_attr(obj, attr)
    assert value & status, f"Status {status} not set in {value}."
    _set_attr(obj, attr, value & ~status)


def assert_status(
    attr: _StatusAttrType, status: int, on_exit: bool = False
) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
    """
    Decorator that marks a method as only callable when the engine is in a certain status.

    Parameters
    ----------
    attr : _StatusAttrType
        The attribute to check the status of.
    status : _StatusType
        The status to check for.
    on_exit : bool, optional
        Whether to check the status on exit, by default False
    """

    def decorator(func: _StatusDecoFuncType) -> _StatusDecoFuncType:
        def check_status(obj: _O_contra) -> bool:
            if not _get_attr(obj, attr) & status:
                raise AssertionError(f"Method {func.__name__} requires status {status}")

        @functools.wraps(func)
        def wrapper(obj: _O_contra, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            with contextlib.ExitStack() as stack:
                if on_exit:
                    stack.callback(check_status, obj)
                else:
                    check_status(obj)
                return func(obj, *args, **kwargs)

        return wrapper

    return decorator


def with_status(attr: _StatusAttrType, status: int) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
    """
    Decorator that applies a status to an attribute, and removes it on exit.
    """

    def decorator(func: _StatusDecoFuncType) -> _StatusDecoFuncType:
        @functools.wraps(func)
        def wrapper(obj: _O_contra, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            with contextlib.ExitStack() as stack:
                _put_status(obj, attr, status)
                stack.callback(_pop_status, obj, attr, status)
                return func(obj, *args, **kwargs)

        return wrapper

    return decorator


def modify_status(
    attr: _StatusAttrType, status: int, *, on_exit=False, action: T.Literal["put", "pop"] = "put"
) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
    """
    Decorator that applies a status to an attribute.

    Parameters
    ----------
    attr : _StatusAttrType
        The attribute to apply the status to.
    status : _StatusType
        The status to apply.
    on_exit : bool, optional
        Whether to apply the status on exit, by default False
    action : str, optional
        Modification action, by default "put", can be "put" or "pop"
    """
    match action:
        case "put":
            apply_fn = _put_status
        case "pop":
            apply_fn = _pop_status
        case _:
            raise ValueError(f"Invalid action {action}.")

    def decorator(func: _StatusDecoFuncType) -> _StatusDecoFuncType:
        @functools.wraps(func)
        def wrapper(obj: _O_contra, *args: _P.args, **kwargs: _P.kwargs) -> _R_co:
            with contextlib.ExitStack() as stack:
                if on_exit:
                    stack.callback(apply_fn, obj, attr, status)
                else:
                    apply_fn(obj, attr, status)
                return func(obj, *args, **kwargs)

        return wrapper

    return decorator


def pop_status(
    attr: _StatusAttrType, status: int, *, on_exit=False
) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
    """
    Decorator that removes a status on entering or exiting a method.
    """
    return modify_status(attr, status, on_exit=on_exit, action="pop")


def put_status(
    attr: _StatusAttrType, status: int, *, on_exit=False
) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
    """
    Decorator that applies a status on entering or exiting a method.
    """
    return modify_status(attr, status, on_exit=on_exit, action="put")


class StatusDescriptor(T.Generic[_S]):
    """
    Descriptor class for status attributes. This simplifies the use of status attributes in classes, by saving the
    attribute name in the descriptor.
    """

    def __init__(self, kind: type[_S], default: _S | int | None = None) -> None:
        self._kind = kind
        self._name = None
        self._owner = None
        self._default = default

    def __set_name__(self, owner, name) -> None:
        if self._owner is not None:
            raise RuntimeError(f"StatusAttribute {self} already bound to {self._owner}.")
        self._owner = owner
        self._name = name

    def __get__(self, obj: _O_contra, objtype=None) -> _S:
        value = obj.__dict__.get(self._name, self._default)
        if value is None:
            raise AttributeError(f"{obj} has no attribute {self._name}.")
        return value

    def __set__(self, obj, value: _S | int) -> None:
        obj.__dict__[self._name] = value

    def __call__(self, status: _S | int) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
        """
        Run the decorated method with a certain status.
        """
        return with_status(self, status)

    def __iter__(self) -> T.Iterator[_S]:
        """
        Returns each individual status flag present in the attribute.
        """
        for flag in self._kind:
            if self & flag:
                yield flag

    def __len__(self) -> int:
        """
        Returns the number of status flags present in the attribute.
        """
        return sum(1 for _ in self)

    def __contains__(self, status: _S | int) -> bool:
        """
        Returns whether a status flag is present in the attribute.
        """
        return bool(self & status)

    def __getitem__(self, status: _S | int) -> bool:
        """
        Returns whether a status flag is present in the attribute.
        """
        return bool(self & status)

    def __setitem__(self, status: _S | int, value: bool) -> None:
        """
        Sets a status flag in the attribute.
        """
        if value:
            self |= status
        else:
            self &= ~status

    def pop_status(self, status: _S | int, on_exit=False) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
        """
        Remove a status on entering or exiting a method.
        """
        return pop_status(self, status, on_exit=on_exit)

    def put_status(self, status: _S | int, on_exit=False) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
        """
        Apply a status on entering or exiting a method.
        """
        return put_status(self, status, on_exit=on_exit)

    def modify_status(
        self, status: _S | int, on_exit=False, action: T.Literal["put", "pop"] = "put"
    ) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
        """
        Modify a status on entering or exiting a method.
        """
        return modify_status(self, status, on_exit=on_exit, action=action)

    def assert_status(self, status: _S | int, on_exit=False) -> T.Callable[[_StatusDecoFuncType], _StatusDecoFuncType]:
        """
        Assert that a status is set on entering or exiting a method.
        """
        return assert_status(self, status, on_exit=on_exit)
