"""Utilities for dealing with the types defined in this package."""

from __future__ import annotations

import typing
from enum import StrEnum
from typing import Any, Iterable, NoReturn, TypeAlias, TypedDict, TypeGuard, TypeVar

from typing_extensions import override

__all__ = []

_Data: TypeAlias = TypedDict
_Type = TypeVar("_Type", bound=TypedDict)


class InvalidKeyError(Exception):
    """
    Raised when a dictionary has invalid keys for a TypedDict.

    This is used instead of a regular KeyError to provide more information, and because
    the use-case is different from a regular KeyError in that the origin is likely
    not a programming error, but rather due to the input data from an external source.
    """

    def __init__(
        self, d: _Data, t: type[TypedDict], /, parents: tuple[str] | None = None
    ) -> None:
        self.parents = parents or ()
        self.name = t.__qualname__ or t.__name__
        self.keys_provided = frozenset(d.keys())
        self.keys_required = t.__required_keys__
        self.keys_optional = t.__optional_keys__

    @property
    def keys_missing(self) -> frozenset[str]:
        return self.keys_required - self.keys_provided

    @property
    def keys_available(self) -> frozenset[str]:
        return self.keys_required | self.keys_optional

    @property
    def keys_unknown(self) -> frozenset[str]:
        return self.keys_provided - (self.keys_required | self.keys_optional)

    @override
    def __str__(self) -> str:
        def _to_str(s: Iterable[str]) -> str:
            return "{" + ", ".join(sorted(s)) + "}"

        err_msg = (
            f"Invalid dictionary keys {_to_str(self.keys_provided)} for typed "
            f"dict {self.name}, expected {_to_str(self.keys_required)} and "
            f"optionally {_to_str(self.keys_optional)}."
        )

        if len(self.keys_missing) > 0:
            str_missing = _to_str(self.keys_missing)
            err_msg += f" Missing keys: {str_missing}."
        if len(self.keys_unknown) > 0:
            str_unknown = _to_str(self.keys_unknown)
            err_msg += f" Unknown keys: {str_unknown}."

        return err_msg


class InvalidTypeError(Exception):
    """
    Raised when a dictionary has invalid types for a TypedDict.

    This is used instead of a regular TypeError to provide more information, and because
    the use-case is different from a regular TypeError in that the origin is likely
    not a programming error, but rather due to the input data from an external source.
    """

    def __init__(
        self,
        d: _Data,
        t: type[TypedDict],
        key_invalid: Any,
        /,
        parents: tuple[str] | None = None,
    ) -> None:
        self.parents = parents or ()
        self.name = t.__qualname__ or t.__name__
        self.key_invalid = key_invalid
        self.type_provided = type(d[key_invalid])
        self.type_required = t.__annotations__[key_invalid]

    @override
    def __str__(self) -> str:
        return (
            f"Invalid dictionary type {self.type_provided} for key {self.key_invalid} "
            f"in typed dict {self.name}, expected {self.type_required}."
        )


def check_typeddict_keys(d: _Data, t: type[_Type], /) -> NoReturn | None:
    """
    Run a check on the keys of a dictionary to verify that they are a subset of the keys of a TypedDict.
    If this is not the case, an error is raised.

    Parameters
    ----------
    d : dict
        The dictionary to verify.
    t : TypedDict
        The TypedDict to verify against.

    Raises
    ------
    InvalidKeysError
        If the dictionary is not a subset of the TypedDict.
    """
    keys_required = t.__required_keys__
    keys_optional = t.__optional_keys__
    keys_given = set(d.keys())

    if not keys_required.issubset(keys_given) and keys_given.issubset(
        keys_required | keys_optional
    ):
        raise InvalidKeyError(d, t)


class OnUnresolved(StrEnum):
    """An enum that specifies the action to take when a type is not a class/alias/typing-type but a string."""

    SKIP = "skip"
    WARN = "warn"
    RAISE = "raise"


def check_typeddict(
    d: _Data,
    t: type[TypedDict],
    *,
    on_unresolved: str | OnUnresolved = OnUnresolved.WARN,
    use_metadata=True,
) -> NoReturn | None:
    """
    Check that a dictionary matches a TypedDict definition by recursively checking keys and value types.

    Parameters
    ----------
    d : dict
        The dictionary to check.
    t : TypedDict
        The TypedDict to check against.
    on_unresolved : str or OnUnresolved, optional
        The action to take when a type is not a class/alias/typing-type but a string, by default OnUnresolved.WARN.
    use_metadata : bool, optional
        Whether to use the metadata of Annotated types to perform an equality check against each item of the metadata,
        by default True. If False, Annotated types are treated as their base type.

    Returns
    -------
    Mapping[str, Callable[[Any], bool]
        A mapping of the dictionary keys to a function that checks if a value is of the runtime type.
    """

    # Input validation
    if not typing.is_typeddict(t):
        raise TypeError(
            f"Expected a TypedDict subclass, got {t.__qualname__ or t.__name__}!"
        )
    if not isinstance(d, dict):
        raise TypeError(f"Expected a dictionary, got {type(d).__qualname__}!")

    # Check the keys
    check_typeddict_keys(d, t)


#     # Check the types
#     for key, ann in get_type_hints(t, include_extras=True).items():  # i.e. t.__annotations__.items()
#         origin = typing.get_origin(ann)

#         # Catch cases of unresolvable types, e.g. those that are not classes but strings
#         if isinstance(ann, str) or isinstance(ann, typing.ForwardRef):
#             match OnUnresolved(on_unresolved):
#                 case OnUnresolved.RAISE:
#                     raise TypeError(f"Type {ann} is not resolvable, cannot check type for {key}!")
#                 case OnUnresolved.WARN:
#                     warnings.warn(f"Type {ann} is not resolvable, skipping type check for {key}!", stacklevel=2)
#                 case OnUnresolved.SKIP:
#                     pass
#                 case _:
#                     raise ValueError(f"Invalid value for `on_unresolvable`: {on_unresolved}!")
#             continue

#         # Check special TypedDict case of Required and NotRequired
#         if origin in (typing.NotRequired, typing.Required):
#             ann = typing.get_args(ann)[0]

#         # Get value in target dict
#         if origin is typing.NotRequired and key not in d:
#             continue
#         val = d[key]

#         if not value_is(val, ann):
# # Check for Annotated types
#         if origin is typing.Annotated:
#             ann, ann_matches = typing.get_args(ann)[0]
#             if not any(val == val_matches for val_matches in ann_matches):
#                 raise InvalidTypeError(d, t, key)

#         # Catch cases of nested typed dicts
#         if typing.is_typeddict(ann):
#             check_typeddict(d[key], ann)
#             continue

#         # Sanitize the annotation
#         if origin is not None:
#             if type(origin) is abc.ABCMeta:
#                 ann = origin
#             elif origin is typing.Union:
#                 ann = typing.get_args(ann)
#             if origin is not None:
#                 ann = origin

#         # Check for Any/Never
#         if ann is typing.Any:
#             continue
#         if ann is typing.Never:
#             raise InvalidTypeError(d, t, key)

#         # Check for None
#         if ann is None:
#             if d[key] is not None:
#                 raise InvalidTypeError(d, t, key)

#         # Final instance check
#         try:
#             if not isinstance(val, ann):
#                 raise InvalidTypeError(d, t, key)
#         except TypeError as e:
#             raise ValueError(
#                 f"Annotated type {ann!r} ({key!r} in {t.__name__!r}) cannot be checked against provided value "
#                 f"of type {type(val)!r}!"
#             ) from e

#         # Check next key
#         continue


def is_dict_type(
    d: _Data, t: type[_Type], /, **typecheck_kwargs
) -> TypeGuard[type[_Type]]:
    """
    Check that a dictionary matches a TypedDict definition by recursively checking
    keys and value types.

    Returns a boolean (type guard) instead of raising an error.
    """

    try:
        check_typeddict(d, t, **typecheck_kwargs)
    except (InvalidKeyError, InvalidTypeError):
        return False
    return True
