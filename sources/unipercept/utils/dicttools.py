from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeGuard, TypeVar, is_typeddict

DictType = TypeVar("DictType", bound=dict)


def is_dict_type(obj: Any, type_: type[DictType]) -> TypeGuard[DictType]:
    """
    Type guard for a TypedDict. Checks if the given object is an instance of the given TypedDict.
    """
    if not is_typeddict(type_):
        raise TypeError(f"Expected TypedDict, got {type_}")

    if not isinstance(obj, dict):
        return False

    obj_keys = set(obj.keys())
    for key, ann in type_.__annotations__.items():
        # Key exists
        if key not in obj_keys:
            if key in type_.__required_keys__:
                return False
            continue
        obj_keys.remove(key)

        # Type is instance
        if is_typeddict(ann):
            if not is_dict_type(obj[key], ann):
                return False
        else:
            try:
                if isinstance(ann, (list, tuple, frozenset, set, Sequence)):
                    if not isinstance(obj[key], ann.__class__):
                        return False
                    for item in obj[key]:
                        if not isinstance(item, ann.__args__[0]):
                            return False
                elif isinstance(ann, type):
                    if not isinstance(obj[key], ann):
                        return False
                elif not isinstance(obj[key], ann):
                    return False
            except TypeError:
                pass

    # No extra keys
    if obj_keys:
        return False

    return True


from collections import defaultdict


def defaultdict_recurrent() -> defaultdict:
    """
    Returns a defaultdict that recurrently creates defaultdicts.
    """
    return defaultdict(defaultdict_recurrent)


def defaultdict_recurrent_to_dict(d: defaultdict) -> dict:
    """
    Returns a dict from a recurrently created defaultdict
    """

    if not isinstance(d, defaultdict):
        return d

    return {k: defaultdict_recurrent_to_dict(v) for k, v in d.items()}
