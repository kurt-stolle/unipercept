from __future__ import annotations

from typing import Any, Sequence, TypedDict, TypeGuard, TypeVar, is_typeddict

DictType = TypeVar("DictType", bound=TypedDict)


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
            else:
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
