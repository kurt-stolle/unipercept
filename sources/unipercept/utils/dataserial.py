import json
from dataclasses import asdict, fields, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    dataclass_transform,
)

import cloudpickle
from typing_extensions import override

_T = TypeVar("_T", bound=object)


def to_dict(self: object) -> dict[str, Any]:
    """
    Get a dictionary from a class instance.
    """

    if not is_dataclass(self):
        raise TypeError(f"Expected dataclass, got {type(self)}")
    return asdict(self)


def to_json(self: object, **json_dumps_kwargs) -> str:
    """
    Get a JSON string from a class instance.
    """

    # Note: cannot use __dict__ as this does not cover the case of nested dataclasses and slots
    # data = {f.name: getattr(self, f.name) for f in fields(self)}
    # for key in data.keys():
    #     value = data[key]
    #     if is_dataclass(value):
    #         data[key] = to_json(value)
    data = to_dict(self)
    return json.dumps(data, **json_dumps_kwargs)


def from_dict(cls: type[_T], data: dict) -> _T:
    """
    Get a class instance from a dictionary.
    """
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass, got {type(cls)}")

    anns = {f.name: f for f in fields(cls)}

    if not data.keys() == anns.keys():
        raise TypeError(f"Expected keys {anns.keys()}, got {data.keys()}")

    for key in data.keys():
        ann = anns[key]
        assert isinstance(ann.type, type), f"Expected type, got {ann.type}"

        if is_dataclass(ann.type):
            data[key] = from_dict(ann.type, data[key])
            continue
    return cls(**data)  # type: ignore


def from_json(cls: type[_T], src: str) -> _T:
    """
    Get a class instance from a JSON string.
    """
    data = json.loads(src)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    return from_dict(cls, data)


class Serializable(Protocol):
    """
    Protocol for serializable dataclasses.
    """

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        ...

    @classmethod
    def from_json(cls, src: str) -> Self:
        ...

    def to_dict(self) -> dict[str, Any]:
        ...

    def to_json(self, **json_dumps_kwargs) -> str:
        ...


_S = TypeVar("_S", bound=Serializable)


class SerializeRegistry(Sequence[type[Serializable]]):
    """
    Registry for serialiable dataclasses. This is used to mark certain classes
    with a similar interface as target for deserialization of a source class
    with fewer fields.

    Overwrites the ``from_json`` and ``from_dict`` methods of all registered
    sources to support deserialization sources to targets.
    """

    def __init__(self):
        self._targets: list[type[Serializable]] = []

    @override
    def __iter__(self):
        return iter(self._targets)

    @override
    def __len__(self):
        return len(self._targets)

    @override
    def __getitem__(self, index):
        return self._targets[index]

    @override
    def __repr__(self):
        return f"{self.__class__.__name__}({self._targets})"

    @property
    def targets(self) -> tuple[type[Serializable], ...]:
        return tuple(self._targets)

    def add(self, cls: type[Serializable]) -> None:
        for idx, cls_target in enumerate(self._targets):
            if cls.__name__ != cls_target.__name__:
                continue
            self._targets[idx] = cls
            break
        else:
            self._targets.append(cls)

    @dataclass_transform()
    def as_target(self, cls: type[_S]) -> type[_S]:
        self.add(cls)
        return cls

    def _target_from_dict(self, data: dict):
        for cls in reversed(self):
            try:
                return from_dict(cls, data)
            except (KeyError, TypeError, ValueError):
                pass
        raise TypeError(f"None of {self} can be deserialized from {data}")

    def _target_from_json(self, src: str):
        data = json.loads(src)
        return self._target_from_dict(data)

    @dataclass_transform()
    def as_source(self, cls: type[_S]) -> type[_S]:
        cls.from_dict = staticmethod(self._target_from_dict)  # type: ignore
        cls.from_json = staticmethod(self._target_from_json)  # type: ignore

        init_subclass = cls.__init_subclass__

        def __init_subclass__(cls, **kwargs):
            init_subclass(**kwargs)
            self.add(cls)

            # # Give the subclass its own registry, such that going down the mro
            # # will not add the subclass to the registry of the parent class.
            # cls = serializable_base(cls)

        cls.__init_subclass__ = classmethod(__init_subclass__)  # type: ignore

        return cls


def serializable_base(cls: type[_S]) -> type[_S]:
    """
    Decorator that adds a private serialization registry to the class, such that
    it will automatically register all subclasses of the class as targets for
    deserialization.
    """
    reg = SerializeRegistry()
    return reg.as_source(cls)


def _serializable(__cls, /):
    __cls.from_dict = classmethod(from_dict)  # type: ignore
    __cls.from_json = classmethod(from_json)  # type: ignore
    __cls.to_dict = to_dict
    __cls.to_json = to_json

    return __cls


def serializable(cls):
    """
    Decorator that adds the ``from_dict``, ``from_json``, ``to_dict`` and
    ``to_json`` methods to a class.
    """

    if TYPE_CHECKING:
        return type(cls.__name__, (Serializable, cls), {})
    else:
        return _serializable(cls)
