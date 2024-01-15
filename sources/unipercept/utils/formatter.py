from typing_extensions import override

__all__ = ["formatter"]


class formatter(property):
    """
    Implements a property that resolves a format string using the object's attributes.
    """

    __slots__ = ("fmt",)

    def __init__(self, fmt: str):
        self.fmt = fmt

    @override
    def __get__(self, obj, objtype=None):
        return self.fmt.format_map({"self": obj})

    @override
    def __set__(self, obj, value):
        raise AttributeError("Cannot set attribute")

    @override
    def __delete__(self, obj):
        raise AttributeError("Cannot delete attribute")
