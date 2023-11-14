from typing import Type

from torch import nn

from unipercept.nn.typings import Activation, Norm


def wrap_norm(name: str) -> Norm:
    from detectron2.layers import get_norm

    return lambda out_channels: get_norm(name, out_channels)


def wrap_activation(module: Type[nn.Module], **kwargs) -> Activation:
    def _new(**kwargs_ins):
        for key, value in kwargs_ins.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(
                    f"Conflicting values for `{key}` passed in wrapped and instantiation call: {kwargs[key]} vs {value}"
                )

        kwargs_ins.update(kwargs)

        return module(**kwargs_ins)

    return _new
